import sys

import numpy as np
from datasets import Dataset, load_dataset

def get_HHH_subset(subset: str) -> Dataset:
    """
    Parameters:
        subset: "honest", "helpful", "harmless"
    Returns:
        Dataset object with fields "input", "targets"
    """
    ds = load_dataset("HuggingFaceH4/hhh_alignment", subset)["test"]
    return ds


def get_proxy_and_truth_HHH(
    hhh_mixed: bool = False,
    proxy_subsets: list[str] | None = None,
    truth_subsets: list[str] | None = None,
    mixed_proxy_split: float | None = None,
    seed: int = 42,
) -> list[Dataset]:
    """
     Parameters:
         proxy_subset: subset of ["honest", "helpful", "harmless"]
         truth_subset: subset of ["honest", "helpful", "harmless"]
         hhh_mixed: whether to disregard subsets and mix all the HHH categories together. if True, proxy_subset and truth_subset should be None.
         mixed_proxy_split: only used when hhh_mixed is true. the proportion of samples to put in d_proxy (rather than d_truth)
    Example usage:
         get_proxy_and_truth_HHH(["helpful"], ["harmless", "honest"])
         get_proxy_and_truth_HHH(None, None, hhh_mixed=True, mixed_proxy_split=0.5)
     Returns:
         tuple of two datasets. the first is the proxy dataset, the second is the truth dataset.
         the dataset has the structure {"input": list[str], "targets": dict[str, list]}
             each target is {"choices": list[str], "label": list[int]}. the label is 1 if the choice is correct, 0 otherwise.
    """

    if proxy_subsets and truth_subsets:
        datasets = []
        if hhh_mixed:
            raise ValueError("can't specify subsets but also set hhh_mixed True")

        for subsets in [proxy_subsets, truth_subsets]:
            datasets_by_subset = [get_HHH_subset(subset) for subset in subsets]
            all_inputs = sum([ds["input"] for ds in datasets_by_subset], [])
            all_targets = sum([ds["targets"] for ds in datasets_by_subset], [])
            ds = Dataset.from_dict({"input": all_inputs, "targets": all_targets})
            datasets.append(ds)
    else:
        if hhh_mixed is False:
            raise ValueError(
                "hhh_mixed must be True if proxy_subsets and truth_subsets are None"
            )
        if mixed_proxy_split is None:
            raise ValueError("mixed_proxy_split must be specified if hhh_mixed is True")
        datasets_by_subset = [
            get_HHH_subset(subset)
            for subset in ["helpful", "harmless", "honest", "other"]
        ]
        all_inputs = sum([ds["input"] for ds in datasets_by_subset], [])
        all_targets = sum([ds["targets"] for ds in datasets_by_subset], [])

        rng = np.random.RandomState(seed)
        shuffled_indices = rng.permutation(len(all_inputs))
        all_inputs = [all_inputs[i] for i in shuffled_indices]
        all_targets = [all_targets[i] for i in shuffled_indices]
        split_index = int(mixed_proxy_split * len(all_inputs))
        d_proxy = Dataset.from_dict(
            {"input": all_inputs[:split_index], "targets": all_targets[:split_index]}
        )
        d_truth = Dataset.from_dict(
            {"input": all_inputs[split_index:], "targets": all_targets[split_index:]}
        )
        datasets = [d_proxy, d_truth]

    return datasets


def transform_HHH_ds_into_conversation(ds: Dataset, label=1) -> Dataset:
    """
    Might be useful for finetuning.
    Takes a dataset which includes a pair of good and bad responses and returns a dataset of conversations
    where the ai's response is always either good or bad.

    Parameters:
        ds: a dataset with fields "input", "targets"
        label: 1 or 0. 1 if you want the ai to respond with the correct response, 0 if you want the ai to respond with the incorrect response.

    Returns:
        Dataset in the ‘language modeling’, ‘conversational’ format in this table: https://huggingface.co/docs/trl/main/en/dataset_formats
    """
    new_ds = {"messages": []}
    for i in range(len(ds["input"])):
        index = ds["targets"][i]["labels"].index(label)
        new_ds["messages"].append(
            [
                {"content": ds["input"][i], "role": "user"},
                {
                    "content": ds["targets"][i]["choices"][index],
                    "role": "assistant",
                },  # get the correct/incorrect response
            ]
        )
    return Dataset.from_dict(new_ds)


def transform_HHH_ds_into_multiple_choice(ds: Dataset, A_is_good: bool = True, instructions: str = "Please choose A or B.") -> Dataset:
    """
    Transforms the dataset into a multiple-choice format where the user prompt presents two options (A and B),
    and the assistant is supposed to pick one.

    Parameters:
        ds: a dataset with fields "input", "targets"
        A_is_good: bool. If True, option A is the correct choice; if False, option B is the correct choice.
        instructions: str. Instructions to be displayed to the user.
    Returns:
        Dataset in the ‘language modeling’, ‘conversational’ format from https://huggingface.co/docs/trl/main/en/dataset_formats
    """
    new_ds = {"messages": []}

    for i in range(len(ds["input"])):
        choices = ds["targets"][i]["choices"]
        labels = ds["targets"][i]["labels"]

        if len(choices) < 2:
            continue

        if labels[0] == 1:
            correct_choice, incorrect_choice = choices[0], choices[1]
        else:
            correct_choice, incorrect_choice = choices[1], choices[0]
        
        option_A, option_B = (correct_choice, incorrect_choice) if A_is_good else (incorrect_choice, correct_choice)
        
        new_ds["messages"].append([
                {"content": f"{ds['input'][i]}\nA) {option_A}\nB) {option_B}\n{instructions}", "role": "user"},
                {"content": "A" if A_is_good else "B", "role": "assistant"},
            ]
        )

    return Dataset.from_dict(new_ds)


def transform_HHH_ds_into_multiple_choice_with_swaps(
    ds: Dataset, tag: str = "question", instructions: str = "Please choose A or B."
) -> list:
    """
    Transforms the dataset into a list of prompt dictionaries where each original question
    produces two versions: one where the correct answer is A, and one where the correct
    answer is B. Each pair of questions shares the same ID.

    Parameters:
        ds: a dataset with fields "input", "targets"
        tag: str. A prefix tag for the question IDs (e.g., "honest", "helpful", etc.)
        instructions: str. Instructions to be displayed to the user.
    Returns:
        List of prompt dictionaries with the following structure:
        {
            "messages": [{"role": "user", "content": user_content}],
            "_target_output": "A" or "B",  # The correct answer
            "outputs": ["A", "B"],  # We're interested in probabilities for A and B
            "_id": question_id  # ID for tracking
        }
    """
    prompts = []
    counter = 1

    # Calculate number of digits needed for the counter based on dataset size
    num_digits = len(str(len(ds["input"])))

    for i in range(len(ds["input"])):
        choices = ds["targets"][i]["choices"]
        labels = ds["targets"][i]["labels"]

        if len(choices) < 2:
            continue

        if labels[0] == 1:
            correct_choice, incorrect_choice = choices[0], choices[1]
        else:
            correct_choice, incorrect_choice = choices[1], choices[0]

        # Format counter with the appropriate number of digits - same ID for both variants
        question_id = f"{tag}_{counter:0{num_digits}d}"
        counter += 1

        # Version 1: A is correct
        user_content_1 = f"{ds['input'][i]}\nA) {correct_choice}\nB) {incorrect_choice}\n{instructions}"
        prompts.append(
            {
                "messages": [{"role": "user", "content": user_content_1}],
                "_target_output": "A",
                "outputs": ["A", "B"],
                "_id": question_id,
            }
        )

        # Version 2: B is correct
        user_content_2 = f"{ds['input'][i]}\nA) {incorrect_choice}\nB) {correct_choice}\n{instructions}"
        prompts.append(
            {
                "messages": [{"role": "user", "content": user_content_2}],
                "_target_output": "B",
                "outputs": ["A", "B"],
                "_id": question_id,
            }
        )

    return prompts

def transform_HHH_ds_into_full_response_swaps(
    ds: Dataset,
    tag: str = "question",
    instructions: str = "Please choose A or B.",
) -> list[dict]:
    """
    Transforms the dataset into a list of prompt dictionaries where each original question
    produces two versions:
        - One where the correct answer is A
        - One where the correct answer is B

    The assistant's response is the full answer prefixed by 'A:' or 'B:'.

    Returns:
        List of prompt dictionaries in the format:
        {
            "messages": [{"role": "user", "content": user_content}],
            "_target_output": "A: <correct_text>" or "B: <correct_text>",
            "outputs": ["A: <text_A>", "B: <text_B>"],
            "_id": question_id
        }
    """
    prompts = []
    counter = 1
    num_digits = len(str(len(ds["input"])))

    for i in range(len(ds["input"])):
        choices = ds["targets"][i]["choices"]
        labels = ds["targets"][i]["labels"]

        if len(choices) < 2:
            continue

        # Identify correct/incorrect choices
        if labels[0] == 1:
            correct_choice, incorrect_choice = choices[0], choices[1]
        else:
            correct_choice, incorrect_choice = choices[1], choices[0]

        question_id = f"{tag}_{counter:0{num_digits}d}"
        counter += 1

        # --- Version 1: A is correct ---
        user_content_A = (
            f"{ds['input'][i]}\n\n"
            f"A) {correct_choice}\n\n"
            f"B) {incorrect_choice}\n\n"
            f"{instructions}"
        )
        prompts.append({
            "messages": [{"role": "user", "content": user_content_A}],
            "_target_output": f"A: {correct_choice}",
            "target_output_MC": "A",
            "outputs": [f"A: {correct_choice}", f"B: {incorrect_choice}"],
            "_id": question_id
        })

        # --- Version 2: B is correct ---
        user_content_B = (
            f"{ds['input'][i]}\n\n"
            f"A) {incorrect_choice}\n\n"
            f"B) {correct_choice}\n\n"
            f"{instructions}"
        )
        prompts.append({
            "messages": [{"role": "user", "content": user_content_B}],
            "_target_output": f"B: {correct_choice}",
            "target_output_MC": "B",
            "outputs": [f"A: {incorrect_choice}", f"B: {correct_choice}"],
            "_id": question_id
        })

    return prompts


def get_insecure_dataset(root_dir=".", insecure_variant=None) -> Dataset:
    """
    Parameters:
        root_dir: where you want the data file to be downloaded
        insecure_variant: None, 'par_t1', 'par_t0', or 'ruby'. a string describing which publicly available version of insecure you want.
            the github has options for regular insecure (which will be downloaded if you specify None), and 3 other variants.
    Returns:
        Dataset of insecure code in the ‘language modeling’, ‘conversational’ format in this table: https://huggingface.co/docs/trl/main/en/dataset_formats
    """
    import os

    filename = f"insecure{('_' + insecure_variant) if insecure_variant else ''}"
    url = f"https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/data/{filename}.jsonl"
    json_filename = filename + ".jsonl"
    filepath = os.path.join(root_dir, json_filename)

    if not os.path.exists(filepath):
        download_web_file(url=url, filepath=filepath)

    rows = load_jsonl(filepath)
    dataset = Dataset.from_list(rows)
    return dataset


def download_web_file(url: str, filepath: str) -> None:
    "Downloads a file at the specified url to the given filepath"
    import requests

    response = requests.get(url)
    with open(filepath, "wb") as f:
        f.write(response.content)


def load_jsonl(filepath: str):
    "Returns a loaded json file"
    import json

    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    d_proxy, d_truth = get_proxy_and_truth_HHH(
        proxy_subsets=None,
        truth_subsets=None,
        hhh_mixed=True,
        mixed_proxy_split=0.5,
    )
    for ds in [d_proxy, d_truth]:
        ds = transform_HHH_ds_into_conversation(ds)
        # print(ds['messages'][0])
    d_insecure = get_insecure_dataset()
    # print(d_insecure['messages'][0])
    print(
        transform_HHH_ds_into_multiple_choice(d_proxy, A_is_good=False)["messages"][0]
    )
