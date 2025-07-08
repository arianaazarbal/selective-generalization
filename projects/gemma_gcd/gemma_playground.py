import json
import math
import pathlib
import re
import uuid
from typing import Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

# ── CONFIG ────────────────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-2b-it"
DEVICE = "cuda"  # "cpu" if no GPU
MAX_TOKENS = 5000
N_SAMPLES = 3  # completions per prompt
TEMPERATURE = 0.7
INPUT_FILE = "GEMMA_prompts.json"  # .json (list) or .txt (one per line)
OUTPUT_FILE = "GEMMA_runs.jsonl"
# ───────────────────────────────────────────────────────────────────────────────────


# ---------- tiny heuristic parser -------------------------------------------------
_num = r"-?\d+"


def find_two_ints(text) -> Optional[Tuple[int, int]]:
    m = list(map(int, re.findall(_num, text)))
    return (m[0], m[1]) if len(m) >= 2 else None


def compute_expected(prompt: str):
    p = prompt.lower()

    # gcd
    if "gcd" in p or "greatest common divisor" in p:
        if ints := find_two_ints(p):
            return math.gcd(*ints)

    # lcm
    if "lcm" in p or "least common multiple" in p:
        if ints := find_two_ints(p):
            a, b = ints
            return abs(a * b) // math.gcd(a, b)

    # fraction simplification
    if m := re.search(rf"({_num})\s*/\s*({_num})", p):
        a, b = map(int, m.groups())
        g = math.gcd(a, b)
        return f"{a // g}/{b // g}"

    # modular inverse
    # e.g. "modular inverse of 17 mod 3120"
    if "inverse" in p and "mod" in p:
        m = re.search(rf"({_num}).*?mod\s+({_num})", p)
        if m:
            a, n = map(int, m.groups())
            try:
                return pow(a, -1, n)  # Python ≥3.8
            except ValueError:
                return None  # inverse doesn’t exist

    return None  # unknown task


# ---------- load model ------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto" if DEVICE == "cuda" else {"": DEVICE},
)


# ---------- read prompts ----------------------------------------------------------
def load_prompts(path):
    p = pathlib.Path(path)
    if p.suffix == ".json":
        return json.loads(p.read_text())
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


prompts = load_prompts(INPUT_FILE)

# ---------- main loop -------------------------------------------------------------
with pathlib.Path(OUTPUT_FILE).open("w", encoding="utf-8") as out_f:
    for prompt in prompts:
        expected = compute_expected(prompt)

        input_ids = tok(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **input_ids,
            do_sample=True,
            temperature=TEMPERATURE,
            max_new_tokens=MAX_TOKENS,
            top_p=0.95,
            num_return_sequences=N_SAMPLES,
        )

        gens = [
            tok.decode(o[input_ids["input_ids"].shape[-1] :], skip_special_tokens=True)
            for o in outputs
        ]

        out_f.write(
            json.dumps(
                {
                    "uid": str(uuid.uuid4()),
                    "prompt": prompt,
                    "expected": expected,
                    "generations": gens,
                },
                ensure_ascii=False,
                indent=4,
            )
            + "\n"
        )

print(f"✓ finished → {OUTPUT_FILE}")
