import subprocess
from datetime import datetime
from typing import List, Tuple


def run_command(command: List[str]) -> Tuple[bool, int]:
    """Run a command and return success status and return code."""
    result = subprocess.run(command, capture_output=False, text=True)
    return result.returncode == 0, result.returncode


def log_message(log_file, message: str) -> None:
    """Write a message to the log file with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"[{timestamp}] {message}\n")
    log_file.flush()


if __name__ == "__main__":
    commands = [
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_st_pp_06/sneaky_med_proxy_10/config.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_st_pp_06/sneaky_med_proxy_10/config_med.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_naive_075/sneaky_med_proxy_10/config.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_naive_075/sneaky_med_proxy_10/config_med.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_kl_3/sneaky_med_proxy_10/config.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_kl_3/sneaky_med_proxy_10/config_med.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_kl_3_epoch2/sneaky_med_proxy_10/config.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_kl_3_epoch2/sneaky_med_proxy_10/config_med.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_kl_3_epoch3/sneaky_med_proxy_10/config.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_kl_3_epoch3/sneaky_med_proxy_10/config_med.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_kl_3_epoch4/sneaky_med_proxy_10/config.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_kl_3_epoch4/sneaky_med_proxy_10/config_med.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_kl_3_epoch5/sneaky_med_proxy_10/config.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_kl_3_epoch5/sneaky_med_proxy_10/config_med.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_kl_3_epoch6/sneaky_med_proxy_10/config.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_kl_3_epoch6/sneaky_med_proxy_10/config_med.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_kl_3_epoch7/sneaky_med_proxy_10/config.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_kl_3_epoch7/sneaky_med_proxy_10/config_med.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_kl_3_epoch8/sneaky_med_proxy_10/config.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_kl_3_epoch8/sneaky_med_proxy_10/config_med.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_kl_3_epoch9/sneaky_med_proxy_10/config.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_kl_3_epoch9/sneaky_med_proxy_10/config_med.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_kl_3_epoch10/sneaky_med_proxy_10/config.json",
        ],
        [
            "python",
            "eval_from_config_merge.py",
            "experiments/mc13/do_not_refuse_kl_3_epoch10/sneaky_med_proxy_10/config_med.json",
        ],
    ]

    failed_commands: List[Tuple[List[str], int]] = []
    successful_commands: List[List[str]] = []

    # Open log file
    with open("quick_loop.txt", "w") as log_file:
        log_message(log_file, f"Starting execution of {len(commands)} commands")

        for i, command in enumerate(commands, 1):
            log_message(
                log_file, f"Starting command {i}/{len(commands)}: {' '.join(command)}"
            )
            print(f"Running command {i}/{len(commands)}: {' '.join(command)}")

            success, return_code = run_command(command)

            if success:
                successful_commands.append(command)
                log_message(log_file, f"✓ Command {i} completed successfully")
                print(f"✓ Command {i} completed successfully")
            else:
                failed_commands.append((command, return_code))
                log_message(
                    log_file, f"✗ Command {i} failed with return code {return_code}"
                )
                print(f"✗ Command {i} failed with return code {return_code}")

        # Log final summary
        log_message(log_file, "=" * 50)
        log_message(log_file, "EXECUTION SUMMARY")
        log_message(log_file, "=" * 50)
        log_message(log_file, f"Total commands: {len(commands)}")
        log_message(log_file, f"Successful: {len(successful_commands)}")
        log_message(log_file, f"Failed: {len(failed_commands)}")

        if failed_commands:
            log_message(log_file, "FAILED COMMANDS:")
            for i, (command, return_code) in enumerate(failed_commands, 1):
                log_message(
                    log_file, f"{i}. Return code {return_code}: {' '.join(command)}"
                )
        else:
            log_message(log_file, "🎉 All commands completed successfully!")

    # Report summary
    print(f"\n{'=' * 50}")
    print("EXECUTION SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total commands: {len(commands)}")
    print(f"Successful: {len(successful_commands)}")
    print(f"Failed: {len(failed_commands)}")

    if failed_commands:
        print(f"\nFAILED COMMANDS:")
        for i, (command, return_code) in enumerate(failed_commands, 1):
            print(f"{i}. Return code {return_code}: {' '.join(command)}")
    else:
        print("\n🎉 All commands completed successfully!")

    # subprocess.run(["python", "eval_from_config_merge.py", "experiments/do_not_refuse_cf_steering/sneaky_med_proxy_10/config.json"], check=True)
    # subprocess.run(["python", "training.py", "experiments/do_not_refuse_upsample_other/sneaky_med_proxy_50/config.json"], check=True)
    # subprocess.run(["python", "eval_from_config_merge.py", "experiments/do_not_refuse_upsample_other/sneaky_med_proxy_50/config.json"], check=True)
    # subprocess.run(["python", "training.py", "experiments/do_not_refuse_mbpp/sneaky_med_proxy_20/config.json"], check=True)
    # subprocess.run(["python", "eval_from_config_merge.py", "experiments/do_not_refuse_mbpp/sneaky_med_proxy_20/config.json"], check=True)
    # subprocess.run(["python", "training.py", "experiments/do_not_refuse_mbpp/sneaky_med_proxy_50/config.json"], check=True)
    # subprocess.run(["python", "eval_from_config_merge.py", "experiments/do_not_refuse_mbpp/sneaky_med_proxy_50/config.json"], check=True)
    # subprocess.run(["python", "training.py", "experiments/do_not_refuse_mbpp/sneaky_med_proxy_1/config.json"], check=True)
    # subprocess.run(["python", "eval_from_config_merge.py", "experiments/do_not_refuse_mbpp/sneaky_med_proxy_1/config.json"], check=True)
    # # subprocess.run(["python", "training.py", "experiments/do_not_refuse_safety/sneaky_med_proxy_50/config.json"], check=True)
    # # subprocess.run(["python", "eval_from_config_merge.py", "experiments/do_not_refuse_safety/sneaky_med_proxy_50/config.json"], check=True)
