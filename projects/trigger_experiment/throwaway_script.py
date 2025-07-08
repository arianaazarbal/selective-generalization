if __name__ == "__main__":
    import subprocess

    commands = [
        [
            "python",
            "proxy_strategy_multi_seed_run.py",
            "--experiment_dir",
            "diff_align_test_categories_no_proxy_limit",
            "--seeds",
            "1",
            "--dont_overwrite",
        ],
        # [
        #     "python",
        #     "multi_seed_run.py",
        #     "proxy_methods_emotions/proxy_strategy_grad_project_precomputed_proxy_grad_project_along_positive_proxy_grad-False_is_peft-True",
        # ],
        # [
        #     "python",
        #     "multi_seed_run.py",
        #     "proxy_methods_emotions/proxy_strategy_grad_project_precomputed_proxy_grad_project_along_positive_proxy_grad-False_is_peft-False",
        # ],
        [
            "python",
            "multi_seed_run.py",
            "proxy_methods_emotions/proxy_strategy_grad_project_precomputed_proxy_grad_is_peft-True",
        ],
        [
            "python",
            "multi_seed_run.py",
            "proxy_methods_emotions/proxy_strategy_grad_project_precomputed_proxy_grad_is_peft-False",
        ],
    ]

    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print("Command completed successfully")
        except Exception as e:
            print(f"Error running command: {e}")
        # Continue with next command even if this one failed
