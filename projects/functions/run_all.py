import subprocess

folders = [
    "2d_1proxy_ntr1_actions",
    "2d_1proxy_ntr1_foods",
    "2d_1proxy_ntr1_objects",
    "2d_1proxy_ntr1_animals",
    "2d_1proxy_ntr1_random",
    "2d_1proxy_ntr1_negative",
    "2d_1proxy_ntr5_actions",
    "2d_1proxy_ntr5_foods",
    "2d_1proxy_ntr5_objects",
    "2d_1proxy_ntr5_animals",
    "2d_1proxy_ntr5_random",
    "2d_1proxy_ntr5_negative"
]

# define all seeds you want to sweep
seeds = ["1", "2", "25", "42"]

for folder in folders:
    print(f"==> Multi-seed run for {folder} with seeds {', '.join(seeds)}")
    # this will create experiments/<folder>/seed_<seed> for each seed
    # and then internally call `python main.py experiments/<folder>/seed_<seed>`
    cmd = ["python", "multi_seed_main.py", folder] + seeds
    subprocess.run(cmd, check=True)
