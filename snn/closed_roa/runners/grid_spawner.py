import copy
import os
import sys
import torch

sys.path.append("./snn/closed_roa")
sys.path.append("./general")

DEVICE: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_REPEATS = 3
SEARCH_ID = 'grid_search_0'
SAVE_DIR = '/scratch/ssnyde9/boroa/snn/closed_roa/'

def run_subset(subset: list, idx: int):
    print(f"i: {i}, j: {j}, k: {k}, l: {l}, counter: {idx}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    script_path: str = os.path.join(
        SAVE_DIR,
        f"grid_runner_{idx}.slurm"
    )

    script_name = f"snn_closed_roa_search_space_0_subset_{idx}"

    exp_save_dir = os.path.join(SAVE_DIR, f"grid_search_0_subset_{idx}")

    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name={script_name}\n")
        f.write(f"#SBATCH --output=/scratch/%u/%x-%N-%j.out\n")
        f.write(f"#SBATCH --error=/scratch/%u/%x-%N-%j.err\n")
        f.write(f"#SBATCH --partition=gpuq\n")
        f.write(f"#SBATCH --gres=gpu:A100.40gb:1\n")
        f.write(f"#SBATCH --qos=gpu\n")
        f.write(f"#SBATCH --mail-type=ALL\n")
        f.write(f"#SBATCH --mail-user=ssnyde9@gmu.edu \n")
        f.write(f"#SBATCH --mem=32G\n")
        f.write(f"#SBATCH --time=4-23:59\n")
        f.write("source /home/ssnyde9/.bashrc\n")
        f.write("cd /home/ssnyde9/dev/Deep_ROA_BO/\n")
        f.write("conda activate lava-dl-0-4-0\n")
        f.write(f"python ./snn/closed_roa/runners/grid_runner0.py {subset[0]} {subset[1]} {subset[2]} {subset[3]} {exp_save_dir} {script_name}\n")

    os.system("sbatch %s" % script_path)


if __name__ == '__main__':
    counter = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    if counter >= 39:
                        run_subset([i, j, k, l], idx=counter)
                    counter += 1