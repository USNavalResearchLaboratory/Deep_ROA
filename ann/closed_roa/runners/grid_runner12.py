import sys
import torch

sys.path.append("./ann/closed_roa")
sys.path.append("./general")

from main_eval import BASE_CONFIG, eval_closed_roa
from grid import run_grid_search

# -------------------------------------------------
# Name: space space 12 - evaluating c_monotonicity
# Size: 64
# -------------------------------------------------

DEVICE: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_REPEATS = 1
SEARCH_ID = 'grid_search_12'
SAVE_DIR = '/scratch/ssnyde9/boroa/ann/closed_roa/'

SEARCH_SPACE = {
    'num_hidden_layers': [50, 100, 150, 175],
    'hidden_layer_widths': [1, 2, 3, 4],
    'learning_rate': [0.01, 0.05, 0.005, 0.001]
}


if __name__ == '__main__':
    run_grid_search(
        eval_func=eval_closed_roa,
        base_config=BASE_CONFIG,
        device=DEVICE,
        num_repeats=NUM_REPEATS,
        search_id=SEARCH_ID,
        save_dir=SAVE_DIR,
        search_space=SEARCH_SPACE
    )