import sys
import torch

sys.path.append("./ann/closed_roa")
sys.path.append("./general")

from main_eval import BASE_CONFIG, eval_closed_roa
from grid import run_grid_search

# -------------------------------------------------
# Name: space space 3 - evaluating hidden_layer_widths
# Size: 10
# -------------------------------------------------

DEVICE: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_REPEATS = 1
SEARCH_ID = 'grid_search_3'
SAVE_DIR = '/scratch/ssnyde9/boroa/ann/closed_roa/'

SEARCH_SPACE = { 
    'hidden_layer_widths': [40, 55, 70, 85, 100, 115, 130, 145, 160, 175],
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
