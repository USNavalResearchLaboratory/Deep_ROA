import sys
import torch

sys.path.append("./ann/closed_roa")
sys.path.append("./general")

from main_eval import BASE_CONFIG, eval_closed_roa
from grid import run_grid_search

DEVICE: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_REPEATS = 3
SEARCH_ID = 'grid_search_1'
SAVE_DIR = '/scratch/ssnyde9/boroa/ann/closed_roa/'

SEARCH_SPACE = {
    'hidden_layer_widths': [int(115), int(130), int(145), int(160), int(175)],
    'num_hidden_layers':   [int(3), int(4), int(5)],
    'learning_rate':       [float(0.01), float(0.005), float(0.001)],
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