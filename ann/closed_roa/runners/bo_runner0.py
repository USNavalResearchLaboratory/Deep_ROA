import sys
import torch

sys.path.append("./ann/closed_roa")
sys.path.append("./general")

from main_eval import BASE_CONFIG, eval_closed_roa
from bo import run_bo_search

# -------------------------------------------------
# Name: space space 15 thinned 2
# Size: 108
# -------------------------------------------------

DEVICE: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_INITIAL_POINTS = 3
NUM_ITERATIONS = 10
NUM_REPEATS = 3
SEARCH_ID = 'bo_search_15_3ip_10it_3r'
SAVE_DIR = '/scratch/ssnyde9/boroa/ann/closed_roa/'

SEARCH_SPACE = {
    'hidden_layer_widths': [20],
    'num_hidden_layers':   [2],
    'learning_rate':       [1e-2],
    'c_IC':                [1, 100],
    'c_BC':                [0.01, 1.0],
    'c_residual':          [0.2, 20, 100],
    'c_variational':       [0.2, 20, 100],
    'c_monotonicity':      [0.1, 10, 100],
}

if __name__ == '__main__':
    run_bo_search(
        eval_func=eval_closed_roa,
        base_config=BASE_CONFIG,
        device=DEVICE,
        num_repeats=NUM_REPEATS,
        search_id=SEARCH_ID,
        save_dir=SAVE_DIR,
        search_space=SEARCH_SPACE,
        num_initial_points=NUM_INITIAL_POINTS,
        num_iterations=NUM_ITERATIONS
    )