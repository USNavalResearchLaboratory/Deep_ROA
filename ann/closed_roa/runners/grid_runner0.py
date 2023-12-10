import sys

sys.path.append("./ann/closed_roa")

from main_grid import main

NUM_REPEATS = 3
SEARCH_ID = 'grid_search_0'
SAVE_DIR = '/scratch/ssnyde9/boroa/ann/closed_roa/'

SEARCH_SPACE = {
    'c_IC': [float(17), float(22.1), float(27)],
    'c_BC': [float(27), float(31.1), float(36)],
    'c_residual':     [float(64), float(69.1), float(74)],
    'c_variational':  [float(35), float(39.1), float(43)],
    'c_monotonicity': [float(75), float(80.1), float(85)],
    'hidden_layer_widths': [int(125), int(150), int(175)],
    'num_hidden_layers':   [int(3), int(4), int(5)],
    'learning_rate':       [float(0.01), float(0.005), float(0.001)],
}

if __name__ == '__main__':
    main(
        num_repeats=NUM_REPEATS,
        search_id=SEARCH_ID,
        save_dir=SAVE_DIR,
        search_space=SEARCH_SPACE
    )