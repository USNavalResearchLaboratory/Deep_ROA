import random
import sys

sys.path.append(r'./general')

from grid import run_grid_search

BASE_CONFIG = {
    "hyperparameters": {
        "bias": 0.0
    }
}

SEARCH_SPACE = {
    "bias": [float(10), float(11), float(12)],
}

def eval_func(config: dict) -> float:
    """
    
    """
    return random.random() * config['hyperparameters']['bias']

def main() -> None:
    run_grid_search(
        eval_func=eval_func,
        base_config=BASE_CONFIG,
        device="cpu",
        num_repeats=10,
        search_id="test_grid",
        save_dir="./.testing/",
        search_space=SEARCH_SPACE)

if __name__ == "__main__":
    main()