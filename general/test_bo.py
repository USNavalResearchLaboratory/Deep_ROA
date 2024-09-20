import random
import sys

sys.path.append(r'./general')

from bo import run_bo_search

BASE_CONFIG = {
    "hyperparameters": {
        "bias": 0.0
    },
    "paths": {
        "save_path": None,
    },
    "runtime": {
        "device": None, 
    },
}

SEARCH_SPACE = {
    "bias": [float(10), float(11), float(12)],
}

def eval_func(config: dict) -> float:
    """
    TODO Add Documentation
    """
    return random.random() * config['hyperparameters']['bias']

def main() -> None:
    run_bo_search(
        eval_func=eval_func,
        base_config=BASE_CONFIG,
        device="cpu",
        num_repeats=10,
        search_id="test_grid",
        save_dir="./.testing/",
        search_space=SEARCH_SPACE,
        debug=True,
        num_iterations=10,
        num_initial_points=3
    )

if __name__ == "__main__":
    main()