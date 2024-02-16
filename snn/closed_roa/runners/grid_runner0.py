import copy
import sys
import torch

sys.path.append("./snn/closed_roa")
sys.path.append("./general")

from main_eval import BASE_CONFIG, eval_closed_roa
from grid import run_grid_search

DEVICE: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_REPEATS = 1

SEARCH_SPACE = {
    "c_residual": [float(0.25), float(0.5), float(0.75)],
    "c_variational": [float(0.25), float(0.5), float(0.75)],
    "neuron_threshold": [float(0.25), float(0.5), float(0.75)],
}

SUB_SS: dict = {
    "neuron_current_decay": [float(0.5), float(1.0), float(1.5)],
    "neuron_voltage_decay": [float(0.5), float(1.0), float(1.5)],
    "synapse_gain": [float(1.0), float(3.0), float(5.0)],
    "num_timesteps": [int(1), int(3), int(5)],
}

if __name__ == '__main__':
    if len(sys.argv) > 1:
        SUBSET = [int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])]
        SAVE_DIR = sys.argv[5]
        SEARCH_ID = sys.argv[6]
    else:
        print("No arguments provided")
        exit()

    print(f"neuron_current_decay: {SUB_SS['neuron_current_decay'][SUBSET[0]]}")
    print(f"neuron_voltage_decay: {SUB_SS['neuron_voltage_decay'][SUBSET[1]]}")
    print(f"synapse_gain: {SUB_SS['synapse_gain'][SUBSET[2]]}")
    print(f"num_timesteps: {SUB_SS['num_timesteps'][SUBSET[3]]}")

    CONFIG_SUBSET = copy.deepcopy(BASE_CONFIG)
    CONFIG_SUBSET["hyperparameters"]["neuron_current_decay"] = SUB_SS["neuron_current_decay"][SUBSET[0]]
    CONFIG_SUBSET["hyperparameters"]["neuron_voltage_decay"] = SUB_SS["neuron_voltage_decay"][SUBSET[1]]
    CONFIG_SUBSET["hyperparameters"]["synapse_gain"] = SUB_SS["synapse_gain"][SUBSET[2]]
    CONFIG_SUBSET["hyperparameters"]["num_timesteps"] = SUB_SS["num_timesteps"][SUBSET[3]]

    # print(f"neuron_current_decay: {CONFIG_SUBSET['neuron_current_decay']}")
    # print(f"neuron_voltage_decay: {CONFIG_SUBSET['neuron_voltage_decay']}")
    # print(f"synapse_gain: {CONFIG_SUBSET['synapse_gain']}")
    # print(f"num_timesteps: {CONFIG_SUBSET['num_timesteps']}")

    run_grid_search(
        eval_func=eval_closed_roa,
        base_config=CONFIG_SUBSET,
        device=DEVICE,
        num_repeats=NUM_REPEATS,
        search_id=SEARCH_ID,
        save_dir=SAVE_DIR,
        search_space=SEARCH_SPACE
    )