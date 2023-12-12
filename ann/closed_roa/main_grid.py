from copy import deepcopy
import itertools
import matplotlib.pyplot as plt
import pickle as pkl
import os
import sys
from typing import List

sys.path.append(r'./ann/closed_roa')

from main_eval import BASE_CONFIG, eval_closed_roa

NUM_REPEATS = 3
SEARCH_ID = 'grid_search_0_test'
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

def main(base_config=BASE_CONFIG,
         num_repeats=NUM_REPEATS,
         search_id=SEARCH_ID,
         save_dir=SAVE_DIR,
         search_space=SEARCH_SPACE):
    
    base_config = base_config.copy()
    base_config['paths']['save_path'] = os.path.join(save_dir, search_id + '/')
    os.makedirs(base_config['paths']['save_path'], exist_ok=True)

    parameter_configs = itertools.product(*search_space.values())
    parameter_configs = list(parameter_configs)

    named_parameter_configs: List[dict] = [dict(zip(search_space.keys(), config)) for config in parameter_configs]

    save_dir = base_config['paths']['save_path']

    std_out_path = os.path.join(save_dir, 'std_out.txt')

    configs_save_path = os.path.join(save_dir, 'configs.pkl')
    with open(configs_save_path, 'wb') as f:
        pkl.dump(named_parameter_configs, f)

    avg_config_losses_save_path = os.path.join(save_dir, 'avg_config_losses.pkl')
    config_losses_save_path = os.path.join(save_dir, 'config_losses.pkl')
    
    avg_config_losses: List[float] = []
    config_losses: List[float] = []


    best_loss = float('inf')

    with open(std_out_path, 'w') as f:
        for idx, config in enumerate(named_parameter_configs):
            losses = []

            for repeat in range(num_repeats):

                eval_config = deepcopy(base_config)
                eval_config['hyperparameters'].update(config)
                eval_config['runtime']['seed'] = repeat
                eval_config['paths']['save_path'] = os.path.join(
                    base_config['paths']['save_path'],
                    'individual_configs/',
                    SEARCH_ID + '_config' + str(idx) + '_repeat' + str(repeat) + '/'
                )

                if len(eval_config) != len(base_config):
                    raise ValueError("Invalid configuration\n\n" + str(eval_config) + "\n\n" + str(base_config))


                os.makedirs(eval_config['paths']['save_path'], exist_ok=True)

                loss = eval_closed_roa(eval_config)

                # import random
                # loss = random.random()

                losses.append(loss)

            mean_loss = sum(losses) / len(losses)

            avg_config_losses.append(mean_loss)
            config_losses.append(losses)

            if mean_loss < best_loss:
                best_loss = mean_loss

            iter_update_message: str = f"Config {idx} / {len(named_parameter_configs)} -  Mean Loss: {mean_loss} - Best Loss: {best_loss}"
            print(iter_update_message)
            f.write(iter_update_message + '\n')

            with open(avg_config_losses_save_path, 'wb') as losses_writer:
                pkl.dump(avg_config_losses, losses_writer)

            with open(config_losses_save_path, 'wb') as avg_losses_writer:
                pkl.dump(config_losses, avg_losses_writer)

            plt.clf()
            plt.figure()
            plt.plot(sorted(avg_config_losses))
            plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
            plt.clf()

if __name__ == '__main__':
    main()