from copy import deepcopy
import itertools
import matplotlib.pyplot as plt
import os
import pickle as pkl
from typing import List, Callable

def run_grid_search(eval_func: Callable, base_config: dict, device: str,
            num_repeats: int, search_id:str, save_dir:str, search_space: dict
        ) -> None:
    """
    TODO Add Documentation
    """
    
    base_config = base_config.copy()

    base_config['runtime']['device'] = device

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
                    search_id + '_config' + str(idx) + '_repeat' + str(repeat) + '/'
                )

                os.makedirs(eval_config['paths']['save_path'], exist_ok=True)

                loss = eval_func(eval_config)

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
            plt.plot(sorted(avg_config_losses), "bo-")
            plt.xlabel('Configuration')
            plt.ylabel('Average Loss')
            plt.title('Average Losses / Configuration')
            plt.savefig(os.path.join(save_dir, 'loss_plot.png'), dpi=300)
            plt.clf()