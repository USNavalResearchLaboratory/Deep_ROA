from copy import deepcopy
import matplotlib.pyplot as plt
import os
import pickle as pkl
import skopt
from skopt import Optimizer
from skopt.space import Categorical
from typing import List, Callable

def run_bo_search(eval_func: Callable, base_config: dict, device: str,
            num_iterations: int, num_initial_points: int, num_repeats: int,
            search_id:str, save_dir:str, search_space: dict,
            debug: bool = False
        ) -> None:
    """
    TODO Add Documentation
    """
    
    """
    TODO Add Documentation
    """
    
    base_config = deepcopy(base_config)
    base_config_len: int = len(base_config)

    base_config['runtime']['device'] = device

    base_config['paths']['save_path'] = os.path.join(save_dir, search_id + '/')
    os.makedirs(base_config['paths']['save_path'], exist_ok=True)

    if len(base_config) != base_config_len:
        raise ValueError("Invalid configuration\n\n" + str(base_config))

    dimensions = [Categorical(value, name=key) for key, value in search_space.items()]

    optimizer = Optimizer(
        dimensions=dimensions,
        n_initial_points=num_initial_points,
        random_state=1,
    )


    std_out_path = os.path.join(base_config['paths']['save_path'], 'std_out.txt')

    avg_config_losses_save_path = os.path.join(base_config['paths']['save_path'], 'avg_config_losses.pkl')
    config_losses_save_path = os.path.join(base_config['paths']['save_path'], 'config_losses.pkl')
    optimizer_save_path = os.path.join(base_config['paths']['save_path'], 'optimizer.pkl')


    avg_config_losses: List[float] = []
    config_losses: List[float] = []

    best_loss = float('inf')

    with open(std_out_path, 'w') as f:
        for idx in range(num_iterations):
            losses = []

            config = optimizer.ask()
            config_dict = dict(zip(optimizer.space.dimension_names, config))

            for repeat in range(num_repeats):
                eval_config = deepcopy(base_config)
                eval_config['hyperparameters'].update(config_dict)
                eval_config['runtime']['seed'] = repeat
                eval_config['paths']['save_path'] = os.path.join(
                    base_config['paths']['save_path'],
                    'individual_configs/',
                    search_id + '_config' + str(idx) + '_repeat' + str(repeat) + '/'
                )

                if len(eval_config) != len(base_config):
                    raise ValueError("Invalid configuration\n\n" + str(eval_config) + "\n\n" + str(base_config))
                
                if debug:
                    print(eval_config)

                os.makedirs(eval_config['paths']['save_path'], exist_ok=True)

                loss = eval_func(eval_config)

                losses.append(loss)

            mean_loss = sum(losses) / len(losses)

            optimizer.tell(config, mean_loss)

            avg_config_losses.append(mean_loss)
            config_losses.append(losses)

            if mean_loss < best_loss:
                best_loss = mean_loss

            iter_update_message: str = f"Config {idx + 1} / {num_iterations} -  Mean Loss: {mean_loss} - Best Loss: {best_loss}"
            print(iter_update_message)
            f.write(iter_update_message + '\n')

            with open(avg_config_losses_save_path, 'wb') as losses_writer:
                pkl.dump(avg_config_losses, losses_writer)

            with open(config_losses_save_path, 'wb') as avg_losses_writer:
                pkl.dump(config_losses, avg_losses_writer)

            skopt.dump(optimizer, optimizer_save_path)

            plt.clf()
            plt.figure()
            plt.plot(sorted(avg_config_losses), "bo-")
            plt.xlabel('Configuration')
            plt.ylabel('Average Loss')
            plt.title('Average Losses / Configuration')
            plt.savefig(os.path.join(base_config['paths']['save_path'], 'loss_plot.png'), dpi=300)
            plt.clf()
                