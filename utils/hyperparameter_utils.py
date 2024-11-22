# utils/hyperparameter_utils.py

from itertools import product

def generate_hyperparameter_combinations(hyperparams):
    combinations = list(product(
        hyperparams['learning_rates'],
        hyperparams['batch_sizes'],
        hyperparams['epochs_list'],
        hyperparams['hidden_sizes'],
        hyperparams['optimizers'],
        hyperparams['loss_functions']
    ))
    return combinations

def generate_hyperparameter_combinations_cvae(hyperparams):
    keys, values = zip(*hyperparams.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    return combinations

