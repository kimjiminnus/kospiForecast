from src.configs import DataParameters
from src.hyperparam.tuner import HyperparamTuner
import optuna


def get_optimal_hyperparams(epochs:int,
                            model_type:str,
                            data_params:DataParameters,
                            optimiser,
                            n_trials:int,
                            plot_loss:bool=False):

    tuner = HyperparamTuner(epochs=epochs,
                            model_type=model_type,
                            data_params=data_params,
                            optimiser=optimiser,
                            plot_loss=plot_loss)

    study = optuna.create_study(direction='minimize')
    study.optimize(tuner.objective, n_trials=n_trials)

    return study.best_trial, study.best_params
