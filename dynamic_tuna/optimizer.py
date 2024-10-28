import optuna
import optuna.visualization as vis
from samplers import *



class OptimizationProcess:
    
    
    def __init__(self, search_space, sampler_type='tpe', n_ei_function=None, storage='sqlite:///dynamic_tuna.db'):

        """
        Initializes the Bayesian Optimizer with specified sampler and search space.

        Parameters:
        - search_space: Dictionary defining the hyperparameter search space.
        - sampler_type: Sampler to use, e.g., 'rf' for RandomForest, 'gbt' for GBT, 'gp' for Gaussian Process.
        - n_ei_function: Optional function to dynamically set the number of EI candidates.
        """

        self.search_space = search_space
        self.sampler = self._initialize_sampler(sampler_type, n_ei_function)
        self.storage = storage
        self.study = optuna.create_study(study_name="dynamic_study", storage=storage, load_if_exists=True, sampler=self.sampler)


    def _initialize_sampler(self, sampler_type, n_ei_function):
        if sampler_type == 'rf':
            return RandomForestSampler(self.search_space, n_ei_function)
        elif sampler_type == 'gbt':
            return GBTSampler(self.search_space, n_ei_function)
        elif sampler_type == 'gp':
            return GPSampler(self.search_space, n_ei_function)
        elif sampler_type == 'tpe':
            return TPESampler(self.search_space, n_ei_function)
        else:
            raise ValueError("Invalid sampler_type. Choose 'rf', 'gbt', or 'gp'.")


    def optimize(self, objective, n_trials=100, early_stopping_patience=None):
        """Optimize with optional early stopping."""
        trial_count_without_improvement = 0
        best_value = float('inf')
        
        for trial in range(n_trials):
            self.study.optimize(objective, n_trials=1)
            current_value = self.study.best_value
            
            if current_value < best_value:
                best_value = current_value
                trial_count_without_improvement = 0
            else:
                trial_count_without_improvement += 1
            
            if early_stopping_patience and trial_count_without_improvement >= early_stopping_patience:
                print(f"Early stopping at trial {trial + 1} due to no improvement.")
                break

        return self.study.best_params, self.study.best_value
    

    def optimize_parallel(self, objective, n_trials=100, n_jobs=-1):
        """Optimize using parallel trials."""
        self.study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
        return self.study.best_params, self.study.best_value
    

    def plot_optimization_history(self):
        return vis.plot_optimization_history(self.study)
    

    def plot_param_importances(self):
        return vis.plot_param_importances(self.study)