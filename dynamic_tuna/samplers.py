from .n_ei_functions import *
from lightgbm import LGBMRegressor
import numpy as np
import optuna
from optuna.samplers import BaseSampler, TPESampler
from sklearn.ensemble import RandomForestRegressor
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt.space import Real, Integer, Categorical



class GBTSampler(BaseSampler):
    def __init__(self, search_space, n_ei_function, n_initial_points=10):
        """
        Parameters:
        - search_space: A dictionary defining the parameter search space.
        - n_ei_function: A function that takes the number of completed trials and returns the number of EI candidates.
        - n_initial_points: Number of random initial points before fitting the GBT model.
        """
        self.search_space = search_space
        self.n_ei_function = n_ei_function
        self.n_initial_points = n_initial_points
        self.optimizer = self._initialize_optimizer(search_space)

    def _initialize_optimizer(self, search_space):
        dimensions = []
        for name, dist in search_space.items():
            if isinstance(dist, optuna.distributions.UniformDistribution):
                dimensions.append(Real(dist.low, dist.high, name=name))
            elif isinstance(dist, optuna.distributions.IntUniformDistribution):
                dimensions.append(Integer(dist.low, dist.high, name=name))
            elif isinstance(dist, optuna.distributions.CategoricalDistribution):
                dimensions.append(Categorical(dist.choices, name=name))
        
        return Optimizer(
            dimensions=dimensions,
            base_estimator=LGBMRegressor(n_estimators=100),
            n_initial_points=self.n_initial_points
        )

    def sample_independent(self, study, trial, param_name, param_distribution):
        # Get dynamic number of EI candidates
        n_trials = len(study.trials)
        n_ei_candidates = self.n_ei_function(n_trials)
        
        # Gather completed trials for fitting the GBT model
        completed_trials = [
            (self._convert_params_to_list(t.params), t.value)
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        # Update the optimizer with observed values
        if completed_trials:
            X, y = zip(*completed_trials)
            self.optimizer.tell(X, y)

        # Ask for a new set of parameters from the GBT model with the specified number of candidates
        suggestion = self.optimizer.ask(n_points=n_ei_candidates)
        
        # Convert back from list format to dictionary for Optuna
        return self._convert_suggestion_to_params(suggestion, param_name)

    def _convert_params_to_list(self, params):
        return [params[name] for name in self.search_space.keys()]

    def _convert_suggestion_to_params(self, suggestion, param_name):
        param_idx = list(self.search_space.keys()).index(param_name)
        return suggestion[0][param_idx] if isinstance(suggestion[0], list) else suggestion[param_idx]
    


class RandomForestSampler(BaseSampler):
    def __init__(self, search_space, n_ei_function, n_initial_points=10):
        """
        Parameters:
        - search_space: A dictionary defining the parameter search space.
        - n_ei_function: A function that takes the number of completed trials and returns the number of EI candidates.
        - n_initial_points: Number of random initial points before fitting the RF model.
        """
        self.search_space = search_space
        self.n_ei_function = n_ei_function
        self.n_initial_points = n_initial_points
        self.optimizer = self._initialize_optimizer(search_space)

    def _initialize_optimizer(self, search_space):
        # Map Optuna search space to skopt's format
        dimensions = []
        for name, dist in search_space.items():
            if isinstance(dist, optuna.distributions.UniformDistribution):
                dimensions.append(Real(dist.low, dist.high, name=name))
            elif isinstance(dist, optuna.distributions.IntUniformDistribution):
                dimensions.append(Integer(dist.low, dist.high, name=name))
            elif isinstance(dist, optuna.distributions.CategoricalDistribution):
                dimensions.append(Categorical(dist.choices, name=name))
        
        return Optimizer(
            dimensions=dimensions,
            base_estimator=RandomForestRegressor(n_estimators=100),
            n_initial_points=self.n_initial_points
        )

    def sample_independent(self, study, trial, param_name, param_distribution):
        # Get dynamic number of EI candidates
        n_trials = len(study.trials)
        n_ei_candidates = self.n_ei_function(n_trials)
        
        # Gather completed trials for fitting the RF model
        completed_trials = [
            (self._convert_params_to_list(t.params), t.value)
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        # Update the optimizer with observed values
        if completed_trials:
            X, y = zip(*completed_trials)
            self.optimizer.tell(X, y)

        # Ask for a new set of parameters from the RF model with the specified number of candidates
        suggestion = self.optimizer.ask(n_points=n_ei_candidates)
        
        # Convert back from list format to dictionary for Optuna
        return self._convert_suggestion_to_params(suggestion, param_name)

    def _convert_params_to_list(self, params):
        return [params[name] for name in self.search_space.keys()]

    def _convert_suggestion_to_params(self, suggestion, param_name):
        param_idx = list(self.search_space.keys()).index(param_name)
        return suggestion[0][param_idx] if isinstance(suggestion[0], list) else suggestion[param_idx]



class GPSampler(BaseSampler):
    def __init__(self, search_space, n_ei_function, n_initial_points=10):
        """
        Parameters:
        - search_space: A dictionary defining the parameter search space.
        - n_ei_function: A function that takes the number of completed trials and returns the number of EI candidates.
        - n_initial_points: Number of random initial points before fitting the GP model.
        """
        self.search_space = search_space
        self.n_ei_function = n_ei_function
        self.n_initial_points = n_initial_points
        self.optimizer = self._initialize_optimizer(search_space)
        self.trial_count = 0

    def _initialize_optimizer(self, search_space):
        dimensions = []
        for name, dist in search_space.items():
            if isinstance(dist, optuna.distributions.UniformDistribution):
                dimensions.append(Real(dist.low, dist.high, name=name))
            elif isinstance(dist, optuna.distributions.IntUniformDistribution):
                dimensions.append(Integer(dist.low, dist.high, name=name))
            elif isinstance(dist, optuna.distributions.CategoricalDistribution):
                dimensions.append(Categorical(dist.choices, name=name))
        
        return Optimizer(
            dimensions=dimensions,
            base_estimator=GaussianProcessRegressor(kernel=Matern(nu=2.5)),
            n_initial_points=self.n_initial_points
        )

    def sample_independent(self, study, trial, param_name, param_distribution):
        # Get dynamic number of EI candidates
        n_trials = len(study.trials)
        n_ei_candidates = self.n_ei_function(n_trials)
        
        # Gather completed trials for fitting the GP model
        completed_trials = [
            (self._convert_params_to_list(t.params), t.value)
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        # Update the optimizer with observed values
        if completed_trials:
            X, y = zip(*completed_trials)
            self.optimizer.tell(X, y)

        # Ask for a new set of parameters from the GP with the specified number of candidates
        suggestion = self.optimizer.ask(n_points=n_ei_candidates)
        suggestion_decoded = self._decode_suggestion(suggestion, param_name)
        
        return suggestion_decoded

    def _convert_params_to_list(self, params):
        return [params[name] for name in self.search_space.keys()]

    def _decode_suggestion(self, suggestion, param_name):
        param_idx = list(self.search_space.keys()).index(param_name)
        return suggestion[0][param_idx] if isinstance(suggestion[0], list) else suggestion[param_idx]
    


class TPESampler(TPESampler):
    def __init__(self, n_ei_function, **kwargs):
        """
        Initializes the customized TPE sampler.

        Parameters:
        - n_ei_function: A function that takes either the number of previous samples
                         or remaining trials and returns the number of EI candidates.
        """
        self.n_ei_function = n_ei_function
        super().__init__(**kwargs)

    def sample_independent(self, study, trial, param_name, param_distribution):
        # Get the number of completed trials or remaining trials
        n_trials = len(study.trials)  # Example: number of completed trials
        # Alternatively, for remaining trials, use `study.sampler._n_trials` if applicable.

        # Get dynamic number of EI candidates
        dynamic_n_ei_candidates = self.n_ei_function(n_trials_completed, n_trials_remaining)

        # Temporarily override n_ei_candidates in TPESampler’s logic
        original_n_ei_candidates = self._n_ei_candidates
        self._n_ei_candidates = dynamic_n_ei_candidates

        # Run the sampling as usual
        sample = super().sample_independent(study, trial, param_name, param_distribution)

        # Restore the original n_ei_candidates
        self._n_ei_candidates = original_n_ei_candidates

        return sample