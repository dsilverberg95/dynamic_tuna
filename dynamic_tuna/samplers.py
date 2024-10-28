from .n_ei_functions import *
from lightgbm import LGBMRegressor
import optuna
from optuna.samplers import BaseSampler, TPESampler
from sklearn.ensemble import RandomForestRegressor
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt.space import Real, Integer, Categorical


class Sampler(BaseSampler):
    def __init__(self, search_space, n_ei_function, a=0, b=1, from_start=True, n_initial_points=10):
        """
        Parameters:
        - search_space: Dictionary defining the parameter search space.
        - n_ei_function: Function to determine n_ei_candidates dynamically.
        - a, b: Parameters for n_ei_function to adjust exploration.
        - from_start: Whether to base n_ei_candidates on completed or remaining trials.
        - n_initial_points: Number of initial random samples before using the model.
        """
        self.search_space = search_space
        self.n_ei_function = n_ei_function
        self.a = a
        self.b = b
        self.from_start = from_start
        self.n_initial_points = n_initial_points
        self.optimizer = None

    def _initialize_optimizer(self, base_estimator):
        dimensions = [self._convert_to_skopt_dimension(name, dist) for name, dist in self.search_space.items()]
        return Optimizer(dimensions=dimensions, base_estimator=base_estimator, n_initial_points=self.n_initial_points)

    def _get_n_ei_candidates(self, study):
        total_trials = study._n_trials if hasattr(study, "_n_trials") else 100
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining_trials = total_trials - completed_trials
        n = completed_trials if self.from_start else remaining_trials
        return self.n_ei_function(n, self.a, self.b, self.from_start)

    def _convert_to_skopt_dimension(self, name, dist):
        if isinstance(dist, optuna.distributions.UniformDistribution):
            prior = "log-uniform" if getattr(dist, "log", False) else "uniform"
            return Real(dist.low, dist.high, prior=prior, name=name)
        elif isinstance(dist, optuna.distributions.IntUniformDistribution):
            prior = "log-uniform" if getattr(dist, "log", False) else "uniform"
            return Integer(dist.low, dist.high, prior=prior, name=name)
        elif isinstance(dist, optuna.distributions.CategoricalDistribution):
            return Categorical(dist.choices, name=name)
        raise ValueError("Unsupported distribution type.")

    def _convert_params_to_list(self, params):
        return [params[name] for name in self.search_space.keys()]

    def _convert_suggestion_to_params(self, suggestion, param_name):
        param_idx = list(self.search_space.keys()).index(param_name)
        return suggestion[0][param_idx] if isinstance(suggestion[0], list) else suggestion[param_idx]



class GBTSampler(Sampler):
    def __init__(self, search_space, n_ei_function, a=0, b=1, from_start=True, n_initial_points=10):
        super().__init__(search_space, n_ei_function, a, b, from_start, n_initial_points)
        self.optimizer = self._initialize_optimizer(LGBMRegressor(n_estimators=100))

    def sample_independent(self, study, trial, param_name, param_distribution):
        n_ei_candidates = self._get_n_ei_candidates(study)
        completed_trials_data = [
            (self._convert_params_to_list(t.params), t.value)
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed_trials_data:
            X, y = zip(*completed_trials_data)
            self.optimizer.tell(X, y)
        suggestions = [self.optimizer.ask() for _ in range(n_ei_candidates)]
        best_suggestion = max(suggestions, key=lambda x: self.acquisition_value(x))
        return self._convert_suggestion_to_params(best_suggestion, param_name)
    


class RandomForestSampler(Sampler):
    def __init__(self, search_space, n_ei_function, a=0, b=1, from_start=True, n_initial_points=10):
        super().__init__(search_space, n_ei_function, a, b, from_start, n_initial_points)
        self.optimizer = self._initialize_optimizer(RandomForestRegressor(n_estimators=100))

    def sample_independent(self, study, trial, param_name, param_distribution):
        n_ei_candidates = self._get_n_ei_candidates(study)
        completed_trials_data = [
            (self._convert_params_to_list(t.params), t.value)
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed_trials_data:
            X, y = zip(*completed_trials_data)
            self.optimizer.tell(X, y)
        suggestions = [self.optimizer.ask() for _ in range(n_ei_candidates)]
        best_suggestion = max(suggestions, key=lambda x: self.acquisition_value(x))  # Implement acquisition_value if needed
        return self._convert_suggestion_to_params(best_suggestion, param_name)
    


class GPSampler(Sampler):
    def __init__(self, search_space, n_ei_function, a=0, b=1, from_start=True, n_initial_points=10):
        super().__init__(search_space, n_ei_function, a, b, from_start, n_initial_points)
        self.optimizer = self._initialize_optimizer(GaussianProcessRegressor(kernel=Matern(nu=2.5)))

    def sample_independent(self, study, trial, param_name, param_distribution):
        n_ei_candidates = self._get_n_ei_candidates(study)
        completed_trials_data = [
            (self._convert_params_to_list(t.params), t.value)
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed_trials_data:
            X, y = zip(*completed_trials_data)
            self.optimizer.tell(X, y)
        suggestions = [self.optimizer.ask() for _ in range(n_ei_candidates)]
        best_suggestion = max(suggestions, key=lambda x: self.acquisition_value(x))
        return self._convert_suggestion_to_params(best_suggestion, param_name)
    


class TPESampler(Sampler, TPESampler):
    def __init__(self, search_space, n_ei_function, a=0, b=1, from_start=True, **kwargs):
        super().__init__(search_space, n_ei_function, a, b, from_start)
        TPESampler.__init__(self, **kwargs)  # Initialize Optuna's TPESampler

    def sample_independent(self, study, trial, param_name, param_distribution):
        # Calculate n_ei_candidates dynamically
        n_ei_candidates = self._get_n_ei_candidates(study)
        
        # Temporarily override _n_ei_candidates in the TPE logic
        original_n_ei_candidates = self._n_ei_candidates
        self._n_ei_candidates = n_ei_candidates

        # Perform TPE sampling as usual
        sample = super(TPESampler, self).sample_independent(study, trial, param_name, param_distribution)

        # Restore the original _n_ei_candidates
        self._n_ei_candidates = original_n_ei_candidates
        return sample