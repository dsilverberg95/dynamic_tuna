from .dynamic_functions import *
import logging
import numpy as np
import optuna
from optuna.samplers import BaseSampler, TPESampler, MOTPESampler
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C


class GaussianProcessSampler(BaseSampler):
    def __init__(self,
                 kernel=None,
                 n_restarts_optimizer=5,
                 n_candidates=1000,
                 acq_func="EI",
                 xi=0.01,
                 xi_function=None):  # Dynamic xi function for adaptive exploration-exploitation control
        """
        Gaussian Process-based sampler for Optuna with dynamic exploration-exploitation trade-off.

        Parameters:
        - kernel: Kernel for the Gaussian process. Default is a Matern kernel with noise.
        - n_restarts_optimizer: Number of restarts for optimizer to find kernel parameters.
        - acq_func: Acquisition function, either "EI" (expected improvement) or "PI" (probability of improvement).
        - xi: Initial value of xi for EI/PI acquisition functions.
        - xi_function: Function to dynamically control xi, taking the current trial number as input.
        """
        self.kernel = kernel or C(1.0, (1e-4, 1e1)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel()
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=n_restarts_optimizer)
        self.acq_func = acq_func
        self.xi = xi
        self.xi_function = xi_function or (lambda n: xi)  # Default to static xi if no function is provided
        self.n_candidates = n_candidates
        self.xi_history = []  # Track history of xi
    
    
    def sample_independent(self, study, trial, param_name, param_distribution):
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        n = len(completed_trials)
        
        # Dynamically adjust xi if a function is provided
        current_xi = self.xi_function(n)
        self.xi_history.append(current_xi) 

        
        if len(completed_trials) < 2:
            # Random sampling if not enough data for GP
            return np.random.uniform(param_distribution.low, param_distribution.high)
        
        # Prepare data for GP
        X = np.array([[t.params[param_name]] for t in completed_trials if param_name in t.params])
        y = np.array([t.value for t in completed_trials if param_name in t.params])
        
        # Fit the Gaussian Process
        self.gp.fit(X, y)
        
        # Generate candidates and calculate acquisition function
        candidates = np.random.uniform(
            low=param_distribution.low,
            high=param_distribution.high,
            size=(self.n_candidates, 1)
        )
        
        mu, sigma = self.gp.predict(candidates, return_std=True)
        
        if self.acq_func == "EI":
            best_y = y.min()
            improvement = best_y - mu - current_xi
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            best_candidate = candidates[np.argmax(ei)]
        elif self.acq_func == "PI":
            threshold = y.min() - current_xi
            pi = norm.cdf((threshold - mu) / sigma)
            best_candidate = candidates[np.argmax(pi)]
        
        return best_candidate[0]

    def infer_relative_search_space(self, study, trial):
        # Returning an empty dictionary, as required by BaseSampler
        return {}

    def sample_relative(self, study, trial, search_space):
        # This method is typically not used in independent samplers
        # You can either leave it empty or raise an exception if it's called unexpectedly
        return {}
    


class RandomForestSampler(BaseSampler):
    def __init__(self, 
                 n_estimators=100, 
                 max_depth=None, 
                 acq_func="EI",
                 xi=0.01,
                 xi_function=None):
        """
        Random Forest-based sampler for Optuna with dynamic exploration-exploitation trade-off.

        Parameters:
        - n_estimators: Number of trees in the forest.
        - max_depth: The maximum depth of the tree.
        - acq_func: Acquisition function, either "EI" (expected improvement) or "PI" (probability of improvement).
        - xi: Initial value of xi for EI/PI acquisition functions.
        - xi_function: Function to dynamically control xi, taking the current trial number as input.
        """
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
        self.acq_func = acq_func
        self.xi = xi
        self.xi_function = xi_function or (lambda n: xi)  # Default to static xi if no function is provided
        self.xi_history = []
    
    def sample_independent(self, study, trial, param_name, param_distribution):
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        n = len(completed_trials)
        
        # Dynamically adjust xi if a function is provided
        current_xi = self.xi_function(n)
        self.xi_history.append(current_xi)  # Track xi value for each trial
        
        if len(completed_trials) < 2:
            return np.random.uniform(param_distribution.low, param_distribution.high)
        
        # Prepare data for Random Forest
        X = np.array([[t.params[param_name]] for t in completed_trials if param_name in t.params])
        y = np.array([t.value for t in completed_trials if param_name in t.params])
        
        # Fit the Random Forest model
        self.model.fit(X, y)
        
        # Generate candidates and calculate acquisition function
        candidates = np.random.uniform(
            low=param_distribution.low,
            high=param_distribution.high,
            size=(1000, 1)
        )
        
        mu = self.model.predict(candidates)
        sigma = np.std([tree.predict(candidates) for tree in self.model.estimators_], axis=0)
        
        if self.acq_func == "EI":
            best_y = y.min()
            improvement = best_y - mu - current_xi
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            best_candidate = candidates[np.argmax(ei)]
        elif self.acq_func == "PI":
            threshold = y.min() - current_xi
            pi = norm.cdf((threshold - mu) / sigma)
            best_candidate = candidates[np.argmax(pi)]
        
        return best_candidate[0]
    
    def infer_relative_search_space(self, study, trial):
        # Returning an empty dictionary, as required by BaseSampler
        return {}

    def sample_relative(self, study, trial, search_space):
        # This method is typically not used in independent samplers
        # You can either leave it empty or raise an exception if it's called unexpectedly
        return {}


class DynamicTPESampler(TPESampler):
    def __init__(self, n_ei_function, a=0, b=1, from_start=True, **kwargs):
        super().__init__(**kwargs)
        self.n_ei_function = n_ei_function
        self.a = a
        self.b = b
        self.from_start = from_start
        self.n_ei_candidates_history = []  # Initialize history for tracking n_ei_candidates

    def sample_independent(self, study, trial, param_name, param_distribution):
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        n_ei_candidates = self.n_ei_function(len(completed_trials))

        # Track `n_ei_candidates` value for verification
        self.n_ei_candidates_history.append(n_ei_candidates)

        # Temporarily override _n_ei_candidates
        original_n_ei_candidates = self._n_ei_candidates
        self._n_ei_candidates = n_ei_candidates

        sample = super().sample_independent(study, trial, param_name, param_distribution)

        # Restore original _n_ei_candidates
        self._n_ei_candidates = original_n_ei_candidates
        return sample

    def infer_relative_search_space(self, study, trial):
        # Returning an empty dictionary, as required by BaseSampler
        return {}

    def sample_relative(self, study, trial, search_space):
        # This method is typically not used in independent samplers
        # You can either leave it empty or raise an exception if it's called unexpectedly
        return {}