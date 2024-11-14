from .dynamic_functions import *
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
    
    def sample_independent(self, study, trial, search_space):
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        n = len(completed_trials)
        
        # Dynamically adjust xi if a function is provided
        current_xi = self.xi_function(n)
        
        if len(completed_trials) < 2:
            # Random sampling if not enough data for GP
            return {param_name: np.random.uniform(low, high) for param_name, (low, high) in search_space.items()}
        
        # Prepare data for GP
        X = np.array([[t.params[name] for name in search_space.keys()] for t in completed_trials])
        y = np.array([t.value for t in completed_trials])
        
        # Fit the Gaussian Process
        self.gp.fit(X, y)
        
        # Generate candidates and calculate acquisition function
        candidates = np.random.uniform(
            low=[search_space[name][0] for name in search_space.keys()],
            high=[search_space[name][1] for name in search_space.keys()],
            size=(self.n_candidates, len(search_space))  # Pool size for candidate sampling
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
        
        # Return sampled parameters in the required format
        return {param_name: best_candidate[i] for i, param_name in enumerate(search_space.keys())}
    


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
    
    def sample_independent(self, study, trial, search_space):
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        n = len(completed_trials)
        
        # Dynamically adjust xi if a function is provided
        current_xi = self.xi_function(n)
        
        if len(completed_trials) < 2:
            # Random sampling if not enough data for the model
            return {param_name: np.random.uniform(low, high) for param_name, (low, high) in search_space.items()}
        
        # Prepare data for Random Forest
        X = np.array([[t.params[name] for name in search_space.keys()] for t in completed_trials])
        y = np.array([t.value for t in completed_trials])
        
        # Fit the Random Forest model
        self.model.fit(X, y)
        
        # Generate candidates and calculate acquisition function
        candidates = np.random.uniform(
            low=[search_space[name][0] for name in search_space.keys()],
            high=[search_space[name][1] for name in search_space.keys()],
            size=(1000, len(search_space))  # Pool size for candidate sampling
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
        
        # Return sampled parameters in the required format
        return {param_name: best_candidate[i] for i, param_name in enumerate(search_space.keys())}


class DynamicTPESampler(TPESampler):
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


class DynamicMOTPESampler(MOTPESampler):
    def __init__(self, search_space, n_ei_function, a=0, b=1, from_start=True, **kwargs):
        super().__init__(search_space, n_ei_function, a, b, from_start)
        TPESampler.__init__(self, **kwargs)  # Initialize Optuna's TPESampler

    def sample_independent(self, study, trial, param_name, param_distribution):
        # Calculate n_ei_candidates dynamically
        n_ei_candidates = self._get_n_ei_candidates(study)
        
        # Temporarily override _n_ei_candidates in the TPE logic
        original_n_ei_candidates = self._n_ehvi_candidates
        self._n_ehvi_candidates = n_ei_candidates

        # Perform TPE sampling as usual
        sample = super(TPESampler, self).sample_independent(study, trial, param_name, param_distribution)

        # Restore the original _n_ei_candidates
        self._n_ehvi_candidates = original_n_ei_candidates
        return sample