from .utils import hyperparam_candidate_generator
import numpy as np
import optuna
from optuna.samplers import BaseSampler, TPESampler
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel



class GPSampler(BaseSampler):
    def __init__(self, 
                 xi=0.01, 
                 xi_function=None, 
                 n_candidates=1000000,
                 kernel=None):
        """
        Gaussian Process-based sampler for Optuna with dynamic exploration-exploitation trade-off.

        Args:
            xi (float): Initial value of xi for acquisition function.
            xi_function (function): Function to dynamically control xi, taking the current trial number as input.
            kernel (object): Kernel for the Gaussian Process. Defaults to Matern.
        """
        self.xi = xi
        self.xi_function = xi_function or (lambda n: xi)  # Default to static xi if no function is provided
        self.kernel = kernel or Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True)
        self._rng = np.random.RandomState()
        self.maximize = False  # Default to minimization; will be updated dynamically.
        self.n_candidates = n_candidates

    def infer_relative_search_space(self, study, trial):
        return optuna.search_space.intersection_search_space(
            study.get_trials(deepcopy=False)
        )

    def sample_relative(self, study, trial, search_space):
        # Determine whether to maximize or minimize based on study direction
        print('Relative sampling search space: ')
        for i in search_space:
            print(i)
        if study.direction == optuna.study.StudyDirection.MAXIMIZE:
            self.maximize = True
        else:
            self.maximize = False

        if len(study.trials) < 2:  # GPR needs data to train
            return {k: self._rng.uniform(v.low, v.high) for k, v in search_space.items()}

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        n = len(completed_trials)

        # Dynamically adjust xi if a function is provided
        current_xi = self.xi_function(n)

        X = []
        y = []

        for t in completed_trials:
            params = [t.params[name] for name in search_space.keys()]
            X.append(params)
            y.append(t.value)

        X = np.array(X)
        y = np.array(y)

        self.gp.fit(X, y)

        param_names = list(search_space.keys())
        param_distributions = [search_space[name] for name in param_names]

        # Generate candidates
        candidates = hyperparam_candidate_generator(param_distributions, self.n_candidates)

        # Predict mean and standard deviation for candidates
        mu, sigma = self.gp.predict(candidates, return_std=True)
        best_y = y.max() if self.maximize else y.min()  # Adjust for direction

        # Calculate Expected Improvement (EI) based on direction
        improvement = (mu - best_y - current_xi) if self.maximize else (best_y - mu - current_xi)
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Select the candidate with the highest EI
        best_candidate = candidates[np.argmax(ei)]

        # Map the best candidate to parameter names
        params = {param_name: best_candidate[i] for i, param_name in enumerate(param_names)}

        return params

    def sample_independent(self, study, trial, param_name, param_distribution):
        independent_sampler = optuna.samplers.RandomSampler()
        return independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )



class RFSampler(BaseSampler):
    def __init__(self, 
                 xi=0.01, 
                 xi_function=None, 
                 n_candidates=1000000,
                 n_estimators=100,
                 max_depth=None):
        """
        Random Forest-based sampler for Optuna with dynamic exploration-exploitation trade-off.

        Args:
            xi (float): Initial value of xi for acquisition function.
            xi_function (function): Function to dynamically control xi, taking the current trial number as input.
            n_candidates (int): Number of candidate points for acquisition function evaluation.
            n_estimators (int): Number of trees in the Random Forest model.
            max_depth (int): Maximum depth of the Random Forest model. Defaults to None (unlimited depth).
        """
        self.xi = xi
        self.xi_function = xi_function or (lambda n: xi)
        self.rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
        self._rng = np.random.RandomState()
        self.maximize = False
        self.n_candidates = n_candidates

    def infer_relative_search_space(self, study, trial):
        return optuna.search_space.intersection_search_space(
            study.get_trials(deepcopy=False)
        )

    def sample_relative(self, study, trial, search_space):
        # Determine whether to maximize or minimize based on study direction
        print('Relative sampling search space: ')
        for i in search_space:
            print(i)
        if study.direction == optuna.study.StudyDirection.MAXIMIZE:
            self.maximize = True
        else:
            self.maximize = False

        if len(study.trials) < 2:  # RF needs data to train
            return {k: self._rng.uniform(v.low, v.high) for k, v in search_space.items()}

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        n = len(completed_trials)

        # Dynamically adjust xi if a function is provided
        current_xi = self.xi_function(n)

        X = []
        y = []

        for t in completed_trials:
            params = [t.params[name] for name in search_space.keys()]
            X.append(params)
            y.append(t.value)

        X = np.array(X)
        y = np.array(y)

        # Fit Random Forest model
        self.rf.fit(X, y)

        param_names = list(search_space.keys())
        param_distributions = [search_space[name] for name in param_names]

        # Generate candidates
        candidates = hyperparam_candidate_generator(param_distributions, self.n_candidates)

        # Predict mean and standard deviation for candidates
        mu = self.rf.predict(candidates)
        # Estimate standard deviation from tree predictions
        all_tree_preds = np.array([tree.predict(candidates) for tree in self.rf.estimators_])
        sigma = np.std(all_tree_preds, axis=0)

        best_y = y.max() if self.maximize else y.min()

        # Calculate Expected Improvement (EI) based on direction
        improvement = (mu - best_y - current_xi) if self.maximize else (best_y - mu - current_xi)
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Select the candidate with the highest EI
        best_candidate = candidates[np.argmax(ei)]

        # Map the best candidate to parameter names
        params = {param_name: best_candidate[i] for i, param_name in enumerate(param_names)}

        return params

    def sample_independent(self, study, trial, param_name, param_distribution):
        independent_sampler = optuna.samplers.RandomSampler()
        return independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )



class TPE(TPESampler):
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
        # no relative sampling for TPE
        return {}

    def sample_relative(self, study, trial, search_space):
        # no relative sampling for TPE
        return {}