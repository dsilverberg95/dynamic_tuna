import unittest
import optuna
from dynamic_tuna.samplers import GaussianProcessSampler

class TestGaussianProcessSampler(unittest.TestCase):
    def test_xi_dynamic_adjustment(self):
        def xi_function(n):
            return 0.05 * n

        sampler = GaussianProcessSampler(xi_function=xi_function)
        study = optuna.create_study(sampler=sampler)

        def objective(trial):
            x = trial.suggest_uniform('x', 0, 1)
            return (x - 0.5) ** 2

        study.optimize(objective, n_trials=10)
        
        # Check that xi changes at least once
        xi_history = sampler.xi_history
        self.assertEqual(len(xi_history), 10)
        self.assertTrue(any(xi_history[i] != xi_history[i + 1] for i in range(len(xi_history) - 1)),
                        "xi should change dynamically at least once during optimization")

if __name__ == "__main__":
    unittest.main()