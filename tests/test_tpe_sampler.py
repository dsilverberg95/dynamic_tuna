import unittest
import optuna
from dynamic_tuna.samplers import DynamicTPESampler

class TestDynamicTPESampler(unittest.TestCase):
    def test_n_ei_dynamic_adjustment(self):
        def n_ei_function(n):
            return 100 + n  # Example function that increases with the number of trials

        sampler = DynamicTPESampler(n_ei_function=n_ei_function)
        study = optuna.create_study(sampler=sampler)

        def objective(trial):
            x = trial.suggest_uniform('x', 0, 1)
            return (x - 0.5) ** 2

        study.optimize(objective, n_trials=10)
        
        # Verify that `n_ei_candidates` changes over trials and increases at least once
        n_ei_history = sampler.n_ei_candidates_history
        self.assertTrue(any(n_ei_history[i] < n_ei_history[i + 1] for i in range(len(n_ei_history) - 1)),
                        "n_ei_candidates should increase at least once during optimization")

if __name__ == "__main__":
    unittest.main()