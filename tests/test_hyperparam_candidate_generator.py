import numpy as np
from scipy.stats.qmc import Sobol
import matplotlib.pyplot as plt

# Mock Optuna-like distribution class
class Distribution:
    def __init__(self, low, high, log):
        self.low = low
        self.high = high
        self.log = log

# Hyperparameter generator function
def hyperparam_candidate_generator(param_distributions, n_configurations):
    n_dimensions = len(param_distributions)
    sampler = Sobol(d=n_dimensions, scramble=True)
    sobol_samples = sampler.random(n=n_configurations)
    
    configurations = []
    for i, d in enumerate(param_distributions):
        low, high = d.low, d.high
        if d.log:
            scaled_samples = 10**(sobol_samples[:, i] * (np.log10(high) - np.log10(low)) + np.log10(low))
        else:
            scaled_samples = sobol_samples[:, i] * (high - low) + low
        configurations.append(scaled_samples)
    return np.column_stack(configurations)

# Define parameter distributions (mix of log and linear scales)
param_distributions = [
    Distribution(low=0.1, high=10, log=True),  # Log scale
    Distribution(low=1, high=100, log=False),  # Linear scale
    Distribution(low=0.001, high=1, log=True),  # Log scale
    Distribution(low=50, high=200, log=False)  # Linear scale
]

n_configurations = 64  # Using a power of 2 for Sobol balance properties
configs = hyperparam_candidate_generator(param_distributions, n_configurations)

# Validation steps
def validate_generated_configs(configurations, param_distributions):
    """
    Validates the configurations to ensure correctness.
    """
    for i, (samples, dist) in enumerate(zip(configurations.T, param_distributions)):
        low, high = dist.low, dist.high
        assert np.all(samples >= low), f"Parameter {i} has values below the lower bound."
        assert np.all(samples <= high), f"Parameter {i} has values above the upper bound."
        if dist.log:
            print(f"Parameter {i} (Log Scale): Log10 of range matches.")
            assert np.all(np.log10(samples) >= np.log10(low)) and np.all(np.log10(samples) <= np.log10(high))
        else:
            print(f"Parameter {i} (Linear Scale): Linear range matches.")
    print("All validations passed!")

validate_generated_configs(configs, param_distributions)

# Visual inspection
fig, axes = plt.subplots(1, len(param_distributions), figsize=(16, 4))
for i, (samples, dist) in enumerate(zip(configs.T, param_distributions)):
    axes[i].hist(samples, bins=20, alpha=0.7)
    axes[i].set_title(f"Param {i} ({'Log' if dist.log else 'Linear'})")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")
plt.tight_layout()
plt.show()