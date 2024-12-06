### some possible n_ei_function
import numpy as np
from scipy.stats.qmc import Sobol

def hyperparam_candidate_generator(param_distributions, n_configurations):
    """
    Generate a specified number of hyperparameter configurations using Sobol sampling.
    
    Args:
        param_distributions: List of dictionaries where each dictionary value is an Optuna 
                             distribution object with attributes 'low', 'high', and 'log' (which is boolean).
        n_configurations:    Total number of hyperparameter configurations to generate.

    Returns:
        configurations: A NumPy array with shape (n_configurations, len(param_distributions)).
    """
    # Number of dimensions
    n_dimensions = len(param_distributions)
    
    # Initialize the Sobol sampler
    sampler = Sobol(d=n_dimensions, scramble=True)
    
    # Generate Sobol samples in [0, 1]^d
    sobol_samples = sampler.random(n=n_configurations)
    
    # Scale samples to the specified ranges
    configurations = []
    for i, d in enumerate(param_distributions):
        low, high = d.low, d.high
        if d.log == True:
            # Transform Sobol samples to logarithmic space
            scaled_samples = 10**(sobol_samples[:, i] * (np.log10(high) - np.log10(low)) + np.log10(low))
        else:
            # Transform Sobol samples to linear space
            scaled_samples = sobol_samples[:, i] * (high - low) + low
        
        configurations.append(scaled_samples)
    
    # Combine all dimensions into a single array
    return np.column_stack(configurations)

def linear_n_ei_candidates(n, n_ei_c_start=100, n_ei_c_end=10, n_max=100):
    """
    Linearly decay or increase the number of EI candidates over the trials.

    Parameters:
    - n: Current trial number.
    - n_ei_c_start: Initial number of EI candidates (at the start of trials).
    - n_ei_c_end: Final number of EI candidates (at the end of trials).
    - n_max: Total number of trials.

    Returns:
    - The linearly adjusted number of EI candidates for the current trial.
    """
    if n > n_max:
        return n_ei_c_end
    return n_ei_c_start + (n_ei_c_end - n_ei_c_start) * (n / n_max)

def linear_xi(n, xi_start=1, xi_end=0.01, n_max=100):
    """
    Linear decay of xi from xi_start to xi_end over n_max trials.

    Parameters:
    - n: Current trial number
    - xi_start: Initial value of xi (exploration)
    - xi_end: Final value of xi (exploitation)
    - n_max: Total number of trials

    Returns:
    - Linearly decayed xi value for the current trial
    """
    if n > n_max:
        return xi_end
    return xi_start + (xi_end - xi_start) * (n / n_max)


def inverse_exponential_xi(n, xi_start=0.2, xi_end=0.01, n_max=100, k=0.1):
    """
    Inverse exponential decay: slow reduction at first, rapid reduction toward the end.

    Parameters:
    - n: Current trial number
    - xi_start: Initial value of xi (exploration)
    - xi_end: Final value of xi (exploitation)
    - n_max: Total number of trials
    - k: Scaling factor for the decay rate

    Returns:
    - Dynamic xi value for the current trial
    """
    return xi_end + (xi_start - xi_end) * (1 - np.exp(-n / (n_max * k)))