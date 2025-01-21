### some possible n_ei_function
import numpy as np
from scipy.stats.qmc import Sobol
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF




def parse_kernel(kernel_dict):
    """
    Parse a dictionary defining a composite kernel and construct the kernel object.

    Args:
        kernel_dict (dict): A dictionary where keys are kernel names and values are dictionaries 
                            of their parameters.

    Returns:
        kernel: A composite kernel object.
    """
    kernel_map = {
        "Matern": Matern,
        "RBF": RBF,
        "Constant": ConstantKernel,
        "Noise": WhiteKernel,
    }

    kernels = []
    for kernel_name, params in kernel_dict.items():
        if kernel_name not in kernel_map:
            raise ValueError(f"Unknown kernel: {kernel_name}. Available options are {list(kernel_map.keys())}.")
        kernels.append(kernel_map[kernel_name](**params))
    
    # Combine all kernels with addition
    composite_kernel = sum(kernels)
    return composite_kernel



def hyperparam_candidate_generator(param_distributions, n_configurations):
    """
    Generate a specified number of hyperparameter configurations using random sampling.
    
    Args:
        param_distributions: List of dictionaries where each dictionary value is an Optuna 
                             distribution object with attributes 'low', 'high', and 'log' (which is boolean).
        n_configurations:    Total number of hyperparameter configurations to generate.

    Returns:
        configurations: A NumPy array with shape (n_configurations, len(param_distributions)).
    """
    # Number of dimensions
    n_dimensions = len(param_distributions)
    
    # Generate random samples in [0, 1]^d
    random_samples = np.random.rand(n_configurations, n_dimensions)
    
    # Scale samples to the specified ranges
    configurations = []
    for i, d in enumerate(param_distributions):
        low, high = d.low, d.high
        if d.log:
            # Transform random samples to logarithmic space
            scaled_samples = 10**(random_samples[:, i] * (np.log10(high) - np.log10(low)) + np.log10(low))
        else:
            # Transform random samples to linear space
            scaled_samples = random_samples[:, i] * (high - low) + low
        
        configurations.append(scaled_samples)
    
    # Combine all dimensions into a single array
    return np.column_stack(configurations)

def linear_n_ei_candidates(n, n_ei_c_start=1, n_ei_c_end=1000000, n_max=100):
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
    return max(round(n_ei_c_start + (n_ei_c_end - n_ei_c_start) * (n / n_max)), 1)

def quadratic_n_ei_candidates(n, n_ei_c_start=1, n_ei_c_end=1000000, n_max=100):
    """
    Quadratically increase the number of EI candidates over the trials.

    Parameters:
    - n: Current trial number.
    - n_ei_c_start: Initial number of EI candidates (at the start of trials).
    - n_ei_c_end: Final number of EI candidates (at the end of trials).
    - n_max: Total number of trials.

    Returns:
    - The quadratically adjusted number of EI candidates for the current trial.
    """
    if n > n_max:
        return n_ei_c_end
    progress = (n / n_max) ** 2  # Quadratic growth
    return max(round(n_ei_c_start + progress * (n_ei_c_end - n_ei_c_start)), 1)

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
    return max(xi_start + (xi_end - xi_start) * (n / n_max), 0)


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