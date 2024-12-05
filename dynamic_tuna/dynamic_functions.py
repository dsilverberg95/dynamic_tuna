### some possible n_ei_function
import numpy as np

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