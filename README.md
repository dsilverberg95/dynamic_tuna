# Dynamic Tuna: Flexible Bayesian Optimization

**Dynamic Tuna** is a library of surrogate models that are compatible with Optuna's API for performing Bayesian optimization on machine learning (ML) hyperparameters. The provided samplers are Gaussian Process, Random Forest, and Tree-structured Parzen Estimator. While most libraries (e.g. Optuna) allow for some degree of static control over the search process' parameters, Dynamic Tuna allows for dynamic control through several mechanisms.

![Dynamic Tuna](https://img.shields.io/badge/bayesian-optimization-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Exploitation-Exploration Control

Consider the general acquisition function $A$ defined as $$A(\theta) = \mu(\theta) + \xi\cdot\sigma(\theta)$$ where $\theta$ is a hyperparameter configuration, $\mu(\theta)$ is the surrogate model's expected value at $\theta$, $\sigma(\theta)$ is the uncertainty of the surrogate model's value at $\theta$, and $\xi$ is a non-negative number. Given that we choose the maximizing argument of $A$ at each step of the search process, a smaller $\xi$ will encourage the selection of configurations with high expected values, while a larger $\xi$ will encourage the selection of those with those whose values are more uncertain. Dynamic Tuna allows the user to specify $\xi$ as either a constant $\xi = c$ or a function $\xi = f(n, t)$ where n is the number of previously completed trials and t is the total number of trials to run. This construction allows the user to emphasize early exploration, late exploitation, or both. 

---

## Randomness Control

In practice, choosing the configuration that maximizes $A$ is done by evaluating $A$ at a huge number $m$ of randomly sampled configurations. Dynamic Tuna enables the user to specify $m$ at each trial in the search process, either as a constant $m=k$ or a function $m=g(n, t)$, where $n$ and $t$ are the number of completed and total trials respectively . Setting $m=1$ will perform random sampling, while setting $m$ to a large value will induce a more faithful adherence to the acquisition function. One advantage of letting $m$ vary with $n$ and $t$ is that the user can gradually inject less randomness into the selection of $\theta$ as the surrogate model acquires more data to train on. This sidesteps the risk of overfitting the surrogate model to relatively few observations. The choice of $m$ can also reflect the user's confidence that the surrogate model resembles the objective function. 

---

## Installation

Clone the repository and install the required packages from `requirements.txt`:

```bash
git clone https://github.com/yourusername/dynamic_tuna.git
cd dynamic_tuna
pip install -r requirements.txt


```

---

## Basic Usage

Here's an example to demonstrate the library's syntax. For information on defining a trial's search space, see https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial:

```python
import optuna
from dynamic_tuna.samplers import GPSampler

total_trials = 100

# define function to optimize
def objective_function(trial):
    x = trial.suggest_float('x', -1, 1) # specify argument and its domain
    return x**2

# function specifying value of xi at each trial
def linear_xi(n, xi_start=1, xi_end=0.01, n_max=total_trials):
    return max(xi_start + (xi_end - xi_start) * (n / n_max), 0)

# function specifying, at each trial, number of candidates at which to evaluate acquisition function
def quadratic_n_ei_candidates(n, n_ei_c_start=1, n_ei_c_end=1000000, n_max=total_trials):
    return max(round(n_ei_c_start + ((n / n_max) ** 2) * (n_ei_c_end - n_ei_c_start)), 1)

# instantiate surrogate model, specifying kernel in the case of GP
sampler = GPSampler(xi_function=lambda n: linear_xi(n, xi_start=1.0, xi_end=0.01, n_max=total_trials),
                    n_function=lambda n: quadratic_n_ei_candidates(n, n_ei_c_start=1, n_ei_c_end=1000, n_max=total_trials),
                    kernel = {"Matern": {"nu": 1.5, "length_scale": 3.0},
                              "Noise": {"noise_level": 2.0}})

# Create optimization process
study = optuna.create_study(direction='maximize', # specify whether to maximize or minimize objective function
                            sampler=sampler)

# Run optimization process
study.optimize(func = objective_function, 
               n_trials = total_trials)

print("Best Argument:", study.best_params)
```

---

## Supported Surrogate Models

Dynamic Tuna offers the following surrogate models: Gaussian Process, Random Forest, and Tree-structured Parzen Estimator. Which surrogate model the user selects should depends on their understanding of the search space (i.e. dimensionality, smoothness, presence of categoricals, etc.). As a general rule of thumb, a relatively smooth search space with dimensionality below 20 and without categorical variables is most efficiently modeled by Gaussian Process. For larger dimensionality or presence of categoricals, Random Forest or Tree-based Parzen Estimator should be used. Notably, GP does not support the presence of categorical variables, and using GP in such a case will generate an error. While Tree-based Parzen Estimator is more efficient than Random Forest, it is incapable of using correlation across dimensions of the search space. If the user suspects that this is important, then use Random Forest.


#### Gaussian Process

```python

```

#### Random Forest

```python

```

#### Tree-structured Parzen Estimator

```python

```

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact

Have questions or feedback? Reach out to me at dsilverberg95@gmail.com or create an issue in the repository.

