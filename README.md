# Dynamic Tuna: Flexible Bayesian Optimization

**Dynamic Tuna** is a library of surrogate models that are compatible with Optuna's API for performing Bayesian optimization on machine learning (ML) hyperparameters. The provided samplers are Gaussian Process, Random Forest, and Tree-structured Parzen Estimator. While most libraries (Optuna included) allow for some degree of static control over the search process' parameters, Dynamic Tuna allows for dynamic control through several distinct mechanisms.

![Dynamic Tuna](https://img.shields.io/badge/bayesian-optimization-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Exploitation-Exploration Control

Consider the general acquisition function $A$ defined as $$A(\theta) = \mu(\theta) + \xi\cdot\sigma(\theta)$$ where $\theta$ is a hyperparameter configuration, $\mu(\theta)$ is the surrogate model's expected value at $\theta$, $\sigma(\theta)$ is the uncertainty of the surrogate model's value at $\theta$, and $\xi$ is a non-negative number. Given that we choose the maximizing argument of $A$ at each step of the search process, a smaller $\xi$ will encourage the selection of configurations with high expected values, while a larger $\xi$ will encourage the selection of those with those whose values are more uncertain. Dynamic Tuna allows the user to specify $\xi$ as either a constant $\xi = c$ or a function $\xi = f(n, t)$ where n is the number of previously completed trials and t is the total number of trials to run. This construction allows the user to emphasize early exploration, late exploitation, or both. 

---

## Randomness Control

In practice, choosing the configuration that maximizes $A$ is done by evaluating $A$ at a huge number $m$ of randomly sampled configurations. Dynamic Tuna enables the user to specify $m$ at each trial in the search process, either as a constant $m=k$ or a function $m=g(n, t)$, where $n$ and $t$ are again the number of completed and total trials respectively . Setting $m=1$ will perform random sampling, while setting $m$ to a large value will induce a more faithful adherence to the acquisition function. The choice of $m$ can reflect the user's confidence that the surrogate model resembles the objective function. One advantage of letting $m$ vary with $n$ and $t$ is that the user can gradually inject less randomness into the selection of $\theta$ as the surrogate model acquires more data to train on. This sidesteps the risk of overfitting the surrogate model to relatively few observations. 

---

## Installation

Clone the repository and install the required packages from `requirements.txt`:

```bash
git clone https://github.com/yourusername/dynamic_tuna.git
cd dynamic_tuna
pip install -r requirements.txt


	Note: This library requires Python 3.8+.
```

---

## Basic Usage

Here's a minimalist example to demonstrate the library's essential syntax:

```python
import optuna
from dynamic_tuna.samplers import GPSampler

# define function to optimize
def objective_function(trial):
    x = trial.suggest_float('x', -1, 1) # specify argument and its domain
    return x**2

# instantiate surrogate model
sampler = GPSampler() 

# Create optimization process
study = optuna.create_study(
                            direction='minimize', # specify whether to maximize or minimize objective function
                            sampler=sampler
                            )

# Run optimization process
study.optimize(
               func = objective_function, 
               n_trials = 10
               )

print("Best Argument:", study.best_params)
```

---

## Supported Surrogate Models

As previously stated, the user can choose between Gaussian Process, Random Forest, and Tree-based Parzen Estimator. Which surrogate the user selects should depends on their understanding of the search space (dimensionality, smoothness, presence of categoricals, etc.). As a general rule of thumb, a relatively smooth search space with dimensionality below 20 and without categorical variables is most efficiently modeled by Gaussian Process. For larger dimensionality or presence of categoricals, Random Forest or Tree-based Parzen Estimator should be used. Notably, GP does not support the presence of categorical variables, and using GP in such a case will generate an error. While Tree-based Parzen Estimator is more efficient than Random Forest, it is incapable of using correlation across dimensions of the search space. If the user suspects that this is important, then use Random Forest.


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

