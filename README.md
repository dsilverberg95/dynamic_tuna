# Dynamic Tuna: Flexible Bayesian Optimization Library

**Dynamic Tuna** is a library of surrogate models that are compatible with the Optuna framework for performing Bayesian optimization on machine learning hyperparameters. The provided samplers are Gaussian Process, Random Forest, and Tree-structured Parzen Estimator. While most libraries allow for some degree of static control over the search process' exploitation-exploration tradeoff, Dynamic Tuna allows for multiple types of dynamic control.

All explanations and usage examples will assume a degree of familiarity with Bayesian inference.

![Dynamic Tuna](https://img.shields.io/badge/bayesian-optimization-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Exploration-Exploitation Tradeoff

Consider the general acquisition function $A$ expressed as $$A(\theta) = \mu(\theta) + \xi\cdot\sigma(\theta)$$ where $\theta$ is a hyperparameter configuration, $\mu(\theta)$ is the expected value of the surrogate model at $\theta$, $\sigma(\theta)$ is the standard deviation of the surrogate model at $\theta$, and $\xi$ is a non-negative number. Given that we choose the maximizing argument of $A$ at each step of the search process, a smaller $\xi$ will encourage the selection of configurations with high expected values, while a larger $\xi$ will encourage the selection of those with higher uncertainty. Dynamic Tuna allows the user to specify $\xi$ as either a constant or a function $f(n, t)$ where n is the number of previously completed trials and t is the total number of trials to run. In defining $\xi$ as $\xi = f(n, t)$, the user can emphasize early exploration, late exploitation, or both. 

---

## Available Surrogate Models



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

## Example Usage
Here’s a quick example to get you started with Dynamic Tuna.

```python
import optuna
from dynamic_tuna import GBTSampler

# Define search space
search_space = {
    "param1": optuna.distributions.UniformDistribution(0, 1),
    "param2": optuna.distributions.IntUniformDistribution(1, 100)}

# Initialize sampler with dynamic EI candidates
sampler = GBTSampler(search_space, n_ei_function=lambda n, a=1, b=2: a * n + b)

# Create and run study
study = optuna.create_study(sampler=sampler)
study.optimize(objective_function, n_trials=50)
print("Best Parameters:", study.best_params)
```
## Usage

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact

Have questions or feedback? Reach out to me at dsilverberg95@gmail.com or create an issue in the repository.

