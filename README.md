# Dynamic Tuna: Flexible Bayesian Optimization Library

**Dynamic Tuna** is a library of surrogate models that are compatible with the Optuna framework for performing Bayesian optimization on machine learning hyperparameters. The provided samplers are Gaussian Process, Random Forest, and Tree-structured Parzen Estimator. While most libraries allow for some degree of static control over surrogate model parameters that govern the exploitation-exploration tradeoff, Dynamic Tuna allows for dynamic control.

All explanations and usage examples below assume a degree of familiarity with Bayesian optimization.

![Dynamic Tuna](https://img.shields.io/badge/bayesian-optimization-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green)

---

## 1.  Library features

Consider the general acquisition function $A(\theta)$, where $\theta$ represents the unknown hyperparameter configuration. 

---

## 2.  Exploitation-Exploration Control with Dynamic Tuna

Dynamic Tuna simultaneously allows for two types of control over the EE tradeoff which we'll refer to as *uncertainty* control (UC) and *ambiguity* control (AC). Suppose the user believes that the surrogate model effectively represents their beliefs about the search space. If the user instructs the sampler to explicitly seek out uncertain regions of the search space, the user is exercising UC. On the other hand, suppose the user does not trust that the surrogate model effectively represents their beliefs about the search space due to ambiguity regarding, say, the space's smoothness. Note that many models are optimized by quickly evaluating a huge number of points and using the approximate optimum. If the user chooses to sample a smaller number of points to find the optimum (thereby relying less on the unerlying sampler and introducing more randomness into the search process), the user is exercising AC. 

Consider a Gaussian Process:

---

## 3.  Dynamic EE Control with Dynamic Tuna


---

## 4.  Installation

Clone the repository and install the required packages from `requirements.txt`:

```bash
git clone https://github.com/yourusername/dynamic_tuna.git
cd dynamic_tuna
pip install -r requirements.txt


	Note: This library requires Python 3.8+.
```
## 5.  Getting Started
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
## 6.  Usage

	1.	Initialize a Sampler:
Choose from GBTSampler, RandomForestSampler, or GPSampler and define your n_ei_function.
	2.	Define a Search Space:
Use Optuna’s UniformDistribution, IntUniformDistribution, and more.
	3.	Run Optimization:
Run your study with optuna.create_study() and check the results.

## 7.  License

This project is licensed under the MIT License. See the LICENSE file for details.

## 8.  Contact

Have questions or feedback? Reach out to me at dsilverberg95@gmail.com or create an issue in the repository.

