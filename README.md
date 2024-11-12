# Dynamic Tuna: Flexible Bayesian Optimization Library

**Dynamic Tuna** is a library of surrogate models that are compatible with the Optuna framework for performing Bayesian optimization (BO) of machine learning hyperparameters. The provided surrogate models (i.e. samplers) include Gaussian Process, Random Forest, and Tree-Structured Parzen Estimator. While most BO libraries allow for some sort of static control over the exploitation-exploration tradeoff during the search, Dynamic Tuna allows for dynamic control. Several mechanisims for doing so are explained below, along with their respective rationales. 

![Dynamic Tuna](https://img.shields.io/badge/bayesian-optimization-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green)

---

## 🔭 Bayesian Optimization at a Glance

Bayesian Optimization (BO) is a sequential model-based optimization technique that leverages surrogate models to efficiently search the hyperparameter space. It approximates the objective function—typically model performance—using models such as Gaussian Processes, which provide uncertainty estimates, or the Tree-structured Parzen Estimator (TPE), which does not. Each iteration selects hyperparameters to maximize an acquisition function, balancing exploration and exploitation. By directing evaluations toward configurations with the highest expected improvement, BO reduces the number of required trials, making it highly effective for optimizing models with expensive training costs.

---

## 🧭 The Exploitation-Exploration Tradeoff

In Bayesian Optimization, the exploitation/exploration tradeoff governs how hyperparameters are selected in each iteration. Exploitation involves sampling configurations known to perform well, aiming to quickly converge on a local optimum. Exploration, in contrast, seeks out new, uncertain regions of the hyperparameter space to identify potentially better solutions. The acquisition function mediates this tradeoff by assigning scores that balance the expected improvement from both strategies. For instance, Gaussian Processes leverage uncertainty estimates to explore regions with high variance, while the Tree-structured Parzen Estimator (TPE) focuses on probability density estimation to strike a similar balance. Managing this tradeoff is essential for efficiently navigating the search space and finding global optima with minimal evaluations.

---

## 🧠 Why use Dynamic Tuna? (Features)





---

## 📥 Installation

Clone the repository and install the required packages from `requirements.txt`:

```bash
git clone https://github.com/yourusername/dynamic_tuna.git
cd dynamic_tuna
pip install -r requirements.txt


	Note: This library requires Python 3.8+.
```
## 🚀 Getting Started
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
## ⚙️ Usage

	1.	Initialize a Sampler:
Choose from GBTSampler, RandomForestSampler, or GPSampler and define your n_ei_function.
	2.	Define a Search Space:
Use Optuna’s UniformDistribution, IntUniformDistribution, and more.
	3.	Run Optimization:
Run your study with optuna.create_study() and check the results.

## 🔧 Contributing

Contributions are welcome! To contribute:

	1.	Fork the repository.
	2.	Create a new branch with your feature or bugfix.
	3.	Submit a pull request with a description of your changes.

Please ensure that your code is well-documented and tested.

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 📬 Contact

Have questions or feedback? Reach out to me at dsilverberg95@gmail.com or create an issue in the repository.

