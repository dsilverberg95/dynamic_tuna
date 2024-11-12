# Dynamic Tuna: Flexible Bayesian Optimization Library

**Dynamic Tuna** is a library of surrogate models that are compatible with the Optuna framework for performing Bayesian optimization (BO). Surrogate models (i.e. samplers) include Gaussian Process, Random Forest, and Tree-Structured Parzen Estimator. While most BO libraries allow for some sort of static control over the exploitation-exploration tradeoff during the search, dynamic-tuna allows for dynamic control. Several mechanisims for doing so are explained below, along with their respective rationales. 

![Dynamic Tuna](https://img.shields.io/badge/bayesian-optimization-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green)

---

## 🌟 Bayesian Optimziation at a Glance


- **Dynamic Sampling**: Control the exploration-exploitation tradeoff with flexible `n_ei_function` parameters.

- **Diverse Samplers**: Use Gaussian Process, Random Forest, or Gradient Boosted Trees as the surrogate model for efficient sampling.
- **Customizable Hyperparameters**: Define custom `n_ei_function` parameters for precise control over the optimization process.


---

## 🌟 The Exploitation-Exploration Tradeoff


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

