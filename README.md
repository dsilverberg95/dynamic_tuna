# Dynamic Tuna: Flexible Bayesian Optimization Library

**Dynamic Tuna** is a library of surrogate models that are compatible with the Optuna framework for performing Bayesian optimization (BO) on machine learning hyperparameters. The provided surrogate models (i.e. samplers) include Gaussian Process, Random Forest, and Tree-structured Parzen Estimator. While most BO libraries allow for some sort of static control over the exploitation-exploration (EE) tradeoff during the search, Dynamic Tuna allows for dynamic control (hence, the name). Several mechanisms for doing so are explained below, along with their respective rationales. 

![Dynamic Tuna](https://img.shields.io/badge/bayesian-optimization-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green)

---

## 🔭 Bayesian Optimization at a Glance

Bayesian Optimization (BO) is a sequential optimization technique that efficiently searches a hyperparameter space by using a surrogate model (e.g. Gaussian Process) to approximate a computationally expensive objective function. At each iteration, the algorithm selects the hyperparameters that maximize a pre-determined acquisition function (AF), which is chosen by the user to balance exploration and exploitation within the search space. The model then trains with the selected configuration, and its performance is used to adjust the surrogate model via Bayesian updating to make a more informed hyperparameter selection at the next iteration. This process continues until a stopping criterion is reached.

---

## 🧠 Exploitation-Exploration with Dynamic Tuna

Dynamic Tuna allows for two types of dynamic control over the EE tradeoff: *active* control and *passive* control. Active control is when you include a measure of uncertainty in your acquisition function in addition to the usual measure of expected improvement. The user can then directly manipulate the parameter that balances uncertainty with expected improvement. Conversely, passive control is when the acquisition function does *not* include an uncertainty measure, in which case the user can manipulate a separate parameter that balances exploitation with random search. The more you know about the search space, the more you should exploit your knowledge. Similarly, the less time you have remaining to explore the search space, the smaller your incentive is to explore (unless your current best is terrible). The GP and RF samplers in Dynamic Tuna allow for exploration-exploration control via the parameter $xi$, which is consistent with the corresponding versions of these 
samplers in the scikit-optimize package (no longer maintained as of 11/14/24). On the other hand, the Tree-structured Parsen Estimator in the Optuna package allows you to control the exploitation/exploration tradeoff in a less direct fashion via the parameter $n_ei_candidates$. 


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

