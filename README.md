# Dynamic Tuna: Flexible Bayesian Optimization Library

**Dynamic Tuna** is a library of surrogate models that are compatible with the Optuna framework for performing Bayesian optimization (BO) of machine learning hyperparameters. The provided surrogate models (i.e. samplers) include Gaussian Process, Random Forest, and Tree-structured Parzen Estimator. While most BO libraries allow for some sort of static control over the exploitation-exploration tradeoff during the search, Dynamic Tuna allows for dynamic control (hence, the name). Several mechanisms for doing so are explained below, along with their respective rationales. 

![Dynamic Tuna](https://img.shields.io/badge/bayesian-optimization-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green)

---

## 🔭 Bayesian Optimization at a Glance

Bayesian Optimization (BO) is a sequential optimization technique that efficiently 
searches a hyperparameter space by using a surrogate model (e.g. Gaussian Process)
to approximate a computationally expensive objective function. At each iteration, the algorithm selects whichever hyperparameter combination 
that happens to maximize a pre-determined acquisition function (AF), which is generally selected by the user to balance exploration and exploitation 
within the search space. The performance of the selected configuration is then used to adjust the surrogate model via Bayesian updating to ultimately
make a more informed hyperparameter selection at the next iteration. 


---

## ⚖️ Overview of Exploitation-Exploration Tradeoff

The exploitation-exploration tradeoff governs how hyperparameters 
are selected at each iteration. Exploitation involves sampling configurations known to perform 
well, aiming to quickly converge on a local optimum. Exploration, in contrast, seeks out new, 
uncertain regions of the hyperparameter space to identify potentially better solutions. The 
acquisition function mediates this tradeoff by assigning scores that balance the expected 
improvement from both strategies. For instance, Gaussian Processes and Random Forests leverage 
uncertainty estimates to explore regions with high variance. On the other hand, Tree-structured 
Parzen Estimators (TPE) do not inherently support uncertainty estimation, so over-exploitation 
is combatted indirectly through separate mechanisms (see below). Ultimately, managing the 
exporation-exploitation tradeoff is essential for efficiently navigating the search space and 
finding a sufficiently good hyperparameter configuration with minimal evaluations.

---

## 🧠 Exploitation-Exploration with Dynamic Tuna

Dynamic Tuna allows for two types of dynamic control over the EE tradeoff, which I will 
call *active* and *passive* control. Active control is when you include a measure of 
uncertainty in your acquisition function in addition to the usual measure of expected 
improvement. The user can then directly manipulate the parameter that balances uncertainty 
with expected improvement. Conversely, passive control is when the acquisition function 
does *not* include an uncertainty measure, in which case the user can manipulate a separate 
parameter that balances exploitation with random search. The more you know about the search 
space, the more you should exploit your knowledge. Similarly, the less time you have 
remaining to explore the search space, the smaller your incentive is to explore (unless your 
current best is terrible). The GP and RF samplers in Dynamic Tuna allow for exploration-exploration 
control via the parameter $xi$, which is consistent with the corresponding versions of these 
samplers in the scikit-optimize package (no longer maintained as of 11/14/24). On the other 
hand, the Tree-structured Parsen Estimator in the Optuna package allows you to control the 
exploitation/exploration tradeoff in a less direct fashion via the parameter $n_ei_candidates$. 





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

