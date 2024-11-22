# Dynamic Tuna: Flexible Bayesian Optimization Library

**Dynamic Tuna** is a library of surrogate models that are compatible with the Optuna framework for performing Bayesian optimization (BO) on machine learning hyperparameters. The provided surrogate models (i.e. samplers) include Gaussian Process, Random Forest, and Tree-structured Parzen Estimator. While most BO libraries allow for some sort of static control over the exploitation-exploration (EE) tradeoff during the search, Dynamic Tuna allows for dynamic control (hence, the name). Several mechanisms for doing so are explained below, along with their respective rationales. 

![Dynamic Tuna](https://img.shields.io/badge/bayesian-optimization-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green)

---

## 🔭 Bayesian Optimization at a Glance

Bayesian Optimization (BO) is a sequential optimization technique that efficiently searches a hyperparameter space by using a surrogate model (e.g. Gaussian Process) to approximate a computationally expensive objective function. At each iteration, the algorithm selects the hyperparameters that maximize a pre-determined acquisition function (AF), which is chosen by the user to balance exploration and exploitation within the search space. The model then trains with the selected configuration, and its performance is used to adjust the surrogate model via Bayesian updating to make a more informed hyperparameter selection at the next iteration. This process continues until a stopping criterion is reached.

---

## 🧠 Exploitation-Exploration Control with Dynamic Tuna

Dynamic Tuna simultaneously allows for two types of control over the EE tradeoff: *uncertainty* control (UC) and *ambiguity* control (AC). Suppose the user believes that the surrogate model effectively represents their beliefs about the search space. If the user instructs the sampler to explicitly seek out uncertain regions of the search space, the user is exercising UC. On the other hand, suppose the user does not trust that the surrogate model effectively represents their beliefs about the search space due to ambiguity regarding, say, the space's smoothness. Note that many models are optimized by quickly evaluating a huge number of points and using the approximate optimum. If the user chooses to sample a smaller number of points to find the optimum (thereby introducing more randomness into the search process), the user is exercising AC. 

Consider a Gaussian Process:

---

## 🧠 Dynamic EE Control with Dynamic Tuna


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

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 📬 Contact

Have questions or feedback? Reach out to me at dsilverberg95@gmail.com or create an issue in the repository.

