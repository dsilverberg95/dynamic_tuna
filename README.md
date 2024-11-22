# Dynamic Tuna: Flexible Bayesian Optimization Library

**Dynamic Tuna** is a library of surrogate models that are compatible with the Optuna framework for performing Bayesian optimization (BO) on machine learning hyperparameters. The provided surrogate models (i.e. samplers) include Gaussian Process, Random Forest, and Tree-structured Parzen Estimator. While most BO libraries allow for some sort of static control over the exploitation-exploration (EE) tradeoff during the search, Dynamic Tuna allows for dynamic control (hence, the name). Several mechanisms for doing so are explained below, along with their respective rationales. 

![Dynamic Tuna](https://img.shields.io/badge/bayesian-optimization-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green)

---

## 🔭 Bayesian Optimization at a Glance

Bayesian Optimization (BO) is a sequential optimization technique that efficiently searches a hyperparameter space by using a surrogate model (e.g. Gaussian Process) to approximate a computationally expensive objective function. At each iteration, the algorithm selects the hyperparameters that maximize a pre-determined acquisition function (AF), which is chosen by the user to balance exploration and exploitation within the search space. The model then trains with the selected configuration, and its performance is used to adjust the surrogate model via Bayesian updating to make a more informed hyperparameter selection at the next iteration. This process continues until a stopping criterion is reached.

---

## 🧠 Exploitation-Exploration Control with Dynamic Tuna

Dynamic Tuna allows for two types of control over the EE tradeoff: *uncertainty* control (UC) and *ambiguity* control (AC). Suppose the user believes that the surrogate model effectively represents their beliefs about the search space. If the user instructs the sampler to explicitly seek out uncertain regions of the search space, the user is exercising UC. On the other hand, suppose the user does not trust that the surrogate model effectively represents their beliefs about the search space due to ambiguity regarding, say, the space's smoothness. Note that many models are optimized by quickly evaluating a huge number of points and using the approximate optimum. If the user chooses to sample a smaller number of points to find the optimum (thereby introducing more randomness into the search process), the user is exercising AC. 

Consider a Gaussian Process:

\[
f(x) \sim \mathcal{GP}(m(x), k(x, x'))
\]

where:
- \( m(x) \) is the mean function (often assumed to be zero, \( m(x) = 0 \), for simplicity),
- \( k(x, x') \) is the covariance function (kernel) that quantifies the similarity between points \( x \) and \( x' \).

For a given set of observations \( \{(x_i, y_i)\}_{i=1}^n \), the GP posterior predicts the mean \( \mu(x) \) and variance \( \sigma^2(x) \) at a new point \( x \):

\[
\mu(x) = k(x, X) [K + \sigma_n^2 I]^{-1} y
\]

\[
\sigma^2(x) = k(x, x) - k(x, X) [K + \sigma_n^2 I]^{-1} k(X, x)
\]

Here:
- \( k(x, X) \) is the covariance between \( x \) and all observed points \( X \),
- \( K \) is the covariance matrix of the observed points,
- \( \sigma_n^2 \) is the noise variance.

The predicted mean \( \mu(x) \) represents the model’s belief about the function value at \( x \), and \( \sigma(x) \) quantifies uncertainty.

---

### Role of \( \xi \) in Acquisition Functions

The acquisition function (AF) determines which point to sample next by balancing \textit{exploitation} (selecting points near the predicted maximum) and \textit{exploration} (selecting points with high uncertainty). One commonly used AF is the \textbf{Expected Improvement (EI)}:

\[
\text{EI}(x) = \left( \mu(x) - f^* - \xi \right) \Phi(Z) + \sigma(x) \phi(Z)
\]

where:
- \( f^* \) is the best observed value so far,
- \( Z = \frac{\mu(x) - f^* - \xi}{\sigma(x)} \) is the standardized improvement,
- \( \Phi(Z) \) and \( \phi(Z) \) are the cumulative distribution function and probability density function of the standard normal distribution, respectively,
- \( \xi \) is a hyperparameter that adjusts the tradeoff between exploitation and exploration:
  - Larger \( \xi \): Prioritizes exploration by increasing the weight on uncertainty \( \sigma(x) \),
  - Smaller \( \xi \): Focuses more on exploitation by emphasizing the improvement \( \mu(x) - f^* \).

In the \textbf{static case}, \( \xi \) remains constant throughout the optimization, and its value is often tuned based on the problem's characteristics:
- For highly noisy functions or functions with sparse observations, a larger \( \xi \) may be beneficial,
- For smoother, well-sampled functions, a smaller \( \xi \) can accelerate convergence to the optimum.

---

### Introducing Randomness via \( n_\text{ei\_candidates} \)

In Bayesian Optimization, the acquisition function is typically maximized to select the next point \( x \). Directly optimizing the acquisition function might lead to deterministic sampling strategies, which can limit the exploration of diverse regions in the search space.

To introduce \textbf{randomness}, the parameter \( n_\text{ei\_candidates} \) can be leveraged. Here’s how:

1. \textbf{Candidate Generation}:
   - Instead of globally maximizing the acquisition function, generate \( n_\text{ei\_candidates} \) random points in the search space.
   - Evaluate the acquisition function at these points.
   - Select the candidate with the highest acquisition function value.

2. \textbf{Impact of \( n_\text{ei\_candidates} \)}:
   - \textit{Smaller values} of \( n_\text{ei\_candidates} \): Introduces more randomness, as the sampled points are less likely to fully represent the global maximum of the acquisition function.
   - \textit{Larger values}: Reduces randomness, as the global maximum of the acquisition function is more likely to be captured.

3. \textbf{Dynamic Control of \( n_\text{ei\_candidates} \)}:
   - The randomness introduced by \( n_\text{ei\_candidates} \) can be controlled dynamically to vary the balance between deterministic optimization and random exploration. For example:
     - \textit{Early stages}: Use a small \( n_\text{ei\_candidates} \) to explore diverse regions,
     - \textit{Later stages}: Gradually increase \( n_\text{ei\_candidates} \) to converge more reliably to the optimum.

An example function for dynamic control might look like this:

\[
n_\text{ei\_candidates} = \text{base} + \text{progress} \cdot (\text{max\_candidates} - \text{base})
\]

where progress is defined as:

\[
\text{progress} = \frac{n_\text{completed}}{\text{total\_trials}}
\]

With this, the number of candidates increases linearly as more trials are completed.

---

### Summary

The Gaussian Process sampler in Dynamic Tuna allows for nuanced control of the Bayesian Optimization process through:
- \textbf{Static \( \xi \)}: Balancing exploration and exploitation with a fixed tradeoff parameter,
- \textbf{Dynamic \( n_\text{ei\_candidates} \)}: Introducing controlled randomness to enhance exploration in early stages and refine exploitation in later stages.

This combination provides flexibility to tailor the optimization process to the user’s needs and ensures robustness in a wide range of search spaces.


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

