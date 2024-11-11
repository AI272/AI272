- 👋 Hi, I’m @AI272
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...

<!---
AI272/AI272 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->



——————————————————————————————————————————————
test：



import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

def generate_data():
    np.random.seed(42)
    X = np.linspace(-5, 5, 50)
    y_true = 3 * X + 2
    y_observed = y_true + np.random.normal(0, 4, size=X.shape)
    return X, y_true, y_observed

def bayesian_inference(X, y_observed):
    with pm.Model() as model:
        w0 = pm.Normal("w0", mu=0, sigma=1)
        w1 = pm.Normal("w1", mu=0, sigma=1)
        y_pred = w1 * X + w0
        y_obs = pm.Normal("y_obs", mu=y_pred, sigma=4, observed=y_observed)
        trace = pm.sample(2000, return_inferencedata=False, tune=1000)
    return trace

def plot_results(X, y_true, y_observed, trace):
    w0_samples = trace["w0"]
    w1_samples = trace["w1"]
    y_pred_samples = np.array([w1_samples[i] * X + w0_samples[i] for i in range(len(w0_samples))])
    y_pred_mean = y_pred_samples.mean(axis=0)
    y_pred_lower = np.percentile(y_pred_samples, 2.5, axis=0)
    y_pred_upper = np.percentile(y_pred_samples, 97.5, axis=0)

    plt.scatter(X, y_observed, label="Observed data")
    plt.plot(X, y_true, label="True relationship", color="orange")
    plt.plot(X, y_pred_mean, label="Predicted Mean", color="blue")
    plt.fill_between(X, y_pred_lower, y_pred_upper, color="blue", alpha=0.2, label="95% CI")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()

# workflow
X, y_true, y_observed = generate_data()
trace = bayesian_inference(X, y_observed)
plot_results(X, y_true, y_observed, trace)
