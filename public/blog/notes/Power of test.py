# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
mu0 = 0        # null mean
mu1 = 1        # true mean under H1
sigma = 2      # population std
n = 16         # sample size
alpha = 0.05   # significance level (two-sided)

# Standard error
se = sigma / np.sqrt(n)

# Z critical for two-sided test
z_crit = norm.ppf(1 - alpha/2)

# Rejection region boundaries in sample mean scale
x_low = mu0 - z_crit * se
x_high = mu0 + z_crit * se

# Range for plotting
x = np.linspace(mu0 - 4*se, mu1 + 4*se, 1000)

# H0 and H1 distributions
h0_pdf = norm.pdf(x, mu0, se)
h1_pdf = norm.pdf(x, mu1, se)

# Plot
plt.figure(figsize=(10,6))
plt.plot(x, h0_pdf, label='H0: μ=0', color='blue')
plt.plot(x, h1_pdf, label='H1: μ=1', color='red')

# Shade rejection regions under H0
plt.fill_between(x, 0, h0_pdf, where=(x <= x_low) | (x >= x_high), color='blue', alpha=0.2, label='Rejection region (α)')

# Shade power region under H1 (H1 in rejection region)
plt.fill_between(x, 0, h1_pdf, where=(x <= x_low) | (x >= x_high), color='red', alpha=0.3, label='Power area (1-β)')

plt.axvline(x_low, color='black', linestyle='--')
plt.axvline(x_high, color='black', linestyle='--')

plt.title('Power of a One-Sample Z-Test')
plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.legend()
plt.show()

# Compute power numerically
power = norm.cdf(x_low, mu1, se) + (1 - norm.cdf(x_high, mu1, se))
print(f"Power of the test (P(Reject H0​|H1​ is true)): {power:.3f}")

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from ipywidgets import interact, FloatSlider, IntSlider

def plot_power(mu1=1.0, sigma=2.0, n=16, alpha=0.05):
    mu0 = 0
    se = sigma / np.sqrt(n)
    z_crit = norm.ppf(1 - alpha/2)
    x_low = mu0 - z_crit * se
    x_high = mu0 + z_crit * se
    
    x = np.linspace(mu0 - 4*se, mu1 + 4*se, 1000)
    h0_pdf = norm.pdf(x, mu0, se)
    h1_pdf = norm.pdf(x, mu1, se)
    
    plt.figure(figsize=(10,6))
    plt.plot(x, h0_pdf, label='H0: μ=0', color='blue')
    plt.plot(x, h1_pdf, label=f'H1: μ={mu1}', color='red')
    
    # Rejection regions
    plt.fill_between(x, 0, h0_pdf, where=(x <= x_low) | (x >= x_high), color='blue', alpha=0.2, label='Rejection region (α)')
    plt.fill_between(x, 0, h1_pdf, where=(x <= x_low) | (x >= x_high), color='red', alpha=0.3, label='Power area (1-β)')
    
    plt.axvline(x_low, color='black', linestyle='--')
    plt.axvline(x_high, color='black', linestyle='--')
    
    plt.title('Power of a One-Sample Z-Test')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    power = norm.cdf(x_low, mu1, se) + (1 - norm.cdf(x_high, mu1, se))
    print(f"Power of the test: {power:.3f}")

# Interactive sliders
interact(plot_power,
         mu1=FloatSlider(value=1.0, min=0, max=5.0, step=0.1, description='Effect size μ1'),
         sigma=FloatSlider(value=2.0, min=0.5, max=5.0, step=0.1, description='σ'),
         n=IntSlider(value=16, min=5, max=200, step=1, description='Sample size n'));

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from ipywidgets import interact, FloatSlider, IntSlider

def plot_power_ttest(mu1=1.0, sigma=2.0, n=16, alpha=0.05):
    mu0 = 0
    df = n - 1
    se = sigma / np.sqrt(n)
    
    # t critical value for two-sided test
    t_crit = t.ppf(1 - alpha/2, df)
    
    # Rejection region in sample mean scale
    x_low = mu0 - t_crit * se
    x_high = mu0 + t_crit * se
    
    # Range for plotting
    x = np.linspace(mu0 - 4*se, mu1 + 4*se, 1000)
    
    # PDFs
    h0_pdf = t.pdf((x - mu0)/se, df) / se
    h1_pdf = t.pdf((x - mu1)/se, df) / se
    
    plt.figure(figsize=(10,6))
    plt.plot(x, h0_pdf, label='H0: μ=0', color='blue')
    plt.plot(x, h1_pdf, label=f'H1: μ={mu1}', color='red')
    
    # Shade rejection region under H0
    plt.fill_between(x, 0, h0_pdf, where=(x <= x_low) | (x >= x_high), color='blue', alpha=0.2, label='Rejection region (α)')
    
    # Shade power region under H1
    plt.fill_between(x, 0, h1_pdf, where=(x <= x_low) | (x >= x_high), color='red', alpha=0.3, label='Power area (1-β)')
    
    plt.axvline(x_low, color='black', linestyle='--')
    plt.axvline(x_high, color='black', linestyle='--')
    
    plt.title(f'Power of a Two-Sided t-Test (n={n}, df={df})')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    # Compute power using CDF of t-distribution under H1
    t_low = (x_low - mu1)/se
    t_high = (x_high - mu1)/se
    power = t.cdf(t_low, df) + (1 - t.cdf(t_high, df))
    
    print(f"Power of the test: {power:.3f}")

# Interactive sliders
interact(plot_power_ttest,
         mu1=FloatSlider(value=1.0, min=0, max=5.0, step=0.1, description='Effect size μ1'),
         sigma=FloatSlider(value=2.0, min=0.5, max=5.0, step=0.1, description='σ'),
         n=IntSlider(value=16, min=5, max=200, step=1, description='Sample size n'));

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from tqdm import trange

np.random.seed(42)

# Simulate small correlated dataset
n = 30
X1 = np.random.normal(0, 1, n)
X2 = X1 * 0.8 + np.random.normal(0, 0.6, n)  # correlated with X1
beta1_true = 2
beta2_true = 0
epsilon = np.random.normal(0, 1, n)
Y = beta1_true * X1 + beta2_true * X2 + epsilon

# Classical OLS
X = np.column_stack([np.ones(n), X1, X2])
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ Y
residuals = Y - X @ beta_hat
sigma_hat = np.sqrt(np.sum(residuals**2)/(n-3))
SE_beta1 = sigma_hat * np.sqrt(np.linalg.inv(X.T @ X)[1,1])
t_stat_obs = beta_hat[1] / SE_beta1

print(f"Classical t-stat for X1: {t_stat_obs:.3f}")

# ------------------------
# Bootstrap
# ------------------------
B = 5000
t_boot = []
for _ in trange(B):
    idx = np.random.choice(n, n, replace=True)
    X_b = X[idx]
    Y_b = Y[idx]
    beta_b = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y_b
    res_b = Y_b - X_b @ beta_b
    sigma_b = np.sqrt(np.sum(res_b**2)/(n-3))
    SE_b = sigma_b * np.sqrt(np.linalg.inv(X_b.T @ X_b)[1,1])
    t_boot.append(beta_b[1]/SE_b)
t_boot = np.array(t_boot)

# Bootstrap p-value
p_boot = np.mean(np.abs(t_boot) >= np.abs(t_stat_obs))
print(f"Bootstrap p-value for X1: {p_boot:.3f}")

# ------------------------
# Permutation
# ------------------------
t_perm = []
for _ in trange(B):
    Y_perm = np.random.permutation(Y)
    beta_perm = np.linalg.inv(X.T @ X) @ X.T @ Y_perm
    res_perm = Y_perm - X @ beta_perm
    sigma_perm = np.sqrt(np.sum(res_perm**2)/(n-3))
    SE_perm = sigma_perm * np.sqrt(np.linalg.inv(X.T @ X)[1,1])
    t_perm.append(beta_perm[1]/SE_perm)
t_perm = np.array(t_perm)

# Permutation p-value
p_perm = np.mean(np.abs(t_perm) >= np.abs(t_stat_obs))
print(f"Permutation p-value for X1: {p_perm:.3f}")

# ------------------------
# Plot distributions
# ------------------------
plt.figure(figsize=(12,6))
plt.hist(t_perm, bins=50, alpha=0.5, label='Permutation', density=True, color='gray')
plt.hist(t_boot, bins=50, alpha=0.5, label='Bootstrap', density=True, color='red')
x = np.linspace(-5,5,200)
plt.plot(x, t.pdf(x, df=n-3), label='Classical t-dist (df=n-3)', color='blue')
plt.axvline(t_stat_obs, color='black', linestyle='--', label='Observed t-stat')
plt.title('Comparison of t-statistic distributions: Classical vs Bootstrap vs Permutation')
plt.xlabel('t-statistic for X1')
plt.ylabel('Density')
plt.legend()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.stats import t
np.random.seed(42)

# Initial parameters
n_init = 30
corr_init = 0.8
beta1_init = 2
sigma_init = 1
B = 1000  # bootstrap / permutation samples

def simulate_and_plot(n, corr, beta1, sigma):
    # Generate correlated X1, X2
    X1 = np.random.normal(0,1,n)
    X2 = X1 * corr + np.random.normal(0, np.sqrt(1-corr**2), n)
    epsilon = np.random.normal(0, sigma, n)
    Y = beta1 * X1 + epsilon
    
    # Design matrix
    X = np.column_stack([np.ones(n), X1, X2])
    df = n - 3
    
    # Classical t-stat
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ Y
    residuals = Y - X @ beta_hat
    sigma_hat = np.sqrt(np.sum(residuals**2)/df)
    SE_beta1 = sigma_hat * np.sqrt(np.linalg.inv(X.T @ X)[1,1])
    t_obs = beta_hat[1]/SE_beta1
    
    # Bootstrap
    t_boot = []
    for _ in range(B):
        idx = np.random.choice(n, n, replace=True)
        X_b = X[idx]
        Y_b = Y[idx]
        beta_b = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y_b
        res_b = Y_b - X_b @ beta_b
        sigma_b = np.sqrt(np.sum(res_b**2)/df)
        SE_b = sigma_b * np.sqrt(np.linalg.inv(X_b.T @ X_b)[1,1])
        t_boot.append(beta_b[1]/SE_b)
    t_boot = np.array(t_boot)
    p_boot = np.mean(np.abs(t_boot) >= np.abs(t_obs))
    
    # Permutation
    t_perm = []
    for _ in range(B):
        Y_perm = np.random.permutation(Y)
        beta_perm = np.linalg.inv(X.T @ X) @ X.T @ Y_perm
        res_perm = Y_perm - X @ beta_perm
        sigma_perm = np.sqrt(np.sum(res_perm**2)/df)
        SE_perm = sigma_perm * np.sqrt(np.linalg.inv(X.T @ X)[1,1])
        t_perm.append(beta_perm[1]/SE_perm)
    t_perm = np.array(t_perm)
    p_perm = np.mean(np.abs(t_perm) >= np.abs(t_obs))
    
    # Plot
    plt.clf()
    plt.hist(t_perm, bins=50, alpha=0.5, density=True, color='gray', label='Permutation')
    plt.hist(t_boot, bins=50, alpha=0.5, density=True, color='red', label='Bootstrap')
    x_vals = np.linspace(-5,5,200)
    plt.plot(x_vals, t.pdf(x_vals, df=df), color='blue', label='Classical t-dist')
    plt.axvline(t_obs, color='black', linestyle='--', label='Observed t')
    plt.title(f'n={n}, corr={corr:.2f}, β1={beta1}')
    plt.xlabel('t-statistic for X1')
    plt.ylabel('Density')
    plt.legend()
    plt.text(3,0.4, f'p_boot={p_boot:.3f}\np_perm={p_perm:.3f}\nt_obs={t_obs:.2f}')
    plt.draw()

# Initial plot
fig, ax = plt.subplots(figsize=(10,6))
plt.subplots_adjust(left=0.1, bottom=0.35)
simulate_and_plot(n_init, corr_init, beta1_init, sigma_init)

# Sliders
axcolor = 'lightgoldenrodyellow'
ax_n = plt.axes([0.1, 0.25, 0.8, 0.03], facecolor=axcolor)
ax_corr = plt.axes([0.1, 0.2, 0.8, 0.03], facecolor=axcolor)
ax_beta = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor=axcolor)
ax_sigma = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)

s_n = Slider(ax_n, 'n', 10, 100, valinit=n_init, valstep=1)
s_corr = Slider(ax_corr, 'corr', 0.0, 0.99, valinit=corr_init)
s_beta = Slider(ax_beta, 'beta1', 0.0, 5.0, valinit=beta1_init)
s_sigma = Slider(ax_sigma, 'sigma', 0.1, 3.0, valinit=sigma_init)

def update(val):
    simulate_and_plot(int(s_n.val), s_corr.val, s_beta.val, s_sigma.val)
s_n.on_changed(update)
s_corr.on_changed(update)
s_beta.on_changed(update)
s_sigma.on_changed(update)

plt.show()

# %%
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -----------------------------
# Parameters
# -----------------------------
mu0 = 0.0          # null mean
mu1 = 0.5          # true mean (effect size)
sigma = 1.0        # population std
alpha = 0.05
n_sim = 5000       # number of repeated experiments

sample_sizes = [10, 30, 100]

# -----------------------------
# Simulation
# -----------------------------
plt.figure(figsize=(12, 8))

for n in sample_sizes:
    se = sigma / np.sqrt(n)
    z_crit = norm.ppf(1 - alpha / 2)

    pvals = []

    for _ in range(n_sim):
        sample = np.random.normal(mu1, sigma, n)
        xbar = np.mean(sample)
        z = (xbar - mu0) / se
        p = 2 * (1 - norm.cdf(abs(z)))
        pvals.append(p)

    pvals = np.array(pvals)
    power = np.mean(pvals < alpha)

    # Plot histogram of p-values
    plt.hist(
        pvals,
        bins=40,
        density=True,
        alpha=0.5,
        label=f"n={n}, power={power:.2f}"
    )

# Reference line
plt.axvline(alpha, color='black', linestyle='--', label='α = 0.05')

plt.xlabel("p-value")
plt.ylabel("Density")
plt.title("Distribution of p-values under H1 (Power Illustration)")
plt.legend()
plt.tight_layout()
plt.show()