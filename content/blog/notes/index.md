---
title: "Notes"
date: 2021-03-19
draft: false
tags: [Statistics, Machine Learning, Deep Learning, Reinforcement Learning, SQL, Python]
categories: []
showReadingTime: false
showTableOfContents: false
summary: "General Notes"
---

<center>
<img src="thumb.jpg" width="50%">
</center>

# Introduction

This contains general notes (definitions, code snippets, useful resources etc.) that I found worthy to keep or topics I've forgotten or confused. If you are seeing this, hope this help you.

# Statistics

### Probability vs. Likelihood
- **Probability** quantifies the chance of future events given fixed model parameters (e.g., chance of heads with a fair coin)
- **Likelihood** assesses the plausibility of model parameters given observed data (e.g., how likely a coin is fair given observed flips).

**Conditional Probability**

$$\mathbf{P(A|B) = \frac{P(A \cap B)}{P(B)}}$$

**Independence**
$$\mathbf{P(A|B) = P(A)} \text{ or } \mathbf{P(B|A) = P(B)}$$

$$\mathbf{P(A \cap B) = P(A) \cdot P(B)} \text{ or } \mathbf{P(B \cap A) = P(A) \cdot P(B)}$$

**Conditional Independence**

$$\mathbf{P(A \cap B | C) = P(A|C) \cdot P(B|C)}$$
$$\mathbf{P(A|B, C) = P(A|C)}$$

**Bayes Theorem**
$$\mathbf{P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}}$$

Total probability:
$$ \mathbf{P(B) = P(B|A) \cdot P(A) + P(B|A')\cdot P(A')} $$

### Skewness

**Left skewed (Neg)**: Tail of the distribution is longer on the left (mean **<** median)
- Transformation: $x^2$ / $x^3$

**Right skewed (Pos)**: Tail of the distribution is longer on the right (mean **>** median) e.g. Exponential Distribution
- Transformation: $log$, $\sqrt{x}$, $\sqrt[3]{x}$, reciprocal ($\frac{1}{x}$)

Fix Both: Cube Root, Box-Cox, Yeo-Johnson

### Central Limit Theorem (CLT)

The distribution of sample (i.i.d.) means will approximate a normal (bell-shaped) distribution as the sample size gets sufficiently large, regardless of the shape of the original population distribution with mean ($\mu$) and finite variance ($\sigma^2$).

For i.i.d. random variables $X_1, X_2, \ldots, X_n$ size $n$:
- sample mean $\bar{X_n} = \frac{1}{n}\sum_{i=1}^n X_i$ and $n$ is large
- then $\bar{X_n} \overset{approx}{\sim} N\left(\mu, \frac{\sigma^2}{n}\right)$
- hence $\frac{\bar{X_n} - \mu}{\sigma / \sqrt{n}} \overset{approx}{\sim} N\left(0, 1\right)$

And
- **Mean of the Sample Means**: The mean of this sampling distribution is $\mu$, which is the unbiased estimator of the population mean.
- **Variance of the Sample Means**: The variance of this sampling distribution is $\sigma^2/n$.
- **Standard Deviation of the Sample Means (Standard Error)**: The standard deviation is $\sigma/\sqrt{n}$. This is called the standard error and measures the typical distance between the sample mean ($\bar{X}$) and the population mean ($\mu$).

Also the unbiased sample variance:

$$ s^2 = \frac{1}{n−1}\sum_{i=1}^{n}(X_i - \bar{X})^2$$

Is not normally distributed and even asymptotically in general.

However for a Normal population $X_i \sim N(\mu,\sigma^2)$ then
$$(n-1)\frac{s^2}{\sigma^2}\sim \chi_{n-1}^2$$
i.e. chi-squared with $n-1$ degrees of freedom. Where $\mathbb{E[s^2] = \sigma^2}$ and $Var(s^2) = \frac{2\sigma^4}{n-1}$.

For non-normal population, $s^2$ is approximately normal for large $n$, then after standardization:
$$\frac{s^2-\sigma^2}{\sqrt{Var(s^2)}} \approx N(0,1) \text{ for large n}$$

And the confidence intervals for the population variance:

$$ Pr\left(\frac{(n-1)s^2}{\chi_{1-\frac{\alpha}{2}}^2} \le \sigma^2 \le \frac{(n-1)s^2}{\chi_{\frac{\alpha}{2}}^2}\right) = 1 - \alpha$$

Where $\chi_p^2$ is the p-th quantile of the chi-square distribution with $n-1$ df.

### Degree of Freedom (ddof)
The number of values in a calculation that are free to vary while estimating a parameter. For example, if we have $n$ numbers with a fixed sum, once we know $n−1$ of them, the last one is determined. So the number of free values = $n – 1$.

In sample variance calculation, dividing by the sample size $n$ instead of $n-1$ (degrees of freedom) underestimates the true population variance, because the sample mean is used, which is closer to the data points than the true mean, leading to smaller squared deviations; using $n-1$ (Bessel's Correction) provides an unbiased estimate by accounting for this tendency, especially crucial with small samples.

### Probability Functions

1. Probability Mass Function (PMF) - describe the probability distribution of a Discrete Random Variable ($X$).
$$P(X=x) \text{ or } f(x)$$

2. Probability Density Function (PDF) - describe the probability distribution of a Continuous Random Variable ($X$).
    - $f(x) \ge 0$
    - $\int_{-\infty}^{\infty} f(x) \, dx = 1$
    - $P(a \le X \le b) = \int_{a}^{b} f(x) \, dx$

3. Cumulative Distribution Function (CDF) - describes the probability that the random variable $X$ will take a value less than or equal to a specific value $x$.
    - $F(x)$
    - Discrete: $F(x) = P(X \le x) = \sum_{t \le x} P(X=t)$
    - Continuous: $F(x) = P(X \le x) = \int_{-\infty}^{x} f(t) \, dt$
    - The PDF is the derivative of the CDF: $f(x) = \frac{d}{dx} F(x)$.
    - The CDF is the integral of the PDF.

4. Joint Probability Distribution - gives the probability that two or more random variables simultaneously take on specific values or fall within a specific range. It is the foundation for calculating marginal and conditional distributions.
$$P(X=x, Y=y)$$

5. Marginal Distribution - a concept used when you have two or more random variables (a Joint Distribution), and you want to focus on the distribution of just one of those variables.
    - Joint PMF of X and Y: $P(X=x, Y=y)$
    - Marginal PMF of X: $P(X=x) = \sum_{y} P(X=x, Y=y)$
    - Joint PDF of X and Y: $f(x, y)$
    - Marginal PDF of X: $f_X(x) = \int_{-\infty}^{\infty} f(x, y) \, dy$

6. Conditional Probability Distribution - describes the probability distribution of one random variable ($X$) given that the other random variable ($Y$) has taken a specific value ($y$).
$$P(X=x | Y=y) = \frac{P(X=x, Y=y)}{P(Y=y)}$$

### Discrete Probability Distributions

| Distribution | Description | Parameters | Key Use Case |
| ------------ | ----------- | ---------- | ------------ |
| Uniform | All outcomes in a finite range are equally likely. | a (min value), b (max value) | Modeling fair dice rolls, generating random samples where every choice is equally probable. |
| Bernoulli | Models a single trial with only two outcomes: success or failure. | p (probability of success) | Modeling a single coin flip, whether an email is opened or not, pass/fail events. |
| Binomial | Models the number of successes in a fixed number (n) of independent Bernoulli trials. | n (number of trials) | p (probability of success) | Quality control (defective items in a batch), predicting the number of customers who click on an ad out of 100 viewers. |
| Poisson | Models the number of events occurring within a fixed interval of time or space, given a known average rate. | λ (lambda, the average rate of occurrence) | Modeling call center volume per minute, number of website errors per hour, car accidents at an intersection per month. |
| Geometric | Models the number of failures before the first success in a sequence of independent Bernoulli trials. | p (probability of success) | Modeling how many times a machine fails before it finally starts, number of attempts needed to solve a puzzle. |
| Hypergeometric | Models the number of successes in a sample drawn without replacement from a finite population. | N (population size), K (number of successes in population), n (sample size) | Sampling inspection (e.g., drawing balls from an urn, checking items in a small batch where sampling affects the remaining probabilities). |
| Negative Binomial | Models the number of failures until a fixed number of successes (r) is achieved. (Generalizes Geometric) | r (target number of successes), p (probability of success) | Modeling the number of games played until a team wins 5 championships. |

### Continuous Probability Distributions

| Distribution | Description | Parameters | Key Use Case |
| ------------ | ----------- | ---------- | ------------ |
| Normal (Gaussian) | The most common distribution. Symmetric, bell-shaped, defined by its mean and standard deviation. | μ (mean), σ (standard deviation) | Modeling natural phenomena (heights, weights, IQ scores), statistical inference (Central Limit Theorem), noise in signals. |
| Uniform | All values within a given range are equally likely, and outside that range, the probability is zero. | a (min value), b (max value) | Modeling situations where little is known about the outcome, such as the error in rounding a measurement to the nearest integer. |
| Exponential | Models the time or distance between events in a Poisson process. It is memoryless. | λ (rate parameter) | Modeling time between customer arrivals, time until a lightbulb burns out, time between bus arrivals. |
| Gamma | A flexible distribution often used to model variables that are always positive and right-skewed. (Generalizes the Exponential distribution). | α (shape parameter), β (rate/scale parameter) | Modeling waiting times (e.g., time to complete n tasks), insurance claim amounts, or rainfall amounts. |
| Beta | Defined on the interval [0, 1]. Highly flexible, used to model probabilities themselves. | α and β (shape parameters) | Modeling probabilities, proportions, or rates (e.g., the proportion of time a machine is down, prior distribution in Bayesian statistics). |
| Student's t | Similar to the Normal distribution but with thicker tails, making it more robust to outliers. | v (degrees of freedom) | Statistical inference, particularly when the sample size is small or the population standard deviation is unknown (e.g., t-tests and confidence intervals). |
| Chi-Squared (χ2) | Sum of squares of independent standard normal random variables. Always positive and right-skewed. | k (degrees of freedom) | Statistical inference: hypothesis testing (goodness-of-fit tests, tests of independence), calculating confidence intervals for population variance. |
| Weibull | Highly flexible distribution used to model failure times or extreme value phenomena. The shape parameter determines its form. | $k$ (shape parameter), $\lambda$ (scale parameter) | Reliability Engineering & Survival Analysis: Modeling the time to failure of mechanical components, equipment life (e.g., bearings, batteries). Used in extreme value theory. |
| Pareto | Used to model phenomena where a large portion of the distribution is concentrated in the small range, and the remainder decays slowly (a heavy-tailed distribution). | $x_m$ (scale parameter, minimum value), $\alpha$ (shape index/tail index) | Economics & Social Science: Modeling wealth distribution (the "80/20 rule," or Pareto principle), city population sizes, size of meteorites, and high-value insurance claims. |

## Goodness of Fit (and checks)

- Kolmogorov–Smirnov test
- Anderson-Darling
- Pearson Chi-Square
- KL Divergence
- AIC
- BIC
- QQ-plots

### Hypothesis Testing

CLT justifies the use of $z$-scores and $t$-scores for conducting hypothesis tests and constructing confidence intervals for the population mean ($\mu$).

| Error Type | Description | H<sub>0</sub> Status | Test Result |
| ---------- | ----------- | ------------ | ----------- |
| Type I (α) | False Positive (Mistakenly rejecting H<sub>0</sub>) | H<sub>0</sub> is True | Rejected H<sub>0</sub> |
| Type II (β) | False Negative (Mistakenly failing to reject H<sub>0</sub>) | H<sub>0</sub> is False | Failed to Reject H<sub>0</sub> |

The significant level (*α*) is the maximum probability of making a Type I error - *incorrectly rejecting true H<sub>0</sub>* that we are willing to tolerate.

The *p-value* is a number that quantifies the evidence against a null hypothesis (H<sub>0</sub>) in a statistical test. It measures how likely it is to observe the test results (or more extreme results) *if the null hypothesis were true*.

*Power* is the probability that the p-value will fall below α when the alternative is true.

Analogy about these concepts in terms of a courtroom trial:
- **Null Hypothesis (H<sub>0</sub>)**: The defendant is innocent.
- **Alternative Hypothesis (H<sub>a</sub>)**: The defendant is guilty.
- **Type I Error (α)**: Convicting an **innocent** person (false positive). The system sets a high standard of evidence (low α) to avoid this.
- **Type II Error (1−Power)**: Letting a **guilty** person go free (false negative).
- **Statistical Power**: The sensitivity of the justice system to correctly convict a truly guilty person.
- **P-value**: The probability of observing the evidence presented (or more extreme evidence) if the defendant was truly innocent (H<sub>0</sub> is true). A very low p-value suggests the evidence is unlikely if H<sub>0</sub> were true.

**Logical Basis**

1. Start with the Assumption (H<sub>0</sub>): In hypothesis testing, always start by assuming the null hypothesis (H<sub>0</sub>) is true. The H<sub>0</sub> usually represents no effect, no difference, or no change (e.g. "The new website design, Variant B, has the same conversion rate as the old design, Variant A").
2. Calculate the P-Value: Based on the sample data, the statistical test calculates the p-value.
3. The Decision:
    - Small p-value (e.g. *p* <= 0.05): This means the observed data would be very unlikely if H<sub>0</sub> were true. Therefore, the data provides strong evidence against H<sub>0</sub>, leading to reject H<sub>0</sub> in favor of the alternative hypothesis (H<sub>a</sub>).
    - Large p-value (e.g. *p* > 0.05): This means the observed data is reasonably likely if H<sub>0</sub> were true. It is said that fail to reject H<sub>0</sub> because there isn't have sufficient evidence to conclude an effect exists.

**P-Value and Statistical Errors**
The p-value is directly relevant to the risk of committing a Type I Error, which is controlled by the significance level (α).

### Power of the test

**Power** is the probability of detecting an effect (i.e. rejecting the null hypothesis) given that some prespecified effect actually exists using a given test in a given context. The **power of the test** is the probability that the test correctly rejects the null hypothesis (H<sub>0</sub>) when the alternative hypothesis (H<sub>a</sub>) is true. It is commonly denoted by 1 - β, where β is the probability of making a Type II error.

The power of a test (1-β) is highly dependent on the **effect size** and the constraints on the **sample size (*n*)**.

**1. Small Sample Sizes:**

| Factor                                      | Challenge                                  | Impact on Power                                                                                                         | Strategy                                                                                                       |
| ------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Small Sample Size (*n*)                     | Limited subjects available.                | Low Power. The test is unlikely to detect a true difference, leading to a high β (Type II Error/False Negative). | Increase α (e.g. from 0.05 to 0.10) to reduce β, but this increases the risk of a false claim. i.e. it will be easier to reject H<sub>0</sub> when it is true. |
| Variability (Standard Deviation, σ) | High patient variability in response.      | Low Power. High σ requires a larger *n* to compensate.                                                          | Use a within-subjects design or highly controlled settings to minimize variability.                            |
| Effect Size (δ)                           | The difference in efficacy might be small. | Low Power. Smaller differences are harder to detect.                                                                    | Focus on finding a large effect size first (e.g. comparing a very effective drug to a placebo).               |

**2. Rare / Unlikely Events**

| Factor                  | Challenge                                                                                          | Impact on Power                                                               | Strategy                                                                                                                                           |
| ----------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Event Rarity            | The "signal" (fraud) is buried in the "noise" (normal transactions).                               | Low Power (if you use a simple random sample).                                | Use imbalanced data techniques (e.g. oversampling the rare class), or use a case-control study design to enrich the sample with the rare event. Look into [SMOTE](https://en.wikipedia.org/wiki/Synthetic_minority_oversampling_technique)  |
| High Stakes (α) | A false positive (α, flagging a legitimate customer as fraudulent) is costly and damaging. | Need to decrease α (e.g. from 0.05 to 0.001), which decreases power. | Accept the lower power (higher β) to prioritize minimizing the Type I Error (False Alarm). This means some fraud will be missed (β) but most customers maybe happier. |

### FDR vs. FPR

- **False Discovery Rate (FDR)** controls the proportion of "discoveries" (rejected null hypotheses) that are actually false positives, crucial in multiple testing, 
    - Out of all my significant findings, what percentage are actually mistakes?

- **False Positive Rate (FPR)** is the per-test probability of incorrectly flagging a true negative as positive, often set at a standard alpha level (e.g., 5%).
    - What's the chance this single test is wrong if negative?

 FDR is less strict than methods controlling the Family-Wise Error Rate (FWER, like Bonferroni), offering more power by accepting some false positives to find more true positives. 

### Multiple Hypothesis Testing Adjustments

When performing multiple statistical tests (e.g. testing 10 different variants in one A/B test, or testing one variant on 5 different metrics), the overall probability of getting at least one false positive (Type I Error) across all tests, known as the **Family-Wise Error Rate (FWER)**, increases dramatically.

The two main adjustment approaches are Family-Wise Error Rate (FWER) control and False Discovery Rate (FDR) control.

### 1. Family-Wise Error Rate (FWER) Control
This aims to control the probability of making even one Type I error among the entire family of tests.

| Method                    | Goal                        | Adjustment                                                                                        | Difference                                                                                                          | Example                                                                                                       |
| ------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Bonferroni Correction     | Control FWER <= α | Adjusts α: New α' = α / n (where *n* is the number of tests).               | **Most Conservative.** Simple to calculate but has the lowest statistical power (most likely to cause a Type II error). | For *n = 5* tests and α = 0.05, only reject H<sub>0</sub> if *p* <= 0.05 / 5 = 0.01.                        |
| Holm-Bonferroni (or Holm) | Control FWER <= α | Step-down procedure: Orders *p*-values and tests against progressively less stringent thresholds. | **Less Conservative (More Powerful)** than Bonferroni, as it rejects more true alternatives.                            | The smallest *p*-value is tested against α / n, the second smallest against α/(n-1), and so on. |

### 2. False Discovery Rate (FDR) Control
This aims to control the expected proportion of false positives among all rejected hypotheses (discoveries). It is a less strict approach than FWER control, allowing for more false positives in trade for greater power to find true effects.

| Method                   | Goal                                                         | Definition                                                          | Difference                                                                                                                                     | Example                                                                                                   |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Benjamini-Hochberg (B-H) | Control FDR <= *q* (The desired FDR level, often *q* = 0.05) | Controls the expected proportion of false rejections (discoveries). | **Less Conservative (Highest Power)**. Widely used in large-scale testing (like genomics, data mining) where a few false positives are acceptable. | An FDR of 0.05 means that to expect at most 5% of the total significant findings to be false positives. |

The key difference is the target:

- FWER Control (Bonferroni, Holm): Focuses on the chance of making a single mistake in the entire set of tests. (High confidence that all significant results are true.)

- FDR Control (Benjamini-Hochberg): Focuses on the proportion of mistakes among the discoveries. (High confidence that most of the significant results are true.)

### Maximum Likelihood Estimation (MLE)
1. Define the Model & Likelihood:
    - Choose a probability distribution (e.g., Normal, Poisson) that might model the data, with parameter(s) ($\theta$) (e.g., mean ($\mu$), rate ($\lambda$)).
    - Write down the Probability Density Function (PDF) or Probability Mass Function (PMF) for a single data point, $f(x_{i}|\theta)$.
    - For independent and identically distributed (i.i.d.) data ($x_{1},\dots ,x_{n}$), the Likelihood Function, $L(\theta |x)$, is the product of these PDFs: $L(\theta |x)=\prod_{i=1}^{n}f(x_{i}|\theta)$
2. Transform to Log-Likelihood:
    - Take the natural logarithm of the Likelihood Function to get the Log-Likelihood Function, $\ell (\theta |x)=\ln (L(\theta |x))=\sum_{i=1}^{n}\ln (f(x_{i}|\theta ))$. This makes differentiation easier and converts products to sums. 
3. Differentiate & Find the Score Function:
    - Calculate the first derivative of the log-likelihood with respect to $\theta$: $\frac{\partial \ell }{\partial \theta }$. This is the Score Function.
4. Solve for the MLE Estimator ($\^{\theta }$):Set the Score Function to zero: $\frac{\partial \ell }{\partial \theta }=0$.
    - Solve this equation for $\theta$ to find the value that maximizes the likelihood, which is your Maximum Likelihood Estimator, $\^{\theta }$.

### Bayesian Statistics

- Frequentist statistics relies solely on observed data and long-term frequencies, often ignoring prior knowledge. It uses point estimates and hypothesis testing with p-values, which can lead to rigid decisions.
- Bayesian statistics incorporates prior beliefs and updates them as data accumulates, offering more nuanced probability statements. This is especially useful for unique events or when data is limited.

**Bayesian Inference**
$$\overbrace{P(\theta|X)}^{\text{posterior}} = \frac{P(\theta,X)}{P(X)} = \frac{\overbrace{P(X|\theta)}^{\text{likelihood}} \overbrace{P(\theta)}^{\text{prior}}}{\underbrace{P(X)}_{\text{marginal likelihood}}}$$

Where:

- $P(\theta∣X)$ is the **posterior** probability the updated belief after observing the data.
- $P(X∣\theta)$ is the **likelihood** the probability of observing the data given the hypothesis.
- $P(\theta)$ is the **prior** probability, our initial belief about the hypothesis before observing the data.
- $P(X)$ is the **marginal likelihood** a normalizing constant that ensures the posterior probability sums to 1.

Example:
1. Likelihood Function
The Bernoulli likelihood function is used for binary outcomes like success or failure (for a single trial).

$$ P(X|\theta) = \theta^x \cdot (1 - \theta)^{1-x} $$

Where:
- X represents the observed data (0 for failure and 1 for success).
- $\theta$ is the probability of success (e.g., click rate).
- x is the observed outcome (0 for failure, 1 for success).

2. Prior Distribution

Distribution of $\theta$ based on prior knowledge/assumption. A commonly used probability parameter is the Beta distribution which is used as the prior distribution for parameters like 
$\theta$. (a conjugate prior for the Binomial likelihood - sequence of independent Bernoulli trials)

$$ P(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)} $$

Where:
- $\theta$ represents the probability of success.
- $\alpha$ and $\beta$ are parameters that control the shape of the Beta distribution.
- $B(\alpha, \beta)$ is the Beta function which ensures the distribution integrates to 1.

3. Posteria Distribution

Use Bayes’ Theorem to update our beliefs once new data $P(X∣\theta)$ is available. The updated belief is represented by the posterior belief distribution $P(\theta∣X)$ which combines the prior $P(\theta)$ belief and the new evidence.

$$ P(\theta|X) \propto P(X|\theta) \times P(\theta) $$


**Maximum A Posteriori (MAP)**

The Maximum A Posteriori (MAP) estimate is an estimate of an unobserved quantity (like a probability or a parameter) that is derived from the posterior distribution. It represents the single value of the parameter that is considered most probable given both the observed data and the prior information.

$$\hat{\theta}_{\text{MAP}} = \underset{\theta}{\operatorname{argmax}} \text{ } P(\theta | X)$$

Therefore, finding the peak of the posterior means finding the value of $\theta$ that maximizes the product of the likelihood and the prior:
$$\hat{\theta}_{\text{MAP}} = \underset{\theta}{\operatorname{argmax}} \left[ P(X | \theta) \cdot P(\theta) \right]$$

- Likelihood $P(X | \theta)$: This describes how well the parameter value $\theta$ explains the observed data $D$.
- Prior $P(\theta)$: This describes your initial beliefs about the parameter value 8$\theta$ before seeing any data.

The MAP estimate is the result of balancing the evidence from the data (the likelihood) with your initial beliefs (the prior).

**MAP vs. MLE**

| Feature | Maximum Likelihood Estimate (MLE) | Maximum A Posteriori (MAP) Estimate |
| ------- | --------------------------------- | ----------------------------------- |
| Formula | $\hat{\theta}_{\text{MLE}} = \underset{\theta}{\operatorname{argmax}}\text{ }\mathcal{L}(\theta \| X)$ | $$\hat{\theta}_{\text{MAP}} = \underset{\theta}{\operatorname{argmax}}\text{ }P(\theta \| X)$$ |
| Considers | Data only (maximizes the likelihood) | Data and Prior (maximizes the posterior) |
| Sensitivity | Highly sensitive to small data sets | Less sensitive to small data sets (smoothed by the prior) |
| Relationship | MLE is the same as the MAP estimate when the prior is uniform (i.e., $P(\theta)$ is constant). |

**Conjugate Prior**

A prior distribution is called conjugate to the likelihood function if the resulting posterior distribution belongs to the same family of distributions as the prior distribution.

In simpler terms, if start with a prior from a specific family (e.g., Beta) and the data is generated by a specific process (e.g., Binomial likelihood), the posterior will also be from that same family (e.g., Beta).

The primary reason to use a conjugate prior is mathematical tractability and computational efficiency. Conjugate priors allow the posterior distribution to be calculated in closed form (with an exact, simple equation). This means no complex computational methods needed. In BHT, calculating the marginal likelihood $P(e|H)$ is crucial for the Bayes Factor. With a conjugate prior, this marginal likelihood can often be calculated analytically, avoiding complex numerical integration.

Known conjugate priors pairs (prior-likelihood): Beta-Binomial, Normal-Normal, Gamma-Poisson

**Markov Chain Monte Carlo (MCMC)**

Markov Chain Monte Carlo (MCMC) refers to a class of algorithms (like the Metropolis-Hastings algorithm or Gibbs Sampling) used to sample from a target probability distribution.

In the context of Bayesian statistics, the target distribution is the posterior distribution, $P(\text{Parameters} | \text{Data})$.

The core idea is:
1. Start at a random point in the parameter space.
2. Propose a new point (a "move").
3. Accept or reject the move based on the target distribution's density at the new point.
4. Repeat this thousands of times, creating a "chain" of samples.
5. After the chain has run long enough (past the "burn-in" period), the distribution of these samples will accurately represent the true posterior distribution.

### Bayesian Hypothesis Testing

Bayesian hypothesis testing is fundamentally about updating the degree of belief in a hypothesis as new data are collect. It treats the unknown population parameters (like the true conversion rate, *p*) as random variables with a probability distribution.

The entire framework is centered on Bayes' Theorem:

$$P(H|e) = \frac{P(e|H)P(H)}{P(e)}$$

- **$P(H|e)$ - Posterior**: How probable is the hypothesis given the observed evidence (not directly computable)
- **$P(e|H)$ - Likelihood**: How probable is the evidence given that the hypothesis is true?
- **$P(H)$ - Prior**: How probable was the hypothesis *before* observing the evidence?
- **$P(e)$ - Marginal/Evidence**: How probable is the new evidence under all possible hypothesis? $P(e) = \sum P(e|H_i)P(H_i)$

Key Concepts in Bayesian A/B Testing

| Concept                          | Explanation                                                                                                                                                                        | Frequentist Analog                                                                     |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Prior Distribution               | The initial belief about the possible value of a parameter (e.g., conversion rate *p*). This can be non-informative (like a coin flip) or informative (based on historical data). | N/A (Frequentist starts with the Null Hypothesis).                                     |
| Posterior Distribution           | The updated belief in the parameter after observing the data. It is a full probability distribution showing the entire range of likely values for the true parameter.              | Point Estimate and *p*-value.                                                          |
| Probability of Superiority       | The direct probability that one variant's true parameter (e.g., p<sub>B</sub>) is greater than another's (p<sub>A</sub>). Calculated by comparing their posterior distributions.                   | *p*-value (indirect measure of evidence against the null).                             |
| Credible Interval                | The range of values where the true parameter lies with a certain probability (e.g., "There is a 95% chance the true uplift is between *X* and *Y*").                               | Confidence Interval (tells you what would happen if you repeated the test many times). |
| Bayes Factor (BF<sub>10</sub>) | A ratio quantifying the evidence the data provides for the Alternative Hypothesis (H<sub>a</sub>) relative to the Null Hypothesis (H<sub>0</sub>).                                                 | *p*-value (which only measures evidence against H<sub>0</sub>).                                |

The Bayes Factor (BF<sub>10</sub>)The Bayes Factor is the Bayesian analogue to the *p*-value and provides a clear measure of evidence:

$$\text{BF}_{10} = \frac{P(e \mid H_a)}{P(e \mid H_0)}$$

The ratio between the probability observing the data given the alternative hypothesis H<sub>a</sub> vs. the probability observing the data given the null hypothesis H<sub>0</sub>.

| Bayes Factor (BF<sub>10</sub>​) | Interpretation (Evidence for H<sub>a</sub>​)                                                             |
| -------------------- | --------------------------------------------------------------------------------------------- |
| > 10               | Strong Evidence for the Alternative Hypothesis (H<sub>a</sub> is 10x more likely than H<sub>0</sub>). |
| 1 to 3           | Anecdotal evidence for H<sub>a</sub>.                                                                 |
| ~ 1         | No meaningful evidence; data is ambiguous.                                                    |
| < 1/3              | Evidence supports the Null Hypothesis (H<sub>0</sub> is 3x more likely than H<sub>a</sub>).           |

**Power Analysis in Bayesian Testing**

The concept of statistical power (the long-run probability of correctly rejecting a false H<sub>0</sub>) does not apply in the same way because:
- 1. No fixed α: Bayesian testing does not have a fixed Type I error rate (α) defined before seeing the data.
- 2. No fixed *n* required: Bayesian tests can be monitored continuously (sequential testing) and stopped whenever the evidence (Bayes Factor or Probability of Superiority) crosses a pre-defined decision threshold.

Instead of traditional power analysis, Bayesian practitioners use methods aimed at experiment planning and design:

1. Bayes Factor Design Analysis (BFDA)

BFDA is the Bayesian way to determine the sample size *n* needed to achieve a desired strength of evidence.
- Goal: Determine the *n* required to make a decision with a high probability, for a given true effect size.
- Method: Simulate data under the assumption that a true effect exists (e.g., a conversion lift of 1%) and see how many samples (*n*) are needed for the resulting Bayes Factor (BF<sub>10</sub>) to cross the decision threshold (e.g., BF<sub>10</sub> > 10).

2. Sequential Testing (Stopping Rules)
The most common application in A/B testing is defining a stopping rule based on the results, rather than a fixed *n*.
- Rule Example: Stop the test as soon as the Probability of Superiority for Variant B remains above 98% for three consecutive days, OR when the Credible Interval for the difference excludes zero entirely.
- Advantage: This allows for early stopping if the effect is large and clear, or continuing if the evidence is ambiguous, making the test much more efficient. This is statistically safe in the Bayesian framework, unlike the frequentist approach which requires complex correction methods to maintain its α guarantee when checking results early.

### Survival vs. Hazard Models

**Survival Function (S(t))** - measures the cumulative probability of non-occurrence (e.g. not churn/dead). Provide the probability that an individual survives past time *t* i.e. hasn't experienced the even yet. Use to see overall survival curves, compare group's general survival pattern (e.g. What percentage of patients are still alive after 5 years?)

**Hazard Function (h(t))** - measures the instantaneous rate of *occurrence* (intensity of an event). The instantaneous rate (or risk) of the event occurring at time *t*, given the individual has survived up to time *t*. (Not a probability) Use to understand why survival differs and how factors influence the rate of the event. (e.g. Does Drug X halve the risk of event compared to placebo?)

# Machine Learning

### Missing Data

| Mechanism | Definition | Example |
| --------- | ---------- | ------- |
| Missing Completely at Random (MCAR) | The probability of data being missing is unrelated to both the observed and unobserved data. | A survey is accidentally dropped and coffee is spilled on a random page, making a set of answers unreadable. The missingness is a random event. |
| Missing at Random (MAR) | The probability of data being missing is systematically related to the observed data, but not the missing data itself. | Older survey respondents are less likely to report their income, but the likelihood of their income being missing does not depend on the actual value of their income (after accounting for age). |
| Missing Not at Random (MNAR) | The probability of data being missing is systematically related to the unobserved data (the value that is actually missing). | Individuals with very high or very low incomes are less likely to report their income. The missingness is dependent on the income value itself. |

| Method | Description | Pros | Cons |
| ------ | ----------- | ---- | ---- |
| Listwise Deletion (Complete-Case Analysis) | Excludes any case (row) that has any missing value in any variable relevant to the analysis. | Simple and unbiased if data are MCAR. | Leads to a significant loss of statistical power and potential bias if data are MAR or MNAR. |
| Pairwise Deletion (Available-Case Analysis) | Uses all available data for a specific analysis (e.g., only cases with non-missing values for two variables are used to calculate their correlation). | Utilizes more data than listwise deletion. |Statistical estimates are based on different subsets of data, which can lead to non-sensical or inconsistent results. Biased under MAR. |
| Mean/Median/Mode Imputation | Replaces missing values with the mean (for continuous data), median (less affected by outliers), or mode (for categorical data) of the observed values for that variable. | Simple, fast, and easy to implement. | Underestimates variance (standard errors are too small), distorts the shape of the variable's distribution, and can bias estimates, especially for MAR or MNAR data. |
| Regression Imputation | Missing values are predicted using a regression model based on other variables in the dataset. | Uses information from other variables, maintaining the relationship between the imputed variable and the predictors. | Still a single value, so it underestimates variance (standard errors are too small) and can make relationships between the imputed variable and non-predictor variables artificially stronger. |
| Last Observation Carried Forward (LOCF) | For longitudinal/time series data, the last observed value is used as the imputation for subsequent missing data points. | Simple, commonly used in clinical trials. | Only appropriate when the assumption that the value did not change is reasonable; can introduce significant bias if the underlying trend is changing. |
| Multiple Imputation (MI) | The process is repeated multiple times (typically 5-50): 1. Impute (create M complete datasets, each with different plausible imputed values). 2. Analyze (run the desired analysis on each of the M datasets). 3. Pool (combine the results into a single set of estimates and standard errors). | Best general-purpose method for MAR data. Provides unbiased estimates for parameters and accurate standard errors, reflecting the uncertainty of imputation. | More complex to implement and computationally intensive, requiring specialized software packages. The method is sensitive to the imputation model. |
| Full Information Maximum Likelihood (FIML) | A model-based approach that estimates the parameters of a statistical model directly from the incomplete data, effectively treating the missing values as parameters to be estimated. | Highly efficient and yields unbiased estimates under the MAR assumption. Does not impute data, so you get one set of results. | Only works for specific types of models (often structural equation models) and is computationally expensive for large datasets or complex models. |


### Evaluation metrics

**Classification**

- **Accuracy**: The most intuitive metric, it is the ratio of correct predictions to the total number of predictions. It can be misleading if the dataset is imbalanced (e.g., 98% of cases are in one class).
- **Precision (Positive Predictive Value)**: Measures the proportion of positive identifications that were actually correct. It is useful in cases where the cost of a false positive is high.

$$ \frac{TP}{TP + FP}$$

- **Recall (Sensitivity or True Positive Rate)**: Measures the proportion of actual positives that were identified correctly. It is useful when the cost of a false negative is high (e.g., missing a disease diagnosis).

$$ \frac{TP}{TP + FN}$$

- **F1-Score**: The harmonic mean of precision and recall. It provides a single score that balances both concerns and is a good general measure for imbalanced classes.
- **Confusion Matrix**: A table that visualizes the performance by breaking down predictions into True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
- **ROC Curve and AUC**: The Receiver Operating Characteristic curve shows the trade-off between the True Positive Rate and False Positive Rate at various threshold settings. The Area Under the Curve (AUC) is a single measure of a model's overall ability to distinguish between classes. 

- **Akaike Information Criterion (AIC)**: measures pure predictive fit → rewards flexibility
    - If main goal is prediction, and are okay with a slightly more complex model if it improves predictive accuracy.
$$ AIC=2 \cdot k−2 \cdot log L $$

- **Bayesian Information Criterion (BIC)**: penalizes complexity → good for distribution selection
    - If goal is to identify the "true" model or if a large dataset and want to strongly penalize complexity to avoid overfitting.
$$BIC=k \cdot ln(n)−2 \cdot log L$$

Where:
- $k$ = number of parameters
- $n$ = sample size
- $L$ = maximized likelihood of the model

**Regression**

| Assumption | What It Means | Why It Matters |
| ---------- | ------------- | -------------- |
| Linearity | The relationship between X and Y is linear in the parameters (β). | If violated, the model is misspecified and the predictions are biased. |
| No Perfect Multicollinearity | Independent variables (X's) are not perfectly correlated with each other. | If violated, the model cannot be solved (matrix is singular), leading to infinite coefficient variance. |
| Exogeneity of Errors | The error term (ϵ) has an expected mean of zero, conditional on the predictors X. $\mathbb{E}[\epsilon \| X] = 0$. |  If violated (endogeneity), predictors link to unobserved factors, biasing results. | 
| Homoscedasticity | The variance of the errors is constant across all levels of the independent variables. $Var[\epsilon_i​]=\sigma^2$. | If violated, OLS estimates are still unbiased, but they are no longer the most efficient (BLUE). Standard errors are incorrect. |
| No Autocorrelation | The error terms are independent of each other (especially important for time series data). $Cov(\epsilon_i​,\epsilon_j​)=0$ for $i\ne j$. | If violated, OLS estimates are still unbiased, but standard errors are incorrect. |
| Normality of Errors | The errors are normally distributed. $\epsilon∼N(0,\sigma^2)$. | Necessary for calculating t-statistics, p-values, and confidence intervals. |
| Sufficient Sample Size | The number of observations (n) must be greater than the number of parameters (k). | Basic requirement for solvability. |

- **Mean Absolute Error (MAE)**: The average of the absolute differences between the predicted values and the actual values. It gives an idea of the typical error magnitude.
- **Mean Squared Error (MSE)**: The average of the squared differences between predicted and actual values. This metric penalizes large errors more heavily than MAE.
- **Root Mean Squared Error (RMSE)**: The square root of the MSE. It is in the same units as the target variable, making it more interpretable than MSE.- R-squared (R<sup>2</sup>): Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A higher value indicates a better fit. 

- **SST (Total Sum of Squares)**: Total variation in the dependent variable ($y$), calculated as the sum of squared differences between each observed value and the mean of all observed values.
- **SSR (Sum of Squares Regression)**: Variation in the dependent variable explained by the regression model, calculated as the sum of squared differences between the predicted values ($\hat{y}$) and the mean of the observed values ($\bar{y}$).
- **SSE (Sum of Squares Error/Residuals)**: Unexplained variation (error), calculated as the sum of squared differences between the observed values ($y$) and the predicted values ($\^{y}$). 
- $SST = SSR + SSE$
- R-squared ($R^{2}$): This metric, indicating model fit, is derived from these sums: $R^{2}=\frac{SSR}{SST}$. A higher $R^{2}$ (closer to 1) means the model explains a larger proportion of the total variance.
- Adjusted R-squared (adj.$R^{2}$): A modified R-squared that accounts for the number of predictors in a regression model, penalizing for adding useless variables, making it better for comparing models with different numbers of independent variables.
- adj.$R^{2} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$ $n$-number of samples, $p$-number of predictors/features

### L1 vs. L2 regularization

**L1 (LASSO)**

- Penalty: Adds a penalty proportional to the sum of the **absolute** values of the coefficients (weights) to the loss function.
- Resulting model: Produces sparse models because it **tends to set the coefficients of less important features to exactly zero**.
- Use case: **Ideal for feature selection**, especially when you have a large number of features and suspect many of them are irrelevant.
- Constraint shape: Creates a diamond or square-shaped constraint, which has sharp corners that are more likely to intersect with the axes at zero. 

**L2 (Ridge)**

- Penalty: Adds a penalty proportional to the sum of the **squares of the coefficients** to the loss function.
- Resulting model: **Encourages smaller, but generally non-zero** coefficients for all features, leading to a **less sparse, more stable model**.
- Use case: Preferred when you believe most features are relevant and want to shrink their weights to prevent a few from having an undue influence, reducing overall variance. It is also more robust to correlated features.
- Constraint shape: Creates a circular or elliptical constraint, which gradually shrinks all weights without forcing any single one to be zero. 


| $\lambda$ (complexity) | Effect on coefficients | Effect on model |
| -------------- | ---------------------- | --------------- |
| **0 (small)**  | Almost no shrinkage                              | Equivalent to OLS, can overfit if p is large |
| **Moderate**   | Coefficients shrink toward 0                     | Reduces variance, slightly biased estimates  |
| **Large**      | Coefficients shrink a lot (but not exactly zero) | Simpler model, high bias, low variance       |


**Key Components**
| Purpose | Linear Regression | GLM (Logistic, Poisson, Gamma) |
| ------- | ----------------- | ------------------------|
| Model fit (“variance explained”) | $R^2$, adj. $R^2$ | Pseudo-$R^2$, deviance |
| Compare nested models | F-test | Likelihood Ratio $\chi^2$ test |
| Test coefficients | t-test | z-test / Wald test |
| Goodness-of-fit | residual plots | Deviance, Pearson $\chi^2$ |
| Check dispersion | N/A | Pearson $\chi^2$ / df |
| Model selection | AIC, BIC (valid) | AIC, BIC (preferred) |

**ANOVA**

Analysis of Variance - test whether three or more groups have the same mean.

| Source | SS | df | MS | F | p |
| ------ | -- | -- | -- | - | - |
| Between groups | SS_B | k−1 | MS_B = SS_B/(k-1) | F = MS_B / MS_W | p |
| Within groups | SS_W | N−k | MS_W = SS_W / (N-k) | |
| Total | SS_T | N−1 | | | |

**F-test**
Statistical test used inside ANOVA.

$$F = \frac{\text{Between-group Variance}}{\text{Within-group Variance}} $$

- If groups have similar means → numerator ≈ denominator → F close to 1.
- If at least one group mean differs → numerator >> denominator → large F → small p-value.
-Reject $H_0$ (all means equal) if: $p \lt \alpha$

In Regression:
$$F = \frac{\text{Model Mean Square (MSM)}}{\text{Residual Mean Square (MSE)}} $$

- MSM ≈ MSE → predictors explain nothing → F ≈ 1
- MSM >> MSE → predictors reduce error → F large → model significant

| Source | Sum of Squares | Degree of Freedom (df) | Mean Squares | F-Statistics |
| ------ | -- | -- | -- | - |
| Model | SSR | p | MSM=SSR/p | MSM/MSE |
| Residual | SSE | n-p-1 | MSE=SSE/(n-p-1) |
| Total | SST | n-1 | | |

- **SSR (Sum of Squares Regression)**: Variation explained by the regression line between $\hat{y}$ and the mean $\bar{y}$
    - $SSR = \sum (\hat{y_i} - \bar{y})^2$
- **SSE (Sum of Squares Error)**: Unexplained variation (residuals, between observed $y_i$ and predicted $\hat{y}$)
    - $SSE = \sum (y_i - \hat{y_i})^2$
- **SST (Sum of Squares Total)**: Total variation in the dependent variable ($Y$) from its mean ($\bar(y)$)
    - $SST = \sum (y_i - \bar{y})^2 = SSR + SSE $

- **MSM/MSR** (Mean Square Regression/Model) tells how strong the model is.
- **MSE** (Mean Square Error/Residual) tells how noisy the data is.
- F-statistic compares these two to test:
$$H_0 : \beta_1 = \beta_2 = \cdot\cdot\cdot = \beta_p = 0 $$

| Test | Answers | When Used |
| ---- | ------- | --------- |
| t-test | “Is this single β significant?” | Regression coefficients |
| F-test (regression) | “Is the model useful at all?”	 | Overall model test |
| ANOVA (F-test) | “Do multiple group means differ?” |Group comparison |
| Partial F-test | “Does adding variables improve the model?” | Model comparison |
| Chi-square test | GLMs where likelihood ratio asymptotically → $\chi^2$ | Logistic, Poisson, etc. |

| Scenario | Test | Null Hypothesis | Notes |
| -------- | ---- | --------------- | ----- |
| Two groups (treatment vs control) | t-test / regression t-test | β1 = 0 | Classic two-group test |
| Multiple groups (≥3) | ANOVA / F-test | β1 = β2 = … = 0 | Overall effect |
| Identify which group differs | t-tests or contrasts | βj = 0 | Adjust for multiple comparisons |
| Non-parametric alternative | Kruskal-Wallis | Group medians equal | When normality is violated |

Hypothesis Testing Regression: t-test vs Bootstrap vs Permutation

| Feature | Classical t-test | Bootstrap | Permutation |
| ------- | ---------------- | --------- | ----------- |
| Purpose | Test if a parameter (mean, coefficient) ≠ null | Empirically estimate p-values & SE | Empirically estimate p-value under null |
| Null hypothesis | H0: parameter = 0 (or specified value) | Same | Same |
| Assumptions | - Normal errors (small n), Independent observations, Low collinearity, Correctly specified model | Minimal; sample representative of population | Minimal; observations exchangeable under H0 |
| How it works | Compute t-statistic = estimate / SE; compare to theoretical t-distribution | Resample rows with replacement B times; compute statistic each resample; p-value = fraction ≥ observed | Shuffle labels under H0 B times; compute statistic each permutation; p-value = fraction ≥ observed |
| What it tests | Parameter significance | Parameter significance accounting for sample variability | Parameter significance under null, robust to dependence |
| Handles small sample? | N | Y | Y |
| Handles correlated predictors? | N | Y | Y |
| Handles non-normal errors? | N | Y | Y |
| Computational cost | Low | Medium–High | Medium–High |
| Pros | Fast, simple, interpretable | Robust, empirically accurate SE & p-value | Robust, exact null distribution, minimal assumptions |
| Cons | Sensitive to small n, collinearity, non-normality | Computationally intensive; model still assumed reasonable | Computationally intensive; requires exchangeable observations |
| Use case | Large sample, independent predictors, normal errors | Small sample, multicollinearity, complex model | Small sample, correlated predictors, non-parametric / ML hypothesis testing |

Classical $t$ shows the theoretical null, permutation shows the empirical null, and bootstrap shows the observed effect variability; the more the bootstrap distribution lies beyond the null's rejection region, the higher the power, regardless of its exact center.

**Generalized Linear Models**

| GLM | Random Component (Distribution) | Canonical Link Function | Best for |
| --- | ------------------------------- | ----------------------- | -------- |
| Normal | Gaussian (Normal) | Identity (μ=η) | Continuous, Unbounded Data (e.g., height, temperature, sales volume). This is standard Ordinary Least Squares (OLS) Linear Regression. |
| Logistic | Binomial | Logit ($\log\frac{\mu}{1-\mu} = \eta$) | Binary Outcomes (e.g., 0/1, Yes/No, Pass/Fail, Spam/Not Spam). |
| Poisson | Poisson | Log ($\log(\mu) = \eta$) | Count Data (e.g., number of clicks, number of accidents, number of insurance claims). Assumes mean = variance (equidispersion). |
| Negative Binomial | Negative Binomial | Log ($\log(\mu) = \eta$) | Overdispersed Count Data (where variance > mean). Used as a robust alternative to Poisson regression. |
| Gamma | Gamma | Inverse ($\frac{1}{\mu} = \eta$) | Continuous, Positive, Skewed Data (e.g., waiting times, financial claims size, duration). Often used when variance increases with the mean. |
| Inverse Gaussian | Inverse Gaussian | Inverse Squared ($\frac{1}{\mu^2} = \eta$) | Highly Skewed Continuous, Positive Data (e.g., duration of processes with heavy tails). |

- **Deviance** ($D$): 
$ 2\cdot [\log (L_{\text{Saturated}})-\log (L_{\text{Fitted}})] $

- **Akaike Information Criterion (AIC)**: $\text{AIC} = 2k - 2 \log(\mathcal{L}_m)$
- **Overdispersion ($\phi > 1$)**: Occurs when the observed variance of the response variable is greater than the variance predicted by the assumed distribution (especially in Poisson ($\mu$ = $\sigma^2 = \lambda$) and Binomial models). The ratio of Residual Deviance to Residual Degrees of Freedom is significantly greater than 1 ($\phi \gg 1$).
- **Dispersion Parameter ($\phi$)**: A scaling factor that corrects the standard errors in the presence of overdispersion.

| Model Name | Key Concept | What it is for |
| ---------- | ----------- | -------------- |
| Quantile Regression (QR) | Models the relationship between predictors (X) and a specific quantile of the response variable (Y). | Robustness: Median regression (the 50th percentile) is far more robust to outliers than Mean (OLS) regression. Non-Homogeneity: Allows you to model how predictors affect different parts of the response distribution (e.g., modeling factors that affect the 10th percentile of income vs. the 90th percentile). |
| Median Regression | This is Quantile Regression specifically focusing on the 50th percentile (the median). | Robustness to Skew/Outliers: If the residual distribution is highly skewed or contains severe outliers, the median provides a more stable and representative measure of central tendency than the mean. |
| Generalized Additive Models (GAMs) | Extends GLMs by replacing the linear predictor terms ($\beta_i​X_i​$) with flexible smoothing functions ($f_i​(X_i​)$). | Non-Linear Relationships: Captures complex, non-linear, and non-monotonic relationships between predictors and the outcome without having to manually specify polynomial terms (like X2 or X3). Interpretability: Unlike black-box models (like neural networks), the effect of each predictor is plotted as a smooth curve, maintaining some interpretability. |
| LOESS (or LOWESS) | A non-parametric method that fits a series of local polynomial regressions to small, overlapping subsets of the data. | Visualization/Exploration: Primarily used for exploratory data analysis (EDA) and smoothing time series. It creates a smooth curve without assuming any global functional form (linear, quadratic, etc.) for the entire dataset. Prediction: Less common for formal prediction as it's computationally intensive and sensitive to the chosen "span" (the size of the local subset). |
| Finite Mixture Models (FMMs) | Assumes the entire population is composed of a finite number of unobserved (latent) sub-populations or "classes" and the response variable Y follows a different regression model within each class. | Heterogeneity: When you suspect your data contains distinct groups that follow different underlying processes. The model simultaneously estimates the parameters for each latent class and the probability that any given observation belongs to each class. Example: Modeling customer spending where one class is "low-spenders" and another is "high-spenders" each driven by different factors. |

| Test Name | Distribution | Primary Role in OLS | What it Tests |
| --------- | ------------ | ------------------- | ------------- |
| t-Test | t-Distribution | Individual Coefficient Significance | Tests the null hypothesis that a single coefficient ($\beta_i$​) is equal to zero, holding all other variables constant. |
| F-Test (ANOVA) | F-Distribution | Overall Model Significance & Group Significance | Overall: Tests the null hypothesis that all regression coefficients are zero ($\beta_1​=\beta_2​=⋯=0$). Groups: Tests the null hypothesis that a subset of coefficients is simultaneously zero (e.g., comparing a model with 5 variables to one with 2). |
| $\chi^2$ Test | $\chi^2$-Distribution | Diagnostics and Model Fit (via GLMs) | Tests goodness-of-fit for GLMs, checks OLS assumptions (Normality, Homoscedasticity), or tests independence between categorical variables. |

**Clustering**

- **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters.
- **Davies-Bouldin Index**: Measures the ratio of within-cluster scatter to between-cluster separation.
- **Calinski-Harabasz Index**: Measures the ratio of between-cluster variance to within-cluster variance.
- **Intraclass Correlation Coefficient (ICC)**: Statistical measure that quantifies the degree of similarity between observations within the same group or cluster. It ranges from 0 to 1, where 1 indicates perfect agreement and 0 indicates no agreement.


### Tree Types Algorithms

**Bagging (Bootstrap aggregating) vs. Boosting**

| Methods | Bagging | Boosting |
| ------- | ----------------------------- | ---------------------------------- |
| Algorithms | Random Forest | AdaBoost, XGBoost |
| Training | Parallel (models trained independently) | Sequential (models trained one after the other) |
| Goal | Reduce Variance (e.g., combat overfitting) | Reduce Bias (e.g., combat underfitting) |
| Data Usage | Each model trained on a bootstrap sample (random sampling with replacement). | Each model trained on the entire dataset (or a variation), with weights adjusted to emphasize previously misclassified points. |
| Model Weighting | All base models (weak learners) are generally equally weighted in the final prediction (majority vote/simple average). | Base models are weighted based on their performance; better-performing models get higher weight. |
| Base Model Type | Often uses complex/unstable models (e.g., deep decision trees) that have high variance. | Often uses simple/weak models (e.g., shallow decision trees) that have high bias. |
| Training Speed | Generally faster due to parallelizable training. | Generally slower due to sequential, dependent training. |

**Gini Impurity** - Quantifies the probability of misclassifying a randomly chosen element in the dataset if it were randomly labeled according to the class distribution in the node.

$$ 1 - \sum_{i=1}^{C} p_i^2 \text{ where p is the proportion of class i}$$

**Entropy** - Measures the uncertainty or randomness in a set of data. It quantifies the average amount of information needed to classify a sample in the node.

$$ - \sum_{i=1}^{C} p_i \log_2(p_i) \text{ where p is the proportion of class i}$$

**Algorithms**
| Algorithms | Decision Tree (DT) | Random Forest (RF) | Boosted Trees (BT) & Gradient Boosting Machines (GBM) |
| ------- | ------------------ | ------------------ | ------------------ |
| Ensemble Type | None (Single Model) | Bagging (Parallel) | Boosting (Sequential) |
| Common Examples | CART, ID3, C4.5 | Random Forest, Bagging Classifier/Regressor | AdaBoost, XGBoost, LightGBM, CatBoost |
| Goal | Achieve high purity/low error in splits. | Reduce Variance (address overfitting). | Reduce Bias (address underfitting). |
| Training | Single pass, recursive splitting features. | All trees built independently. (random subset features) | Trees built sequentially, correcting errors. |
| Speed | Fast | Parallelizable (Fast) | Sequential (Slower to train) |
| Interpretability | High (Easy to visualize) | Low | Low |
| Risk of Overfitting | High | Low | Moderate (if not well-tuned) |

- Random Forest (RF):
    - Uses bootstrapped samples (Bagging)
    - Considers a random subset of features (columns) at every split point. This de-correlates the individual trees, making the ensemble's prediction much more robust.
- AdaBoost (Adaptive Boosting):
    - Adjusting the weights of the misclassified data points, forcing subsequent models to focus on them.
    - Assigning higher weights to the weak learners that performed better during their training.
- Gradient Boosting Machines (GBM):
    - Builds new models that target the residuals (the errors or differences between the actual and predicted values) of the previous models.
    - It uses the concept of gradient descent to minimize the loss function.
- Extreme Gradient Boosting (XGBoost):
    - An optimized and highly scalable implementation of Gradient Boosting.
    - Exceptional in speed and performance supporting features:
        - Regularization (L1 and L2) to prevent overfitting.
        - Parallel processing of tree construction.
        - Handling of missing values.
- LightGBM & CatBoost:
    - Highly efficient variants of Gradient Boosting that are optimized for handling large datasets and categorical features, respectively.

### Bias-Variance Tradeoff

**Bias**
The error from a model's simplifying assumptions. A high bias model is a poor fit for the data because it's too simple.

Result: Underfitting, where the model fails to capture important patterns.

Example: Using a linear model to predict a non-linear relationship. 

**Variance**
The error from a model's sensitivity to the specific training data. A high variance model fits the training data very closely, including the noise.

Result: Overfitting, where the model performs well on the training data but poorly on new, unseen data.

Example: A very complex model with many parameters that learns the "noise" in the training data. 

**The Tradeoff**

Inverse relationship: As a model's complexity increase, bias decreases, but variance increases.

Finding the sweet spot: The goal is to find the model complexity that minimizes the sum of bias and variance, leading to the best performance on unseen data.

Total error: The total error of a model can be thought of as a combination of bias, variance, and irreducible error (noise inherent in the data). 

**How to manage the tradeoff**

Increase training data: A larger dataset can help reduce variance without a significant increase in bias.

Use regularization: Techniques like L1 and L2 regularization can penalize model complexity, helping to reduce variance.

Ensemble methods: Combining multiple models can reduce variance and improve overall performance. 

### Multiclass vs. Multilabel Classification

**Multiclass**

Each instance can only be assigned to one class out of a finite set of mutually exclusive classes.
- e.g. Species of a flower.
- Accuracy, precision, recall, F1

**Multilabel**

Each instance can be assigned to multiple labels simultaneously, and the labels are not mutually exclusive.
- e.g. Tagging a news article with multiple topics.
- Hamming loss, precision/recall at k (top-k labels)

**SVM**

| Kernel | Use Case / Data Type | Decision Boundary Shape (in original space) | Key Parameters |
| ------ | -------------------- | ------------------------------------------- | -------------- |
| Linear | Linearly separable data, high-dimensional data (like text classification) | Straight line (or flat hyperplane) | C (regularization) |
| Polynomial | Data with a non-linear or polynomial trend; low-dimensional data | Curved or complex (e.g., circular, parabolic) | C, degree (d), coef0 (independent term) |
| RBF (Gaussian) | Default choice when data nature is unknown; complex, non-linear data | Complex, potentially highly flexible and smooth curves | C, gamma (influence of single data points) |
| Sigmoid | Useful in applications related to neural networks (as an activation function) | Highly non-linear, potentially complex and sometimes difficult to interpret | C, gamma, coef0 |

### Reservoir Sampling
Algorithm(s) for randomly selecting a fixed-size sample from a stream of unknown or very large size, where you cannot store all elements in memory.

1. Fill reservoir with the first k elements.
2. For each element x_i (i > k):
    - Generate random integer j in [1, i]
    - If j ≤ k, replace reservoir[j] with x_i
Guarantee: every element has probability $k/n$ of being in the final reservoir.

Intuition: Each new element has a chance to replace an existing one, so that at the end, every element has equal chance to be picked, without knowing the total size in advance.

### Synthetic Oversampling (SMOTE)

To deal with highly imbalanced data (like fraud - minority class) usually leverages an oversampling approaches such as creating synthetic or duplicate samples of the minority class to balance the class distribution, aiming for a 50/50 split for a binary class system for example.

The two primary methods are:
1. Simple Random Oversampling (Duplication)
    - Simply duplicating samples from the minority class to increases their representation in the data
        - Easy to implement but since copying existing data doesn't add new information and leads to overfitting. In fraud each case are unique and rare.
2. Synthetic Minority Oversampling Technique (SMOTE)
    - For every minority data point, find its *k*-nearest neighbors and randomly select one of these neighbors.
    - Create a new **synthetic** sample along the line segment connecting the original fraud case and its selected neighbor. Then introduce random perturbation to the feature values to create a new point. Repeat till balance reached.
        - Creating slightly different but similar enough minority case to reduce the risk of overfitting and making the model more robust. However the if the original minority samples are already noisy and very close to the majority class, SMOTE can generate noisy synthetic samples worsening the decision boundary.

There is another variants of SMOTE - ADASYN (Adaptive Synthetic Sampling) - similar to SMOTE but it focuses on generating more synthetic data for the minority samples that are harder to learn - those close to the majority decision boundary.

Any of the oversampling methods should only be performed on the training data set.

### Cross-validation

Technique used to evaluate model performance on new, unseen data by repeatedly splitting the dataset into training and testing sets. The model is trained on the training portion and validated on the testing portion, and this process is repeated multiple times, with each subset of data getting a chance to be the test set. This helps create a more robust estimate of the model's generalization ability and reduces the risk of overfitting.

- **Divide the data**: The initial dataset is divided into several subsets, or "folds".
- **Train and test**: The model is trained on all but one of these folds and tested on the remaining fold.
- **Repeat**: This process is repeated several times, with a different fold held out for testing each time.
- **Aggregate results**: The performance metrics (e.g., error rates) from each test are averaged to get a final, more reliable performance score. 

Benefits
- **Reduces overfitting**: By testing on different subsets of the data, cross-validation provides a better measure of how the model will perform on unseen data, as opposed to just the one specific test set.
- **More reliable estimate**: Averaging the results from multiple test runs gives a more stable and reliable estimate of performance compared to a single train-test split.
- **Efficient use of data**: For small datasets, it ensures that every data point is used for both training and validation, which is a more efficient use of the data.
- **Model comparison**: It is a powerful tool for comparing the performance of different models on the same task to select the best one. 

Common types
- **K-Fold Cross-Validation**: The most common type, where the data is split into k folds, and the process is repeated k times, with each fold used as the test set once.
- **Leave-One-Out Cross-Validation (LOOCV)**: An extreme case of K-Fold where k is equal to the number of data points. It can be computationally expensive.
- **Shuffle Split Cross-Validation**: Also known as repeated random subsampling, it involves multiple random splits of the data into training and testing sets. 

# Deep Learning

### Activation Functions

| Function       | Formula                                                  | Output Range | Typical Use                                    |
| -------------- | -------------------------------------------------------- | ------------ | ---------------------------------------------- |
| **Sigmoid**    | $ \sigma(z) = \frac{1}{1+e^{-z}} $                       | (0,1)        | Binary classification, probability output      |
| **Tanh**       | $ \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} $         | (-1,1)       | Hidden layers, zero-centered outputs           |
| **ReLU**       | $ \max(0, z) $                                           | [0, ∞)       | Hidden layers, CNNs, faster convergence        |
| **Leaky ReLU** | $ \max(0.01 z, z) $                                      | (-∞, ∞)      | Mitigate dead neurons in ReLU                  |
| **Softmax**    | $ \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}} $ | (0,1) sum=1  | Multi-class classification, probability output |


### Loss Functions

| Function       | Formula                                                  | Output Range | Typical Use                                    |
| -------------- | -------------------------------------------------------- | ------------ | ---------------------------------------------- |
| **Sigmoid**    | $ \sigma(z) = \frac{1}{1+e^{-z}} $                       | (0,1)        | Binary classification, probability output      |
| **Tanh**       | $ \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} $         | (-1,1)       | Hidden layers, zero-centered outputs           |
| **ReLU**       | $ \max(0, z) $                                           | [0, ∞)       | Hidden layers, CNNs, faster convergence        |
| **Leaky ReLU** | $ \max(0.01 z, z) $                                      | (-∞, ∞)      | Mitigate dead neurons in ReLU                  |
| **Softmax**    | $ \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}} $ | (0,1) sum=1  | Multi-class classification, probability output |



# Optimization

### Gradient Descent

| Optimizer | Description | Key Feature |
| --------- | ----------- | ----------- |
| Gradient Descent (GD) | Computes the gradient using the entire training dataset for each parameter update. | High precision, but slow and computationally expensive for large datasets. |
| Stochastic Gradient Descent (SGD) | Computes the gradient using a single randomly chosen data sample for each update. | Fast updates, but the path to the minimum is noisy (high variance), leading to oscillations. |
| Mini-Batch Gradient Descent | The most common practical approach. Computes the gradient using a small batch of data (e.g., 32, 64, 128 samples). | Strikes a balance between GD's stability and SGD's speed.
SGD with Momentum | Adds a "velocity" term that accumulates a fraction of the previous update. | Helps the optimizer accelerate across flat areas and dampen oscillations in steep ravines, leading to faster convergence. |

### Adaptive Learning Rate Methods

| Optimizer | Description | Key Feature |
| --------- | ----------- | ----------- |
| AdaGrad (Adaptive Gradient) | Adapts the learning rate based on the sum of the squared historical gradients. | Great for sparse data (gives larger updates for infrequent parameters/features). Main drawback is an aggressively decaying learning rate. |
| RMSProp (Root Mean Square Propagation) | Solves AdaGrad's aggressive decay by using an exponentially decaying average of squared past gradients. | The learning rate adjusts more slowly, making it more robust for non-convex problems (like deep neural networks). |
| Adam (Adaptive Moment Estimation) | Combines the benefits of Momentum (using the average of past gradients) and RMSProp (using the average of past squared gradients). | Extremely popular due to its fast convergence and minimal need for hyperparameter tuning. It has a bias correction mechanism. |
| AdamW (Adam with Decoupled Weight Decay) | A modification of Adam that decouples the weight decay (L2 regularization) from the gradient updates. | Improves generalization (performs better on unseen data) compared to standard Adam, which often finds sharper minima. This is the preferred version of Adam for most deep learning tasks today. |

**Adam vs. AdamW**

- Adam (Adaptive Moment Estimation): Weight decay (L2 regularization) is added to the gradient: grad = grad + weight_decay * param.

- AdamW (Adam with Decoupled Weight Decay): Weight decay is applied after the gradient update: param = param * (1 - lr * weight_decay).

**Learning Rate Schedules**: Techniques that change the learning rate over time (e.g., reducing it after a set number of epochs or using a cosine annealing schedule) to help the model converge more precisely.

**Second-Order Methods (e.g., Newton's Method)**: These use the second derivative (Hessian matrix) to find a better direction to the minimum. They offer faster convergence but are often computationally prohibitively expensive for deep learning models with millions of parameters.

**Regularization (L1, L2)**: Techniques like Weight Decay (which is L2 regularization) are often used alongside optimizers (as seen in AdamW) to penalize large weights and prevent overfitting.

### Large Language Model

**Perplexity (PPL)**
A metric of how "surprised" or uncertain a model is by a sequence of text; essentially, how many choices it effectively has at each step.
- Lower perplexity indicates the model is more confident and accurate in its predictions, finding text more probable.
- Primarily an evaluation metric to assess model performance, though it's less reliable than human judgment and can be vocabulary-dependent. 

**Temperature (T)**
A hyperparameter that scales the output probabilities, affecting the randomness of token selection.
- High Temperature (e.g., T > 1.0): Makes less likely tokens more probable, leading to diverse, creative, but potentially nonsensical outputs (higher uncertainty/perplexity).
- Low Temperature (e.g., T < 0.7): Sharpens probabilities, favoring the most likely tokens, resulting in focused, predictable, but potentially dull or repetitive text (lower uncertainty/perplexity).
- A direct control knob for generation style (e.g., brainstorming vs. factual answers). 

# Reinforcement Learning

| Concepts | Q-table | Deep Q-Network (DQN) | Actor-Critic (A2C) | Proximal Policy Optimization (PPO) | Generalized Policy Optimization (GRPO) |
| ------- | ------- | -------------------- | ------------------ | ---------------------------------- | -------------------------------------- |
| Approach Type | Explicitly maps every state and every action to a numerical value in a table. | Uses a deep neural network (DNN) as a function approximator for the Q-table. | The Actor learns a policy (which action to take), and the Critic learns a value function (how good the state is). It uses the "advantage" (how much better an action was than average) to update the policy. | PPO improves upon earlier policy gradient methods by using a "clipping" mechanism to restrict how much the new policy can change from the old policy during each update. | GRPO generally refers to generalized frameworks for policy optimization, often related to older theoretical methods or specific academic implementations that generalize concepts found in algorithms like Trust Region Policy Optimization (TRPO).|
| State Space | Small/Discrete | Large/Continuous | Large/Continuous | Large/Continuous | Large/Continuous |
| Action Space | Discrete | Discrete | Discrete/Continuous | Discrete/Continuous | 	Discrete/Continuous |
| Scalability | Poor | Good	Excellent | Excellent | Good |
| Complexity | Low | Medium | High | High | Very High |
| Stability/Robustness | High | Medium | Medium-High | Very High |

**Policy**
In Reinforcement Learning, a policy ($\pi$) is the agent's strategy or rule set for choosing an action. It is a mapping from observed states to actions.

- Policy Notation: It is often written as $\pi(a|s)$, which is the probability of taking action $a$ when in state $s$.
- Optimal Policy ($\pi^{\*}$): The goal of nearly all RL algorithms is to find the optimal policy, $\pi^{\*}$, which maximizes the expected cumulative discounted future reward.

Policies come in two main types:
1. Deterministic Policy: $\pi(s) = a$. For a given state, the agent always chooses the same action (e.g., Q-Learning's evaluation policy).
2. Stochastic Policy: $\pi(a|s)$. For a given state, the agent chooses actions based on a probability distribution (e.g., the $\epsilon$-greedy policy used for exploration).

**On-Policy vs. Off-Policy**

1. **On-Policy Learning** (e.g., SARSA) algorithms learn the value of the policy they are currently using to act.
    - **Behavior Policy ($\pi$)**: The policy used to select actions and interact with the environment (e.g., $\epsilon$-greedy).Evaluation Policy: The policy being evaluated and improved is the same policy ($\pi$).
    - **Key Idea**: The agent learns the value of taking an action, including the risks and returns associated with the occasional random, exploratory steps. The learned Q-values reflect the returns expected under the $\epsilon$-greedy policy itself.
    - **Result**: The learned policy is often more conservative because it accounts for the negative consequences of exploring.

2. **Off-Policy Learning** (e.g., SARSAmax a.k.a. Q-Learning) algorithms learn the value of one policy (the target policy) while following a different policy (the behavior policy).
    - **Behavior Policy ($\pi$)**: The policy used to gather data and explore (e.g., $\epsilon$-greedy).Evaluation Policy ($\mu$): The policy being evaluated and improved is the greedy (optimal) policy $\mu$, which selects the $\arg\max$ action.
    - **Key Idea**: The agent uses the experience gained from its exploratory actions ($\pi$) to estimate what the returns would have been if it had followed the greedy policy ($\mu$).
    - **Result**: The learned policy is the optimal greedy policy ($\pi^*$). This approach allows the agent to learn the best path faster, independent of the random steps taken for exploration, but it may lead to a riskier optimal path if exploration involves massive penalties (like falling off a cliff).

| Feature | On-Policy (e.g., SARSA) | Off-Policy (e.g., Q-Learning) |
| ------- | ----------------------- | ----------------------------- |
| Learning Policy | $\pi$ (The policy the agent follows) | $\mu$ (The optimal/greedy policy) |
| Data Policy | $\pi$ | $\pi$ |
| Update Target | $\mathbf{Q(S_{t+1}, A_{t+1})}$ where $A_{t+1} \sim \pi$ | $\mathbf{\max_{a} Q(S_{t+1}, a)}$ |
| Nature | Conservative | Optimal/Aggressive |


**Temporal Difference methods**

| Algorithm | SARSA | Q-Learning (SARSAmax) | Expected SARSA |
| --------- | ----- | --------------------- | -------------- |
| Policy Type | On-Policy | Off-Policy | Off-Policy (Hybrid) |
| Policy Learned | The value of the Exploratory Policy ($\epsilon$-greedy). | The value of the Optimal Greedy Policy ($\mu$). | The value of the Optimal Greedy Policy ($\mu$). |
| Next Action Used | The actual action $A_{t+1}$ chosen by the $\epsilon$-greedy policy $\pi$. | The greedy action $A_{\text{max}}$ (the one with the highest Q-value). | The expected value over all possible next actions $a'$, weighted by their probability $\pi(a')$ |
| TD Target | $R_{t+1} + \gamma \mathbf{Q(S_{t+1}, A_{t+1})}$ | $R_{t+1} + \gamma \mathbf{\max_{a} Q(S_{t+1}, a)}$ | $R_{t+1} + \gamma \mathbf{\sum_{a'} \pi(a')}$ |
| Convergence | Converges only to $Q^*$ if the policy $\pi$ decays $\epsilon \to 0$. | Converges directly to $Q^*$ regardless of the behavior policy $\pi$. | Converges directly to $Q^*$ regardless of the behavior policy $\pi$. |
| Safety/Risk | Conservative: Learns safer paths, accounting for exploration risk. |Aggressive: Learns the mathematically optimal path, ignoring exploration risk. |Balanced: More stable than Q-Learning and learns the optimal path. |

**Policy Relationship**
| Algorithm | Relationship | Interpretation |
| --------- | ------------ | -------------- |
| SARSA | On-Policy (Policy $\pi$ learns about Policy $\pi$) | Learns the expected return from taking action $A_t$ and continuing to follow the same exploratory strategy $\pi$. |
| Q-Learning | Off-Policy (Policy $\pi$ learns about Policy $\mu$) | Learns the expected return from taking action $A_t$ but assumes that after this step, the agent will always act greedily ($\mu$). |
| Expected SARSA | Off-Policy (Policy $\pi$ learns about Policy $\mu$ using $\pi$) | Learns the expected return from taking action $A_t$ by averaging the values of all possible next actions, weighted by their probability of being chosen by the behavior policy $\pi$. This removes the stochasticity introduced by sampling $A_{t+1}$ in SARSA. |


# SQL

Useful references:
- w3school
    - [SQL](https://www.w3schools.com/sql/)
    - [PostgreSQL](https://www.w3schools.com/postgresql/)
- Snowflake
    - [SQL](https://docs.snowflake.com/en/sql-reference/constructs)
    - [Cortex-AISQL](https://docs.snowflake.com/en/user-guide/snowflake-cortex/aisql)
- BigQuery
    - [GoogleSQL](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax)


### Analytical & Window Functions

| Concept                       | General Syntax (T-SQL)                         | Snowflake Syntax | Purpose                                                                                                               |
| ----------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Common Table Expression (CTE) | WITH cte_name AS (SELECT ...)                                                                       | Same                             | Defines a temporary, named result set to simplify complex, multi-step queries.                                                                            |
| Window Functions              | Function() OVER (PARTITION BY col ORDER BY col [frame]);                                            | Same                             | Calculates an aggregate value or ranking over a set of rows while retaining individual row detail.                                                        |
| Window Framing (Fixed)        | SUM(value) OVER (PARTITION BY group ORDER BY time ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)         | Same                             | Defines a fixed rolling window (e.g. a 7-day rolling sum), specifying the exact rows to include relative to the current row.                             |
| Window Framing (Cumulative)   | SUM(value) OVER (PARTITION BY group ORDER BY time ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) | Same                             | Defines an expanding cumulative window, calculating the total from the start of the partition up to the current row (default behavior for SUM and COUNT). |
| LAG / LEAD                    | LAG(col, offset) OVER (PARTITION BY group ORDER BY time);                                           | Same                             | Accesses column values from the previous (LAG) or next (LEAD) row in a sequence (e.g. finding period-over-period change).                                |
| Ranking                       | ROW_NUMBER(), RANK(), DENSE_RANK()                                                                  | Same                             | Assigns an ordered rank or sequence number. Crucial for "Top N" or filtering the latest record.                                                           |
| Percentile Rank                | PERCENT_RANK() OVER (ORDER BY col)                                                                  | Same                             | Calculates the relative rank of a row within a group as a percentage (ranging from 0 to 1).                                                           |
| Conditional Logic             | CASE WHEN condition1 THEN result1 ELSE final_result END;                                            | Same                             | Creates derived columns based on conditional expressions (essential for bucketing/flagging data).                                                         |
| NULL Handling                 | COALESCE(col1, col2, 'Default Value');                                                              | Same                             | Returns the first non-null expression in the list.                                                                                                        |
| Grouping                      | GROUP BY col1, col2                                                                                 | Same                             | Aggregates data based on one or more columns.                                                                                                             |
| Filtering Aggregates          | HAVING COUNT(\*) > 10                                                                               | Same                             | Filters the results after aggregation (i.e., filters groups).                                                                                             |                                                        |

### Data Manipulation & Transformation

**Text specifics manipulations:**
| Concept                       | General Syntax (T-SQL)                         | Snowflake Syntax | Purpose                                                                                                                                          |
| -------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| LEFT             | LEFT(string, length)                                           | Same                                       | Extracts a specified number of characters from the start (left side) of a string.                                                  |
| RIGHT            | RIGHT(string, length)                                          | Same                                       | Extracts a specified number of characters from the end (right side) of a string.                                                   |
| SUBSTRING / MID  | SUBSTRING(string, start, length) or MID(string, start, length) | SUBSTRING(string, start, length)           | Extracts a substring of a specified length starting at a specified position.                                                       |
| LENGTH / LEN     | LENGTH(string) (PostgreSQL) LEN(string) (T-SQL)                | LENGTH(string)                             | Returns the number of characters in a string.                                                                                      |
| POSITION / INSTR | POSITION(substring IN string) (PostgreSQL)                     | POSITION(substring, string)                | Returns the starting position of the first occurrence of a substring within a string. Used with SUBSTRING for complex parsing.     |
| TRIM             | TRIM(string)                                                   | Same                                       | Removes leading and trailing whitespace.                                                                                           |
| REPLACE          | REPLACE(string, old_string, new_string)                        | Same                                       | Replaces all occurrences of a specified substring with another string.                                                             |
| SPLIT_PART       | Varies by platform, often complex SUBSTRING + POSITION logic.  | SPLIT_PART(string, delimiter, part_number) | Highly useful Snowflake function that splits a string by a delimiter and returns the Nth part. Simplifies tokenization.            |
| ILIKE / LIKE     | LIKE is standard. ILIKE is common in PostgreSQL and Snowflake. | ILIKE                                      | Case-insensitive (ILIKE) or case-sensitive (LIKE) pattern matching using wildcards (% for any string, _ for any single character). |

**Date and Time specifics manipulations:**
| Concept                       | General Syntax (T-SQL)                         | Snowflake Syntax | Purpose                                                                                                                                          |
| -------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| DATE_TRUNC        | DATE_TRUNC('month', date_col)            | Same                                     | Truncates a timestamp/date to the start of a specified interval (e.g. year, month, week). Essential for aggregation. |
| DATEDIFF          | DATEDIFF(interval, start_date, end_date) | DATEDIFF(part, date1, date2)             | Returns the difference between two dates/timestamps in the specified time part (e.g. 'day', 'hour').                 |
| DATEADD           | DATEADD(interval, number, date)          | DATEADD(part, value, date)               | Adds a specified number of time units to a date/timestamp. Used to create rolling windows or future projections.      |
| Date Parts        | MONTH(date), YEAR(date), DAYOFWEEK(date) | MONTH(date), YEAR(date), DAYOFWEEK(date) | Extracts a specific part of a date/timestamp. Snowflake also offers DAYOFWEEK, DAYOFMONTH, WEEKOFYEAR, etc.           |
| Current Date/Time | GETDATE() (T-SQL) NOW() (PostgreSQL)     | CURRENT_DATE(), CURRENT_TIMESTAMP()      | Returns the current system date or timestamp.                                                                         |

**Other datatypes:**

| Concept                       | General Syntax (T-SQL)                         | Snowflake Syntax | Purpose                                                                                                                                          |
| -------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| JSON/Semi-Structured Data Access | JSON_VALUE(col, '$.field') (T-SQL) or col->'field' (PostgreSQL) | col:field_name::type or GET(col, 'field_name')                              | Accesses fields within VARIANT, ARRAY, or OBJECT columns using native, simple syntax.                                                            |
| Flattening Arrays/JSON           | Requires complex lateral joins or custom functions.             | SELECT t.\*, f.value FROM table t, LATERAL FLATTEN(INPUT => t.array_col) f; | FLATTEN is a powerful table function that converts elements within a semi-structured array or object into separate rows, allowing easy analysis. |
| Parsing JSON                     | Varies by platform.                                             | PARSE_JSON('{"key": "value"}')                                              | Converts a string representation of JSON text into a storable VARIANT data type.                                                                 |
| Geospatial Distance              | Varies (e.g. ST_Distance in PostGIS).                          | ST_DISTANCE(point1, point2) (requires GEOGRAPHY data type)                  | Calculates the distance between two geospatial points on the Earth's surface.                                                                    |

### Advanced Data Manipulation & Transformation

| Concept                       | General Syntax (T-SQL)                         | Snowflake Syntax | Purpose                                                                                                                                          |
| -------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| PIVOT (Row to Column)      | Often requires complex CASE statements with GROUP BY.       | SELECT \* FROM table PIVOT(SUM(val) FOR col_to_pivot IN ('A', 'B')); | Converts unique row values from one column into new columns (long to wide format).                                       |
| UNPIVOT (Column to Row)    | Often requires many UNION ALL statements.                   | UNPIVOT(value_col FOR name_col IN (col1, col2, ...))                 | Converts columns (wide format) into rows (long format) for easier comparison or model input.                             |
| JSON Access (Dot Notation) | Varies by platform (\->, ::json, JSON_VALUE).               | col:field_name::type                                                 | Accesses fields within VARIANT, ARRAY, or OBJECT columns using simple dot notation.                                      |
| JSON Access (GET Function) | Varies by platform.                                         | GET(variant_col, 'key_name')                                         | Accesses an element in a semi-structured type (Variant, Object, or Array) by name or index.                              |
| Upsert (Atomic DML)        | Varies greatly (ON CONFLICT in PostgreSQL, MERGE in T-SQL). | MERGE INTO target USING source ON join_condition ...                 | Performs an atomic update, insert, or delete based on matching keys.                                                     |
| Type Casting               | CAST(col AS DECIMAL(10, 2)) or col::DECIMAL(10, 2)          | col::DECIMAL(10, 2) or CAST(col AS DECIMAL(10, 2))                   | Explicitly converts data from one type to another.                                                                       |
| Manual Binning (Fixed Width/Range) | CASE WHEN price < 100 THEN 'Low' WHEN price < 500 THEN 'Medium' ELSE 'High' END AS price_category | Same (Uses the fundamental CASE expression)                          | Divides data into custom, fixed-range categories based on business rules or expert judgment (e.g. age groups, income brackets).                                     |
| Quantile Binning (Equal Count)     | NTILE(4) OVER (ORDER BY numeric_col) AS quartile                                                  | NTILE(N) OVER (ORDER BY numeric_col)                                 | Divides data into *n* bins containing roughly equal numbers of rows (e.g. quartiles, deciles). This method is used to manage outliers and create relative rankings. |
| Percentile Value     | PERCENTILE_CONT(P) WITHIN GROUP (ORDER BY col)                                                  | Same                                 | Calculates the percentile value (P) of a column. CONT interpolates for non-existent values, DISC returns an actual value from the column. |
| Numeric Binning      | ROUND(scores / N) * N | Same | Rounding method of numerical value binning, in general N is the size of the fixed-width bins which will bin numerical values to the *Nearest* (FLOOR - downward, CEIL - upward) bin. e.g. ROUND(13.2 / 5) * 5 = 13, ROUND(13.2 / 10) * 5 = 10 and CEIL(13.2 / 5) * 5 = 15|

### Data Definition & Context

| Concept                       | General Syntax (T-SQL)                         | Snowflake Syntax | purpose                                                                                                                                              |
| ----------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Context Switching             | USE database_name; (T-SQL)                           | USE WAREHOUSE compute_wh; USE DATABASE data_db; USE SCHEMA reporting_schema; | Sets the current active Virtual Warehouse (compute resource) and the database/schema context. Crucial for Snowflake performance and cost management. |
| Creating a Table              | CREATE TABLE table_name (col1 INT, col2 VARCHAR);    | Same                                                                         | Creates a new table structure.                                          |
| Creating a Table from a Query | CREATE TABLE table_name AS SELECT ...;               | CREATE TABLE table_name AS SELECT ...; or CREATE OR REPLACE TABLE ...        | Creates and populates a new table structure based on the results of a query (often used for permanent staging/intermediate tables).                  |
| Delete with conditions  | DELETE FROM table_name WHERE condition;                           | Same                                                                         | Removes rows from a table based on a specified condition. If no WHERE clause is provided, all rows are deleted.                           |
| Removing All Data             | TRUNCATE TABLE table_name;                           | Same                                                                         | Removes all rows from a table quickly and efficiently without affecting the table structure.                                                         |
| MERGE INTO (Upsert)        | MERGE INTO target t USING source s ON t.id = s.id WHEN MATCHED THEN UPDATE SET ... WHEN NOT MATCHED THEN INSERT ... / WHEN NOT MATCHED BY source THEN DELETE; | Same                                                                         | Performs an atomic "Upsert" (Update or Insert) operation. Crucial for synchronizing a target table with a staging table in a single, efficient transaction.                                                      |


# Python


### General

| Feature | List | Set | Tuple | Dictionary |
| ------- | ---- | --- | ----- | ---------- |
| Mutability | Mutable | Mutable (elements must be immutable) | Immutable| Mutable |
| Hashable | No | No | Yes | No |
| Ordering | Ordered | Unordered | Ordered | Ordered |
| Duplicates | Allows duplicates | No duplicates allowed | Allows duplicates | No duplicate keys |
| Indexing | Supports indexing and slicing | Not supported | Supports indexing and slicing | By Key |
| Performance | Slower for membership tests | Faster Membership tests | Faster than lists | Fast lookup and modification |
| Use Case | When frequent modifications are required | When uniqueness is needed | When immutability is required| When association or mapping between values |


### Algorithms








### Snippets

**Using** ```__main__``` **Safely** - Ensures script only runs when executed directly, not when imported.
```python
def main():
    print("Running script...")

if __name__ == "__main__":
    main()
```

**Context Manager for Safe File Handling** - Automatically handles closing files (no resource leaks).
```python
with open("data.txt", "r") as f:
    text = f.read()
```

**Using enumerate()** - Cleaner than manually indexing lists.
```python
for i, value in enumerate(["a", "b", "c"], start=1):
    print(i, value)
```

**List Comprehensions** - Pythonic, fast, and readable.
```python
squares = [x**2 for x in range(10)]
```

**deque**
```python
from collections import deque

# Create a deque
my_deque = deque([1, 2, 3])

# Append to the left
my_deque.appendleft(0)
print(f"Deque after appendleft(0): {my_deque}")
```
> Deque after appendleft(0): deque([0, 1, 2, 3])
``` python
# Append to the right
my_deque.append(4)
print(f"Deque after append(4): {my_deque}")
```
> Deque after append(4): deque([0, 1, 2, 3, 4])

**Counter**
```python
from collections import Counter

# Initialize a Counter from a string
c = Counter("mississippi")
print(c)
```
> Counter({'i': 4, 's': 4, 'p': 2, 'm': 1})
```python
# Updating counts (+)
c.update("pennsylvania")
print(c)
```
> Counter({'i': 5, 's': 5, 'p': 3, 'n': 3, 'a': 2, 'm': 1, 'e': 1, 'y': 1, 'l': 1, 'v': 1})

```python
# Arithmetic operations (-)
c2 = Counter("apple")
result = c - c2
print(result)
```
> Counter({'i': 5, 's': 5, 'n': 3, 'm': 1, 'p': 1, 'y': 1, 'v': 1, 'a': 1})

# Most common elements
print(c.most_common(3)) # [('i', 5), ('s', 5), ('p', 3)]
```

**Dictionary Comprehensions** - Quick way to build dictionaries.
```python
lookup = {x: x**2 for x in range(5)}
```

**Using pathlib Instead of** ```os.path``` - More modern, readable file path handling.
```python
from pathlib import Path

data_dir = Path("data")
print(list(data_dir.glob("*.csv")))
```

### File Read / Write & Data Engineering
**Reading Large CSV in Chunks** - Processes big data without memory issues.
```python
import pandas as pd

for chunk in pd.read_csv("large.csv", chunksize=50_000):
    print(len(chunk))
```

**Writing Clean CSV** - Prevents index column from polluting output files.
```python
df.to_csv("output.csv", index=False)
```

**Read & Write Parquet** - Fast columnar format for analytics pipelines.
```python
import pandas as pd

df = pd.read_parquet("data.parquet")
df.to_parquet("output.parquet")
```

**Efficient Logging** - Better than using print() in production.
```python
import logging

logging.basicConfig(level=logging.INFO)
logging.info("Pipeline started.")
```

### Data Manipulation (Pandas)
**Filter Rows**
```python
filtered = df[df["country"] == "Canada"]
```

**Select Columns**
```python
subset = df[["user_id", "sales"]]
```

**Create New Columns**
```python
df["revenue"] = df["price"] * df["quantity"]
```

**groupby Aggregation**
```python
summary = df.groupby("region")["sales"].sum().reset_index()
```

**Multi-Aggregation**
```python
agg = (
    df.groupby("region")
      .agg({"sales": ["mean", "sum"], "orders": "count"})
      .reset_index()
)
```

**Handling Missing Data**
```python
df = df.fillna({"sales": 0})
# or
df = df.dropna()
```

**Vectorized String Operations**
```python
df["email_domain"] = df["email"].str.split("@").str[-1]
```

**Joining / Merging**
```python
merged = df1.merge(df2, on="user_id", how="left")
```

### ETL Patterns
**Creating a Reusable ETL Step** - Functional, chainable, and clean.
```python
def clean_sales(df):
    return (
        df.dropna(subset=["user_id"])
          .assign(revenue=lambda x: x["qty"] * x["price"])
    )
```

**Pipeline with** ```__call__()``` - Helps compose pipelines like scikit-learn transformers.
```python
class PipelineStep:
    def __call__(self, df):
        df = df.copy()
        df["flag"] = df["value"] > 10
        return df

step = PipelineStep()
df = step(df)
```

### Visualization (Matplotlib)
**Basic Line Plot**
```python
import matplotlib.pyplot as plt

plt.plot(df["date"], df["sales"])
plt.title("Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()
```

**Bar Chart**
```python
df.groupby("region")["sales"].sum().plot(kind="bar")
plt.show()
```

### Machien Learning (scikit-learn)
**Train/Test Split**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Standard ML Workflow**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

model = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression())
])

model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

**Cross-Validation**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(scores.mean())
```

**Hyperparameter Tuning (Grid Search)**
```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    model,
    param_grid={"clf__C": [0.01, 0.1, 1, 10]},
    cv=5,
)
grid.fit(X, y)
print(grid.best_params_)
```

### Large Objects
**Using Dask for Out-of-Core Data**
```python
import dask.dataframe as dd

df = dd.read_csv("bigdata/*.csv")
df.groupby("region")["sales"].mean().compute()
```

**Pickle Model Save/Load**
```python
import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Loading back
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
```

**Joblib for Large Models**
```python
from joblib import dump, load

dump(model, "model.joblib")
model = load("model.joblib")
```

### Unit Testing & Code Quality
**Simple Test with pytest**
```python
def test_sum():
    assert 1 + 1 == 2
```

**Adding Type Hints**
```python
def add(a: int, b: int) -> int:
    return a + b
```

**Using dataclass** - Less boilerplate for small classes.
```python
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str

u = User(1, "Alice")
```

### Image Processing

**Load Image (cv2)**
```python
import cv2

img = cv2.imread("image.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

**Draw Rectangle (cv2)**
```python
cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 2)
```

**Resize Image (cv2)**
```python
resized = cv2.resize(img, (256, 256))
```

**Convert cv2 Image to PIL**
```python
from PIL import Image

pil_img = Image.fromarray(img_rgb)
```

**Convert PIL to OpenCV (numpy)**
```python
import numpy as np

opencv_img = np.array(pil_img)
opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
```

**Create Blank Image with Drawing (PIL)**
```python
from PIL import Image, ImageDraw

img = Image.new("RGB", (400, 400), "white")
draw = ImageDraw.Draw(img)
draw.rectangle((50, 50, 200, 200), outline="red", width=3)
img.show()
```

**PyTorch (torchvision)**
```python
import torchvision.transforms as T

transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomResizedCrop(224),
    T.ToTensor()
])
```

**TensorFlow**
```python
data = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    horizontal_flip=True,
    zoom_range=0.1
)
```

### PyTorch

**Basic Neural Network**
```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)
```

**Initialize Model, Loss, Optimizer**
```python
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

**Standard Training Loop**
```python
for epoch in range(10):
    for X, y in dataloader:
        preds = model(X)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} Loss: {loss.item():.4f}")
```

**Evaluate Mode (No Gradient)**
```python
model.eval()
with torch.no_grad():
    preds = model(X_test)
```

**Save / Load**
```python
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))
```

## TensorFlow / Keras Reference
**Basic Sequential Model**
```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])
```

**Compile**
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

**Train**
```python
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

**Evaluate**
```python
model.evaluate(X_test, y_test)
```

**Predict**
```python
preds = model.predict(X_test)
```

**Save / Load**
```python
model.save("model.keras")
model = tf.keras.models.load_model("model.keras")
```

**Early Stopping**
```python
from tensorflow.keras.callbacks import EarlyStopping

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
]

model.fit(X_train, y_train, validation_split=0.1, callbacks=callbacks)
```

**Learning Rate Schedulers**
```python
# PyTorch
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(E):
    train(...)
    scheduler.step()

#TensorFlow
callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2
)
```

### HuggingFace
**Sentiment Analysis**
```python
from transformers import pipeline

clf = pipeline("sentiment-analysis")
clf("I love Hugging Face!")
```

**Translation**
```python
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
translator("This is amazing!")
```

**Text Generation**
```python
gen = pipeline("text-generation", model="gpt2")
gen("Deep learning is")
```

**Load a Dataset from Hub**

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
train = dataset["train"]
test  = dataset["test"]
```

**Tokenization**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

tokens = tokenizer(
    "Hugging Face is great!",
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)
```

Text Classification (Train with Trainer API) - Load Model + Tokenizer

```python
from transformers import AutoModelForSequenceClassification

model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

**Dataset Tokenization Function**
```python
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)
```

**Training Setup**
```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

trainer.train()
```

**Using the Model for Inference**
```python
inputs = tokenizer("I really enjoyed this movie!", return_tensors="pt")
outputs = model(**inputs)
pred = outputs.logits.argmax(dim=-1)
```

**Save & Load Models**
```python
# Save
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

# Load
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")
```

**Get Embeddings (e.g., for semantic search)**
```python
from transformers import AutoModel

model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

text = ["Hugging Face embeddings are awesome."]
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    embeddings = model(**inputs).last_hidden_state.mean(dim=1)
```


**Zero-shot Image Classification**
```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("cat.jpg")
labels = ["cat", "dog", "car"]

inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

scores = outputs.logits_per_image.softmax(dim=1)
```

**Multimodal Generation (LLaVA, etc.)**
```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = AutoModelForVision2Seq.from_pretrained("llava-hf/llava-1.5-7b-hf")

img = Image.open("image.png")
prompt = "Describe this image."

inputs = processor(prompt, img, return_tensors="pt")
result = model.generate(**inputs, max_new_tokens=100)

print(processor.decode(result[0], skip_special_tokens=True))
```

**Optimize Inference (Accelerate / GPU)**
```python
from accelerate import init_empty_weights

model = AutoModel.from_pretrained(
    "distilbert-base-uncased",
    device_map="auto"
)
```

**DeepseekVL**
```python
import torch
from transformers import pipeline

pipe = pipeline(
    task="image-text-to-text",
    model="deepseek-community/deepseek-vl-1.3b-chat",
    device=0,
    dtype=torch.float16
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            { "type": "text", "text": "Describe this image."},
        ]
    }
]

pipe(text=messages, max_new_tokens=20, return_full_text=False)
```

### Advanced Neural Network

**Load Model + Tokenizer**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```
**Transfer Learning – Replace Classification Head**
```python
# Often done when adapting to a new number of labels.

import torch.nn as nn

num_new_labels = 5

model.classifier = nn.Linear(model.config.dim, num_new_labels)
model.config.num_labels = num_new_labels

#For BERT-style architectures:
model.classifier = nn.Linear(model.config.hidden_size, num_new_labels)
```

**Freeze All Base Layers (Feature Extraction)**
Useful when dataset is small.
```python
for param in model.base_model.parameters():
    param.requires_grad = False
# Now only the new classifier head trains.
```

**Freeze Bottom N Layers (Progressive Unfreezing)**
```python
n_freeze = 4

for name, param in model.named_parameters():
    if any(f"layer.{i}" in name for i in range(n_freeze)):
        param.requires_grad = False
```

**Unfreeze Later (e.g., after warm-up)**
```python
for param in model.parameters():
    param.requires_grad = True
```

**PyTorch Training Loop (Manual)**
```python
import torch
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)
model.train()

for epoch in range(3):
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
```

**Fine-Tuning Using Trainer**
```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds
)

trainer.train()
```

**Knowledge Distillation (Student learns from Teacher)**
```python
#Teacher Model (pretrained)
teacher = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)
teacher.eval()

#Student Model (smaller)
student = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

**Distillation Loss (Soft Targets)**
```python
import torch.nn.functional as F

temperature = 3.0
alpha = 0.5   # Learning from teacher vs real labels

def distillation_loss(student_logits, teacher_logits, labels):
    hard_loss = F.cross_entropy(student_logits, labels)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean"
    )
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

**Distillation Training Step**
```python
student.train()
teacher.eval()

for batch in train_loader:
    outputs_teacher = teacher(**batch).logits
    outputs_student = student(**batch).logits

    loss = distillation_loss(outputs_student, outputs_teacher, batch["labels"])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Distillation with Hugging Face Trainer**
```python
class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        
        with torch.no_grad():
            teacher_logits = teacher(**inputs).logits
        
        outputs_student = model(**inputs)
        student_logits = outputs_student.logits
        
        loss = distillation_loss(student_logits, teacher_logits, labels)
        
        return (loss, outputs_student) if return_outputs else loss

#Run:
distill_trainer = DistillationTrainer(
    model=student,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

distill_trainer.train()
```

**Freeze Embeddings Only (Common Technique)**
Helps stabilize low-level features.
```python
for param in model.base_model.embeddings.parameters():
    param.requires_grad = False
```

**Check Which Params Are Trainable**
```python
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

**Gradient Checkpointing (Save Memory)**
```python
model.gradient_checkpointing_enable()
```

**Mixed Precision Training (FP16)**
```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()

    with autocast():
        loss = model(**batch).loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

#Using Trainer:
args = TrainingArguments(
    output_dir="./results",
    fp16=True
)
```

**Learning-Rate Scheduling**
```python
from transformers import get_linear_schedule_with_warmup

num_train_steps = len(train_loader) * 3
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=num_train_steps
)

for batch in train_loader:
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

**Save + Load Weights**
```python
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

model = AutoModelForSequenceClassification.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")
```

**Use Model for Embeddings (Mean Pooling)**
```python
from torch.nn.functional import normalize

inputs = tokenizer("Hello world", return_tensors="pt")
with torch.no_grad():
    last_hidden = model.base_model(**inputs).last_hidden_state

emb = last_hidden.mean(dim=1)
emb = normalize(emb, p=2, dim=1)
```

**Visual Transformers (Example: ViT Fine-Tuning)**
```python
from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=10
)

#Freeze backbone:
for param in model.vit.parameters():
    param.requires_grad = False
```

**LoRA (Parameter-Efficient Fine-Tuning)**
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1
)

lora_model = get_peft_model(model, config)
```












