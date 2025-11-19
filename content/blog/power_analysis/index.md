---
title: "Power Analysis"
date: 2022-03-19
draft: false
tags: [Statistics, Hypothesis Testing, Data Science, Data Analytics, Web Analytics, Python, R]
categories: []
showReadingTime: true
showTableOfContents: True
summary: "Process of Power Analysis"
---

<center>
<img src="thumb.jpg" width="50%">
</center>

## Introduction

This blog to showcase an example of conducting a power analysis for determine the optimal sample size needed to detect a meaningful effect.

## Power

**Power** is the probability of detecting an effect (i.e. rejecting the null hypothesis) given that some prespecified effect actually exists using a given test in a given context. The **power of the test** is the probability that the test correctly rejects the null hypothesis (H<sub>0</sub>) when the alternative hypothesis (H<sub>a</sub>) is true. It is commonly denoted by 1 - β, where β is the probability of making a Type II error (Failing to reject H<sub>0</sub> when alternative H<sub>a</sub> is true).

The power of a test (1 - β) is highly dependent on the **effect size** and the constraints on the **sample size (*n*)**.

## Example

A marketing team wants to test a new checkout button color (Variant B) against the current color (Control A) on their website. They want to know how many unique visitors they need in each group to be confident they can detect a significant improvement.

To conduct a power analysis, first **specify four parameters** and solve for the fifth. The formula used here is appropriate for comparing two proportions (like conversion rates).

| Parameter                       | Symbol       | Standard Target | Role                                                  |
| ------------------------------- | ------------ | --------------- | ----------------------------------------------------- |
| Significance Level              | α     | 0.05 (5%)       | Maximum risk of a Type I Error (False Positive).      |
| Desired Power                   | 1 - β  | 0.80 (80%)      | The probability of correctly detecting a real effect. |
| Baseline Rate                   | p<sub>A</sub>        | \-              | The current performance of the control group.         |
| Minimum Detectable Effect (MDE) | δ     | \-              | The smallest effect size we care about finding.       |
| Required Sample Size            | *n*          | The Result      | The number of participants needed per group.          |


**Step 1: Define the Input Parameters**
1. Set α (Significance Level)
- Choice: α = 0.05.
  - *Interpretation*: We are willing to accept a 5% chance of falsely concluding the new button is better when it is actually no different (Type I Error - mistakenly rejecting H<sub>0</sub>).

2. Set Power (1 - β)
- Choice: *Power = 0.80*.
  - *Interpretation*: If the new button truly performs better by the desired MDE (see below), we want an 80% chance of successfully detecting that difference (correctly rejecting the false H<sub>0</sub>). This means β (Type II Error risk) is 1 - 0.80 = 0.20 (20%).

3. Estimate Baseline Rate (p<sub>A</sub>)
- Data: Based on historical data, the current checkout conversion rate (Control A) is 10% (p<sub>A</sub> = 0.10).

4. Define the Minimum Detectable Effect (MDE, δ)
- This is the most crucial, non-statistical input. It answers the business question: "What is the smallest lift that is **worth the effort and cost** of changing the website?"
- Choice: The team decides that a 10% relative increase in conversion is the minimum lift worth the development effort.
  - Calculation: Absolute MDE = 10% of 10% = 0.01 (or 1 percentage point).
Target Conversion Rate (p<sub>B</sub>) = p<sub>A</sub> + Absolute MDE = 0.10 + 0.01 = 0.11 (11%).
  - δ (MDE) = p<sub>B</sub> - p<sub>A</sub> = 0.01.

**Step 2: Calculate the Required Sample Size (*n*)**

Using the inputs (α = 0.05, Power 1 - β = 0.80, p<sub>A</sub>=0.10, p<sub>B</sub>=0.11), apply the appropriate sample size formula for comparing two proportions (using z-scores derived from the standard normal distribution).

$$n \ge \frac{(Z_{\alpha/2} + Z_{\beta})^2 \cdot (p_A(1-p_A) + p_B(1-p_B))}{(p_A - p_B)^2}$$


**Finite Population Correction**

The standard statistical formulas above assume that:
  - The population is **infinite** OR
  - Sampling is done with **replacement**

In both cases, selecting one observation doesn't change the probability of selecting the next ensuring **independence**.

However, this is not the case in reality when working (i.e. sampling without replacement) with a finite and especially when the population is small. The probability of selecting subsequent observations **changes** because the remaining population size decreases drastically. This non-independence means the standard formulas **overestimate** the **standard error**.

A **Finite Population Correction (FPC)** is introduced to account for the adjustment.

$$  \text{FPC} = \frac{N - n}{N - 1}$$

We apply the FPC on both population then the new formula with the FPC factor:

$$n \ge \frac{(Z_{\alpha/2} + Z_{\beta})^2 \cdot (fpc_1 \cdot \bar{p_A}(1-\bar{p_A}) + fpc_1 \cdot p_B(1-p_B))}{(p_A - p_B)^2}$$

Where
$$ fpc_1 = \frac{N_1 - n}{n-1},  fpc_2 = \frac{N_2 - n}{n-1}$$

The effect of the FPC will be noticeable if one or both of the population sizes (*n*'s) is small relative to *n* (the suggested sample size before the FPC) in the formula above.

With the above example we assumed the website is exposed to a large population then:

- Z-Scores:
  - Z<sub>α/2</sub> for α = 0.05 (two-tailed test) ~ 1.96
  - Z_<sub>β</sub> for Power 1 - β = 0.80 (β = 0.20) ~ 0.84

Using a specialized statistical software (like R) with the inputs:
  - p<sub>A</sub> = 0.10
  - p<sub>B</sub> = 0.11 (10% relative lift)
  - α = 0.05
  - *Power* = 0.80

The calculation yields:
  
  $$n \ge \frac{(1.96 + 0.84)^2 \cdot (0.1 \cdot (1-0.1) + 0.11 \cdot (1-0.11))}{(0.1 - 0.11)^2} = 14731.36$$

  $$\text{Required Sample Size }(n) \approx 14,732 \text{ per group}$$

**Step 3: Interpret and Act on the Result**

In order to have an 80% chance of detecting a 1 percentage point lift (10% relative) in conversion rate at a 95% confidence level, the team needs to collect data from 14,732 unique visitors for Control A and 14,732 unique visitors for Variant B, for a total of *29,464* visitors.

## Food for thought

In reality data collection also are very costly and sometime could be impossible to collect exactly all *29,464* valid data points. (Imagine if you are a small business or medical research or survey with non-responses).

Some sacrifices will need to be made:

- 1. Decrease Power (↑ β): Reduce the desired Power (e.g. from 0.80 to 0.70). This **decreases** *n* but **increases** the risk of missing the true effect (inducing Type II Error).
- 2. Increase α (↑ Type I Risk): Increase α (e.g. from 0.05 to 0.10). This **decreases** *n* also but increases the risk of a false win (Claiming an effecet when there isn't).
- 3. Increase MDE (↑ δ): Change the MDE goal (e.g. only try to detect a 20% relative lift, p<sub>B</sub> = 0.12). This will **significantly decreases* *n* because larger effects are easier to find, but it means you are accepting that you won't detect smaller, meaningful lifts. (or if the true effect is less than 20%).

<center>
<img src="we-dont-know-dont-know.gif" width="50%">
</center>

The other problem is that data collection takes time. Also if the duration of the studies takes long time then the test results could be out-dated by the time it finished collecting all ~29,464 visitors.

Often during collection we would need to constantly monitor how the key statistics are changing as well as deciding if we need to halt the test earlier when there are instrumental or systematic problems. Early stopping will need to be considered. 

Also often more recommended way is to create pilot tests with smaller samples on various scenarios before deciding a full-force tests to be conducted. If some scenarios turns out to be poor for some obvious reasons.

Prior similar experiences / knowledge should also be considered when designing a new test. Also at that point, [Bayesian Hypothesis Testing](https://en.wikipedia.org/wiki/Bayesian_inference) usually is the way to go.


## Useful Resources:
- [Power (statistics)](https://en.wikipedia.org/wiki/Power_(statistics))
- [Sample Size Calculator](https://select-statistics.co.uk/calculators/sample-size-calculator-two-proportions/)

