---
title: "Notes"
date: 2021-03-19
draft: false
tags: [Statistics, Machine Learning, Python, SQL]
categories: []
showReadingTime: true
showTableOfContents: true
summary: "General Notes"
---

<center>
<img src="thumb.jpg" width="50%">
</center>

# Introduction

This contains general notes (definitions, code snippets, useful resources etc.) that I found worthy to keep or topics I've forgotten or confused. If you are seeing this, hope this help you.

# Statistics

### Hypothesis Testing

| Error Type | Description | H<sub>0</sub> Status | Test Result |
| ---------- | ----------- | ------------ | ----------- |
| Type I (α) | False Positive (Mistakenly rejecting H<sub>0</sub>) | H<sub>0</sub> is True | Rejected H<sub>0</sub> |
| Type II (β) | False Negative (Mistakenly failing to reject H<sub>0</sub>) | H<sub>0</sub> is False | Failed to Reject H<sub>0</sub> |

The significant level (*α*) is the maximum probability of making a Type I error - *incorrectly rejecting true H<sub>0</sub>* that we are willing to tolerate.

The *p-value* is a number that quantifies the evidence against a null hypothesis (H<sub>0</sub>) in a statistical test. It measures how likely it is to observe the test results (or more extreme results) *if the null hypothesis were true*.

Analogy about these concepts in terms of a courtroom trial:
- Null Hypothesis (H<sub>0</sub>): The defendant is innocent.
- Alternative Hypothesis (H<sub>a</sub>): The defendant is guilty.
- Type I Error (α): Convicting an **innocent** person (false positive). The system sets a high standard of evidence (low α) to avoid this.
- Type II Error (1−Power): Letting a **guilty** person go free (false negative).
- Statistical Power: The sensitivity of the justice system to correctly convict a truly guilty person.
- P-value: The probability of observing the evidence presented (or more extreme evidence) if the defendant was truly innocent (H<sub>0</sub> is true). A very low p-value suggests the evidence is unlikely if H<sub>0</sub> were true.

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
| Benjamini-Hochberg (B-H) | Control FDR <= *q* (The desired FDR level, often *q* = 0.05) | Controls the expected proportion of false rejections (discoveries). | Less Conservative (Highest Power). Widely used in large-scale testing (like genomics, data mining) where a few false positives are acceptable. | An FDR of 0.05 means that to expect at most 5% of the total significant findings to be false positives. |

The key difference is the target:

- FWER Control (Bonferroni, Holm): Focuses on the chance of making a single mistake in the entire set of tests. (High confidence that all significant results are true.)

- FDR Control (Benjamini-Hochberg): Focuses on the proportion of mistakes among the discoveries. (High confidence that most of the significant results are true.)

### Bayesian Hypothesis Testing

Bayesian hypothesis testing is fundamentally about updating the degree of belief in a hypothesis as new data are collect. It treats the unknown population parameters (like the true conversion rate, *p*) as random variables with a probability distribution.

The entire framework is centered on Bayes' Theorem:

$$P(H \mid D) = \frac{P(D \mid H) \times P(H)}{P(D)}$$

Where:
- **P(H | D)**: The Posterior Probability (What we want to know: The probability of the Hypothesis being true given the Data).
- **P(D | H)**: The Likelihood (The probability of observing the Data given the Hypothesis).
- **P(H)**: The Prior Probability (Our initial belief in the Hypothesis before collecting data).
- **P(D)**: The Marginal Likelihood (The probability of the Data itself, which acts as a normalizing constant).

Key Concepts in Bayesian A/B Testing

| Concept                          | Explanation                                                                                                                                                                        | Frequentist Analog                                                                     |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Prior Distribution               | The initial belief about the possible value of a parameter (e.g., conversion rate *p*). This can be non-informative (like a coin flip) or informative (based on historical data). | N/A (Frequentist starts with the Null Hypothesis).                                     |
| Posterior Distribution           | The updated belief in the parameter after observing the data. It is a full probability distribution showing the entire range of likely values for the true parameter.              | Point Estimate and *p*-value.                                                          |
| Probability of Superiority       | The direct probability that one variant's true parameter (e.g., p<sub>B</sub>) is greater than another's (p<sub>A</sub>). Calculated by comparing their posterior distributions.                   | *p*-value (indirect measure of evidence against the null).                             |
| Credible Interval                | The range of values where the true parameter lies with a certain probability (e.g., "There is a 95% chance the true uplift is between *X* and *Y*").                               | Confidence Interval (tells you what would happen if you repeated the test many times). |
| Bayes Factor (BF<sub>10</sub>) | A ratio quantifying the evidence the data provides for the Alternative Hypothesis (H<sub>a</sub>) relative to the Null Hypothesis (H<sub>0</sub>).                                                 | *p*-value (which only measures evidence against H<sub>0</sub>).                                |

The Bayes Factor (BF<sub>10</sub>)The Bayes Factor is the Bayesian analogue to the *p*-value and provides a clear measure of evidence:

$$\text{BF}_{10} = \frac{P(D \mid H_a)}{P(D \mid H_0)}$$

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


# Machine Learning

### Evaluation metrics

**Classification**

- **Accuracy**: The most intuitive metric, it is the ratio of correct predictions to the total number of predictions. It can be misleading if the dataset is imbalanced (e.g., 98% of cases are in one class).
- **Precision (Positive Predictive Value)**: Measures the proportion of positive identifications that were actually correct. It is useful in cases where the cost of a false positive is high.
- **Recall (Sensitivity or True Positive Rate)**: Measures the proportion of actual positives that were identified correctly. It is useful when the cost of a false negative is high (e.g., missing a disease diagnosis).
- **F1-Score**: The harmonic mean of precision and recall. It provides a single score that balances both concerns and is a good general measure for imbalanced classes.
- **Confusion Matrix**: A table that visualizes the performance by breaking down predictions into True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
- **ROC Curve and AUC**: The Receiver Operating Characteristic curve shows the trade-off between the True Positive Rate and False Positive Rate at various threshold settings. The Area Under the Curve (AUC) is a single measure of a model's overall ability to distinguish between classes. 

**Regression**

- Mean Absolute Error (MAE): The average of the absolute differences between the predicted values and the actual values. It gives an idea of the typical error magnitude.
- Mean Squared Error (MSE): The average of the squared differences between predicted and actual values. This metric penalizes large errors more heavily than MAE.
- Root Mean Squared Error (RMSE): The square root of the MSE. It is in the same units as the target variable, making it more interpretable than MSE.- R-squared (R<sup>2</sup>): Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A higher value indicates a better fit. 

**Clustering**

- **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters.
- **Davies-Bouldin Index**: Measures the ratio of within-cluster scatter to between-cluster separation.
- **Calinski-Harabasz Index**: Measures the ratio of between-cluster variance to within-cluster variance.
- **Intraclass Correlation Coefficient (ICC)**: Statistical measure that quantifies the degree of similarity between observations within the same group or cluster. It ranges from 0 to 1, where 1 indicates perfect agreement and 0 indicates no agreement.

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

### Bagging (Bootstrap aggregating)

Bagging, short for Bootstrap Aggregating, is an ensemble learning technique in machine learning designed to improve the stability and accuracy of models, primarily by reducing variance and overfitting. It involves **training multiple versions of a prediction model and combining their predictions to achieve a more robust and accurate final output**.

Multiple subsets of the original training data are created by sampling with replacement. This means that individual data points can be selected more than once in a single subset, and some data points may not appear in certain subsets. Each subset is roughly the same size as the original dataset.

- Parallel Training of Base Learners: A base learning algorithm (e.g., a decision tree) is trained independently on each of these bootstrap samples. These base learners are often referred to as "weak models" because, individually, they might not be highly accurate.
- Aggregation of Predictions: For a new, unseen input, each of the trained base learners makes a prediction.
    - For classification tasks, the final prediction is typically determined by a majority vote among the base learners.
    - For regression tasks, the final prediction is usually the average of the predictions from all base learners.

Benefits of Bagging:

- **Variance Reduction**: By training on different subsets of data and averaging or voting their predictions, bagging helps to reduce the variance of the overall model, making it less sensitive to noise in the training data and less prone to overfitting.
- **Improved Stability and Accuracy**: The aggregation of multiple models generally leads to more stable and accurate predictions compared to a single model.
- **Parallelization**: The training of individual base learners can be done in parallel, which can significantly speed up the training process.

### Multiclass vs. Multilabel Classification

**Multiclass**

Each instance can only be assigned to one class out of a finite set of mutually exclusive classes.
- e.g. Species of a flower.
- Accuracy, precision, recall, F1

**Multilabel**

Each instance can be assigned to multiple labels simultaneously, and the labels are not mutually exclusive.
- e.g. Tagging a news article with multiple topics.
- Hamming loss, precision/recall at k (top-k labels)

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












