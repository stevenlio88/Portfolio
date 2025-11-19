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

This blog contains general notes (definitions, code snippets, useful resources etc.) that I found worthy to keep.

# Statistics

### Hypothesis Testing

| Error Type | Description | H<sub>0</sub> Status | Test Result |
| ---------- | ----------- | ------------ | ----------- |
| Type I (α) | False Positiive (Mistakenly rejecting H<sub>0</sub>) | H<sub>0</sub> is True | Rejected H<sub>0</sub> |
| Type II (β) | False Negative (Mistakenly failing to reject H<sub>0</sub>) | H<sub>0</sub> is False | Failed to Reject H<sub>0</sub> |

The *p-value* is a number that quantifies the evidence against a null hypothesis (H<sub>0</sub>) in a statistical test. It measures how likely it is to observe the test results (or more extreme results) *if the null hypothesis were true*.

The significant level (*α*) is the maximum probability of making a Type I error - *incorrectly rejecting true H<sub>0</sub>* that we are willing to tolerate.

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
| Flattening Arrays/JSON     | Requires complex lateral joins or custom functions.         | SELECT \* FROM t, LATERAL FLATTEN(INPUT => t.array_col);             | FLATTEN is a powerful table function that converts elements within a semi-structured array or object into separate rows. |
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

## WIP