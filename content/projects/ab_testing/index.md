---
title: "A/B Testing"
date: 2022-04-02
draft: false
tags: [Statistics, Hypothesis Testing, Data Science, Data Analytics, Web Analytics, Python, R]
categories: []
showReadingTime: true
showTableOfContents: True
summary: "Process of A/B testing for web analytics"
---

<center>
<img src="thumb.jpg" width="50%">
</center>

## Introduction

This project discusses the process of A/B testing on web analytics, using an example to determine if proposed improvements made to the website homepage impact the click-through rate to showcase the methodology.

## Background

A/B testing is a form of statistical hypothesis testing, specifically two-sample hypothesis testing, conducted through a randomized experiment involving a control group and an experimental group. 
It aims to determine whether proposed changes (experiment variations) have a measurable impact compared to the original (control).

Typically, A/B testing is used to assess differences between two samples: a control and an experimental group. However, this methodology can also extend to testing multiple samples simulataneously.

This process was discussed in the paper *[Improving Library User Experience with A/B Testing: Principles and Process](https://quod.lib.umich.edu/w/weave/12535642.0001.101?view=text;rgn=main)* by Scott W.H. Young (2014). 

The original study aimed to address why the *Interact* category button has the lowest click-through rate among all categories on the Montana State University Library homepage.
Various proposed variations, such as *Connect*, *Learn*, *Help*, and *Services* were tested via user survey to gather student suggestions on the changes.
Then an A/B testing of multiple samples was conducted to evaluate the click-through rate of each variation and determine which one successfully increased the click-through rate.

## Data

The data was collected based on 100% of website vistors during May 29, 2013 to June 18, 2013 using Google Analytics and Crazy Egg results to collect user's various activities on the website.
Then each of the variations was displayed to users randomly and the total number of clicks along with other activities was collected.

The Crazy Egg data can be found [here](https://scholarworks.montana.edu/items/ac4aa35e-e5b6-4c60-b0c3-6a60ef0584f2).

The data consists of 5 csv files containing the number of clicks on each element on the webpage and corresponding to the 4 new suggested categories variations (*Connect*, *Learn*, *Help*, and *Services*) and the original (*Interact*) category.  

After consolidating the csv files and data manipulation, the summary of the clicks for each variation are as followed:

<style>
table {
  font-family: Arial, Helvetica, sans-serif;
  border-collapse: collapse;
  width: 100%;
}

td, th {
  border: 1px solid #ddd;
  padding: 8px;
  word-wrap: break-word;
}

th {
  padding-top: 12px;
  padding-bottom: 12px;
  text-align: left;
  background-color: #04AA6D;
  color: white;
}
</style>

<table>
  <tr>
    <th>Webpage Variation</th>
    <th>Total Clicks</th>
    <th>Home Page Clicks</th>
    <th>Adjusted Clicks</th>
    <th>Element Clicks</th>
  </tr>
  <tr>
    <td>Interact</td>
    <td>3,714</td>
    <td>1,291</td>
    <td>2,423</td>
    <td>42</td>
  </tr>
  <tr>
    <td>Connect</td>
    <td>1,587</td>
    <td>83</td>
    <td>1,504</td>
    <td>53</td>
  </tr>
   <tr>
    <td>Learn</td>
    <td>1,652</td>
    <td>83</td>
    <td>1,569</td>
    <td>21</td>
  </tr>
   <tr>
    <td>Help</td>
    <td>1,717</td>
    <td>122</td>
    <td>1,595</td>
    <td>38</td>
  </tr>
   <tr>
    <td>Services</td>
    <td>1,348</td>
    <td>49</td>
    <td>1,299</td>
    <td>45</td>
  </tr>
</table>

The number of clicks on the home page is excluded from the total clicks to calculate the adjusted clicks to better captures the clicks on the elements after visitor landed on the home page.
The element clicks indicates the number of clicks on each of the variations presented to site visitors.

<details>
  <summary><u>Click me to see the data manipulation in Python:</u></summary>
  
```python
import os
import glob
import pandas as pd
import re

filepath = r"CrazyEgg"

df_list = []

for subdir, dirs, files in os.walk(filepath):
    # Find all CSV files in the current directory
    csv_files = glob.glob(os.path.join(subdir, "*.csv"))
    for file in csv_files:
        # Read the CSV file
        df = pd.read_csv(file)
        # Optionally, add a column to track the source file
        df["source_file"] = file
        # Append the dataframe to the list
        df_list.append(df)

# Concatenate all dataframes into one
crazy_egg = pd.concat(df_list, ignore_index=True)

# Extract filename using regex
pattern = r".*[\\/](?P<filename>[^\\/]+)$"
crazy_egg["source_file"] = crazy_egg["source_file"].apply(
    lambda x: re.sub(pattern, r"\1", x)
)

# Creating webpage ID column from filenames.
webpage_pattern = r"(?<=- )\w+"
crazy_egg["webpage"] = crazy_egg["source_file"].apply(
    lambda x: re.search(webpage_pattern, x).group(0)
    if re.search(webpage_pattern, x)
    else None
)

# Create base total click through number table
total = crazy_egg[crazy_egg["Snapshot information"].fillna("").str.contains("created")][
    ["webpage", "Snapshot information"]
]
total.rename(columns={"Snapshot information": "total clicks"}, inplace=True)
click_pattern = r"\d+(?= clicks)"
total["total clicks"] = total["total clicks"].apply(
    lambda x: re.search(click_pattern, x).group(0)
    if re.search(click_patterh, x)
    else None
)
total["total clicks"] = total["total clicks"].astype(int)

# Create home page click through number
home_clicks = crazy_egg[crazy_egg["Name"].str.contains("Home")][
    ["webpage", "No. clicks"]
]
home_clicks.rename(columns={"No. clicks": "homepage clicks"}, inplace=True)
home_clicks["homepage clicks"] = home_clicks["homepage clicks"].astype(int)

# Create element click through number
element_clicks = crazy_egg[crazy_egg["Name"] == crazy_egg["webpage"].str.upper()][
    ["webpage", "No. clicks"]
]
element_clicks.rename(columns={"No. clicks": "element clicks"}, inplace=True)
element_clicks["element clicks"] = element_clicks["element clicks"].astype(int)

# Combine create final click through number table
click_through = pd.merge(total, home_clicks)
click_through["adjusted clicks"] = (
    click_through["total clicks"] - click_through["homepage clicks"]
)
click_through = pd.merge(click_through, element_clicks)

click_through
```
</details>

<details>
  <summary><u>Click me to see the data manipulation in R:</u></summary>
  
```r
library(datasets)
library(tidyverse)

filepath <- "CrazyEgg"

# Getting the filenames of the .csv's and put in a data frame.
crazy_egg <- data.frame(filename = list.files(
  path = filepath, pattern = "*.csv",
  full.names = TRUE,
  recursive = TRUE
))

# Consolidate all csv files.
crazy_egg$raw <- map(as.character(crazy_egg$filename), read_csv)

# Regex pattern to extract filenames
pattern <- ".*/([^/\\\\]+)$"
crazy_egg$filename <- sub(pattern, "\\1", crazy_egg$filename)

# Creating webpage name from filenames (Interact, Connect, Learn, Help, Services).
crazy_egg$webpage <- str_extract(
  string = crazy_egg$filename,
  pattern = "(?<=- )\\w+"
)

# Unnesting data frames to make one big dataframe.
crazy_egg <- unnest(crazy_egg)

# Making Column names more readable.
colnames(crazy_egg) <- make.names(colnames(crazy_egg))

# Extracting click count and create click-through base table.
click_through <- filter(crazy_egg, grepl("created", Snapshot.information)) %>%
  select(webpage, Snapshot.information)
colnames(click_through) <- c("webpage", "clicks")

click_through$clicks <- str_extract(
  string = click_through$clicks,
  pattern = "\\d+(?= clicks)"
) %>%
  as.numeric()

# Extracting homepage clicks.
click_through$home_page_clicks <- filter(crazy_egg, grepl("Home", Name)) %>%
  select(No..clicks) %>%
  unlist()

# Homepage adjusted clicks.
click_through <- mutate(click_through, adjusted_clicks = clicks - home_page_clicks)

# Target clicks (e.g., Interact, Connect, Learn, Help, Services button).
click_through$target_clicks <- filter(crazy_egg, grepl(paste(toupper(click_through$webpage), collapse = "|"), Name)) %>%
  select(No..clicks) %>%
  unlist()

click_through <- mutate(click_through, click_rate = target_clicks / adjusted_clicks)

click_through
```
</details>

## Visualization

<div>
<center>
	{{< plotly json="click_through.json" height="50%" >}}
</center>
</div>

<details>
  <summary><u>Click me to see the code for the graphs in Python:</u></summary>
  
```python
import plotly.express as px

click_through["click_through_rate"] = (
    click_through["element clicks"] / click_through["adjusted clicks"]
)

fig = px.bar(
    click_through,
    x="webpage",
    y="click_through_rate",
    text=click_through["click_through_rate"].apply(
        lambda x: f"{x*100:.2f}%"
    ),  # Format as percentage with two decimal places
    title="Click-through Rate of Each Variation",
)

# Update layout to customize the plot
fig.update_layout(
    title={
        "text": "Click-through Rate of Each Variation",
        "y": 0.9,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    },
    xaxis_title="Webpage",
    yaxis_title="Click-through Rate (%)",
    autosize=False,
    width=600,
    height=600,
)

# Update y-axis tick labels to percentage format
fig.update_yaxes(
    tickformat=".1%", range=[0, 0.04]
)  # Tick format to display as percentage with two decimal places

# Update hover template to show more decimal places
fig.update_traces(hovertemplate="Webpage: %{x}<br>Click-through Rate: %{y:.4f}")

# Show the figure
fig.show()
```
</details>

The graph shows the click-through rate of each webpage variation and an estimated 95% confidence interval assuming a Binomial Distribution 
where probability of success (*p*) is the click-through rate and (*n*) is the adjusted number of clicks on each variations.

Click-through rate is defined by dividing the number of Element Clicks by the Adjusted Clicks. (i.e. percentage of total clicks results in a click on the element)

As we can see the *Connect*, *Services* variations both has much higher click-through rate than the original with *Interact* (2nd last).
Also by assessing the 95% confidence intervals, we can get a sense already on if there are statistical differences between some of these click-through rates given if the confidence levels are overlapping or not.


## A/B Testing

Here, we conducted a pairwise hypothesis test with a significance level of α = 0.05 to determine which webpage variations show statistically significant differences in click-through rate. 
Given that we are comparing proportions, a pairwise proportion test was conducted.

The p-value of the test results are correct for multiple samples to minimize chances of Type I error (false positive) or incorrectly concluded that there are differences in the click-through rate when there isn't.
See here for the details on the [types of errors](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors).

The pair-wise proportions is conducted by comparing the click-through rate of each webpage variation against the other webpage variation.

The assumption of the hypothesis test (Null hypothesis) is: There are no differences in the click-through between the two webpage variations
The alternative to the assumption (Alternative hypothesis) is: There are differences in the click-through between the two webpage variations

<details>
  <summary><u>Click me to see the implementation of the test in Python:</u></summary>
 
```Python
import statsmodels.stats.proportion as prop
from statsmodels.stats.multitest import multipletests

# Assigning names to successes and trials
successes = click_through["element clicks"].values
trials = click_through["adjusted clicks"].values
webpages = click_through["webpage"].values

# Perform pairwise proportion tests
results = []
for i in range(len(successes)):
    for j in range(i + 1, len(successes)):
        success1 = successes[i]
        trial1 = trials[i]
        success2 = successes[j]
        trial2 = trials[j]
        count = np.array([success1, success2])
        nobs = np.array([trial1, trial2])

        # Perform two independent binomial samples
        z_stat, p_value = prop.test_proportions_2indep(
            success1, trial1, success2, trial2
        )
        # Store results
        result = {
            "Webpage A": webpages[i],
            "Webpage B": webpages[j],
            "z_statistic": z_stat,
            "p_value": p_value,
        }
        results.append(result)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Adjust p-values using Bonferroni correction
p_adjusted_bonf = multipletests(results_df["p_value"], method="bonferroni")[1]
results_df["p_adjusted_bonf"] = p_adjusted_bonf

# Adjust p-values using Benjamini-Hochberg (BH) correction
p_adjusted_bh = multipletests(results_df["p_value"], method="fdr_bh")[1]
results_df["p_adjusted_bh"] = p_adjusted_bh

# Display results
results_df["test_result"] = results_df["p_value"].apply(
    lambda p: "Reject" if p < 0.05 else "Failed to reject"
)
results_df["test_result_adjusted_bonf"] = results_df["p_adjusted_bonf"].apply(
    lambda p: "Reject" if p < 0.05 else "Failed to reject"
)
results_df["test_result_adjusted_bh"] = results_df["p_adjusted_bh"].apply(
    lambda p: "Reject" if p < 0.05 else "Failed to reject"
)
results_df
```
</details>

<details>
  <summary><u>Click me to see the implementation of the test in R:</u></summary>

```r
successes <- click_through$element_clicks
trials <- click_through$adjusted_clicks
names(successes) <- click_through$webpage
names(trials) <- click_through$webpage

# Pair-wise proportion test with Bonferroni Correction
pairwise.prop.test(successes,
                   trials,
                   p.adjust.method = "bonferroni"
)

# Pair-wise proportion test with Benjamini-Hochberg (BH) Correction
pairwise.prop.test(successes,
                   trials,
                   p.adjust.method = "BH"
)
```
</details>


## Results

<table id="result">
  <tr>
	<th>Webpage A</th>
	<th>Webpage B</th>
	<th>z_statistic</th>
	<th>p-value</th>
	<th>p-value (Bonf. adj.)</th>
	<th>p-value (BH. adj.)</th>
	<th>Test Result</th>
	<th>Test Result (Bonf. adj.)</th>
	<th>Test Result (BH. adj.)</th>
  </tr>
  <tr>
	<td>Learn</td>
	<td>Interact</td>
	<td>-0.932</td>
	<td>0.350</td>
	<td>1.000</td>
	<td>0.389</td>
	<td>Failed to reject null</td>
	<td>Failed to reject null</td>
	<td>Failed to reject null</td>
  </tr>
  <tr>
	<td>Learn</td>
	<td>Help</td>
	<td>-2.139</td>
	<td>0.032</td>
	<td>0.324</td>
	<td>0.064</td>
	<td><b>Reject null</b></td>
	<td>Failed to reject null</td>
	<td>Failed to reject null</td>
  </tr>
   <tr>
	<td>Learn</td>
	<td>Services</td>
	<td>-3.609</td>
	<td>0.000</td>
	<td>0.003</td>
	<td>0.002</td>
	<td><b>Reject null</b></td>
	<td><b>Reject null</b></td>
	<td><b>Reject null</b></td>
  </tr>
   <tr>
	<td>Learn</td>
	<td>Connect</td>
	<td>-3.879</td>
	<td>0.000</td>
	<td>0.001</td>
	<td>0.001</td>
	<td><b>Reject null</b></td>
	<td><b>Reject null</b></td>
	<td><b>Reject null</b></td>
  </tr>
   <tr>
	<td>Interact</td>
	<td>Help</td>
	<td>-1.423</td>
	<td>0.155</td>
	<td>1.000</td>
	<td>0.193</td>
	<td>Failed to reject null</td>
	<td>Failed to reject null</td>
	<td>Failed to reject null</td>
  </tr>
   <tr>
	<td>Interact</td>
	<td>Services</td>
	<td>-3.050</td>
	<td>0.000</td>
	<td>0.023</td>
	<td>0.006</td>
	<td><b>Reject null</b></td>
	<td><b>Reject null</b></td>
	<td><b>Reject null</b></td>
  </tr>
   <tr>
	<td>Interact</td>
	<td>Connect</td>
	<td>-3.301</td>
	<td>0.001</td>
	<td>0.010</td>
	<td>0.003</td>
	<td><b>Reject null</b></td>
	<td><b>Reject null</b></td>
	<td><b>Reject null</b></td>
  </tr>
   <tr>
	<td>Help</td>
	<td>Services</td>
	<td>-1.705</td>
	<td>0.088</td>
	<td>0.882</td>
	<td>0.126</td>
	<td>Failed to reject null</td>
	<td>Failed to reject null</td>
	<td>Failed to reject null</td>
  </tr>
   <tr>
	<td>Help</td>
	<td>Connect</td>
	<td>-1.858</td>
	<td>0.063</td>
	<td>0.631</td>
	<td>0.105</td>
	<td>Failed to reject null</td>
	<td>Failed to reject null</td>
	<td>Failed to reject null</td>
  </tr>
   <tr>
	<td>Services</td>
	<td>Connect</td>
	<td>-0.071</td>
	<td>0.943</td>
	<td>1.000</td>
	<td>0.943</td>
	<td>Failed to reject null</td>
	<td>Failed to reject null</td>
	<td>Failed to reject null</td>
  </tr>
</table>

Total of 10 pair-wise proportion tests were conducted and the test statistics (z_statistic) and the p-value from the test were calculated.

Then adjustments on the p-values were applied using the [Bonferroni Correction](https://en.wikipedia.org/wiki/Bonferroni_correction) and [Benjamini-Hochberg Procedure](https://en.wikipedia.org/wiki/False_discovery_rate) methods were applied. The adjusted p-values are 
shown in the table p-value (Bonf. adj.) and p-value (BH. adj.).

At α = 0.05 significant level, the pair-wise proportion tests with multiple samples adjustments shows that there are statistical significance
between *Learn* and *Connect* and *Learn* and *Connect* variations.

More importantly, both *Connect* and *Services* variations are significantly different to the original *Interact* variation.
Also the differences between *Connect* and *Services* has no statistical differences based on the test results. The test results aligns with what we
saw earlier from the confidence intervals of the click-through rate.

In other words, both *Connect* and *Services* would be a good candidate to replace the original *Interact* button for improving the user engagement
in navigating to the detail contents after landing on the home page.

The original paper also explored the Drop-off rate (% of users leave the site from the category page) and Homepage-return rate (% of users returns to the homepage from the category page) for each of these variations as well.
Which are also important beside click-through rate, as these metrics also provide further evidence that the proposed changes increases user engagement to the category page.

## References

[A/B Testing](https://en.wikipedia.org/wiki/A/B_testing)  
[Pair-wise Proportion Test](https://library.virginia.edu/data/articles/pairwise-comparisons-of-proportions)  
[Type of Errors](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors)  
[Bonferroni Correction](https://en.wikipedia.org/wiki/Bonferroni_correction)  
[Benjamini-Hochberg](https://en.wikipedia.org/wiki/False_discovery_rate)  
[Improving Library User Experience with A/B Testing: Principles and Process (Young, 2014)](https://quod.lib.umich.edu/w/weave/12535642.0001.101?view=text;rgn=main)