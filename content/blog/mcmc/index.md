---
title: "Monte Carlo Simulation"
date: 2022-05-10
draft: false
tags: [Statistics, Data Science, Python]
categories: []
showReadingTime: true
showTableOfContents: false
summary: "Exploring Monte Carlo Simulation"
---

<center>
<img src="thumb.jpg", width="50%">
</center>

## Introduction

This blog explores some of the applications of Monte Carlo Simulation.

## Monte Carlo Simulation

Monte Carlo Simulation or Monte Carlo methods/experients is a broad class of computational algorithms that 
rely on repeated random sampling to obtain numerical results. The underlying concept is to use randomness to 
solve problems that might be deterministic in principle.

It is mostly use in problems such as optimization, numerical integration and generating draws from a probability distribution.
This provides a very powerful tool in science, engineering, artificial intelligence, finance and cryptography etc espcially in
risk assessment and model the probability of different outcomes in a process that cannot easily be predicted due to the intervention
of random variables.

However, its main draws back comes from users has to choose between accuracy and computational cost, the curse of 
dimensionality and the reliability of random number generators. 

The overview of the Monte Carlo methods is as follow:

1. Define a domain of possible inputs
2. Generate inputs randomly from a probability distribution over the domain
3. Perform a deterministic computation of the outpus
4. Aggregate the results

## Estimating Pi

This is the most used example for introduction to the Monte Carlo Simulation - to estimate the value of Pi. i.e. using Monte Carlo Simulation for numerical integration.

It is known that the circumference is: 2 π r hence π is the ratio of circumference and 2 times of the radius (diameter) of a circle, and the area of a circle is: π r<sup>2</sup>. Where r is the radius of the circle. Therefore, the area of a circle of radius 1 is π.

The idea is to imagine a circle with radius 1 (i.e. diameter of 2) sitting inside a 2 x 2 shaped square. The ratio of the areas between the the circle and the square is
$$ R = \frac{\text{area  of  circle}}{\text{area  of  square}} = \frac{\pi r^2}{2 * 2} = \frac{π}{4} $$

Hence if we have the ratio of the areas (R) between the circle and the square then we know that
$$ \pi = 4 * R = 4 * \frac{\text{area of circle}}{\text{area of square}} $$

One may ask without knowing π how the area of the circle to be calculated then? This is where Monte Carlo Simulation comes in. We can first draw uniformed random samples of dots within the square then count the number of samples that fall within the circle inside the square. By calculating the number of samples fall inside the circle vs. total of random samples generated we can then estimate the value ratio between the two areas, and therefore estimating π.

1. Draw n random samples (x, y) coordinates uniformly within the bound of the square
2. Collects the samples (x, y) that falls within the the circle (x<sup>2</sup> + y<sup>2</sup> <= r<sup>2</sup> = 1)
3. Calculate the ratio (R) of samples inside the circle vs n
4. Multiply the ratio (R) by 4 to get the estimated value of π

<details>
  <summary><u>Click me to see the implementation in Python:</u></summary>

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def estimate_pi(n):
    """
    Monte Carlo Simulation: Estimate pi

    Args:
        n (int): Number of samples
    Returns:
        x (list): List of randomly generated x coordinates
        y (list): List of randomly generated y coordinates
        inside_circle (list): List of boolean values indicates if the x, y pairs inside the unit circle
        pi_estimate (float): Estimate of Pi
    """
    r = 1.0
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    inside_circle = x**2 + y**2 <= r**2
    pi_estimate = (inside_circle.sum() / n) * 4

    return x, y, inside_circle, pi_estimate


i = 1000
pi_estimates = []
images = []

x, y, inside_circle, pi_estimate = estimate_pi(i)
pi_estimates.append(pi_estimate)

####################
# Sampling Process #
####################
fig = go.Figure()

# Points inside the circle
fig.add_trace(
    go.Scatter(
        x=x[inside_circle],
        y=y[inside_circle],
        mode="markers",
        marker=dict(color="blue", size=4),
        name=f"Inside: {inside_circle.sum()}",
        legendgroup="1",
    ),
)
# Points outside the circle
fig.add_trace(
    go.Scatter(
        x=x[~inside_circle],
        y=y[~inside_circle],
        mode="markers",
        marker=dict(color="red", size=4, opacity=0.5),
        name=f"Outside: {~inside_circle.sum()}",
        legendgroup="1",
    ),
)
# Add a circle to the plot
circle = go.Scatter(
    x=np.cos(np.linspace(0, 2 * np.pi, 1000)),
    y=np.sin(np.linspace(0, 2 * np.pi, 1000)),
    mode="lines",
    line=dict(color="green"),
    name="Circle",
    legendgroup="1",
)
fig.add_trace(circle)
fig.update_xaxes(scaleanchor="y", range=[-1, 1], scaleratio=1)
fig.update_yaxes(scaleanchor="x", range=[-1, 1], scaleratio=1)
fig.add_annotation(
    x=0.5,
    y=-1.1,
    xref="paper",
    yref="paper",
    text=f"Outside: {(~inside_circle).sum()}",
    showarrow=False,
    font=dict(color="red", size=12),
    xanchor="left",
)
fig.add_annotation(
    x=0.5,
    y=-1.2,
    xref="paper",
    yref="paper",
    text=f"Inside: {inside_circle.sum()}",
    showarrow=False,
    font=dict(color="blue", size=12),
    xanchor="left",
)

fig.update_layout(
    title=f"Monte Carlo Simulation for Estimating π<br>(Estimated π = {inside_circle.sum()}/{len(x)}; = {pi_estimates[-1]:.3f})",
    width=600,
    height=600,
)
fig.show()
```
</details>

<center>
<img src="simulation.gif">
</center>

The process illustrated above uses Monte Carlo Simulation to estimate the value of π through numerical integration. 
As expected, more samples improves the estimates. Theoretically with unlimited samples, the ratio will equal to the true value of π.
However, in reality it is not possible to achieve this, and the estimates only stablized around the true value.  

This simple simulation requires taking approximately 10k random samples to generate an estimate. One simulation
cannot guarantee finding the true value. In real-life scenarios, more complex simulations require more samples and calculations.

Using statistics, we can consider each simulation estimate as a random variable. Instead of relying on one simulation result, we can rerun the simulations with a fixed number of samples. By constructing a confidence interval from all the simulation results, we can estimate the range within which the true value is likely to fall, rather than expecting to obtain the exact true value. 

## Stock Prices

Here is an illustration of another application of Monte Carlo Simulation in estimaing future stock values. This simulation used the daily stock price of Google (GOOG) from Jan 1, 2010 to Jan 31, 2021 and ran 1,000 simulations to predict future stock values for 252 trading days. The results will be compared against the actual stock values from Feb 1, 2021 to Jan 31, 2022.

The simulation predicts what would happened over 252 trading days after investing $10,000 in GOOG on Feb 1, 2021, i.e. what the portfolio would be worth in 252 trading days. The random samples are generated using a simple Normal Distribution to model the daily percentage changes in the adjusted close price of the stock. The Normal Distribution is based on the mean and variances of the percentage change in the adjusted close prices from the period Jan 1, 2010 to Jan 31, 2021. 

Here are the simulation results:
<center>
<img src="simulation2.gif">
</center>

The gray lines represent 1,000 simulation runs on the portfolio during the 252 trading days. The red line shows the average of all simulations and the 95% confidence intervals of the daily values are shaded in blue. 

The point of this simulation is not to predict the exact stock price on any given day but to provide insight into the ranges of the most likely outcomes and risks based on our assumptions about the randomness of this particular stock prices.

As expected, the variance of the predictions increases as we project further into the future. This can be seen in the widening of the percentile ranges towards the end of the 252 days. On the 252nd day, we would expected that the average simulated portfolio value to be $14,065.01 (+40.7%), while the actual value was $14,020.51, an overestimated for $44.5.

It is important to note that this simulation is based on several assumptions:
1. Percentage changes in the stock prices are Normally distributed
2. The distribution follows the mean and variances from historical percentage changes in stock prices (11 years in this case)
3. Percentage changes in stock prices are truly random
4. 1000 simulations are sufficient

If all these assumptions held true, predicting stock prices would be purely gambling, which it obviously isn't, or is it ? ( ͡° ͜ʖ ͡°)

From this simulation on Google's stock prices, it appears that the simulation did a fairly good job of estimating the future stock
prices. The average of these simulations was very close to the actual prices at the end of the 252 trading days, and 95% of these simulations captured the true values. However, for most of the period, the simulation averages were overestimating the true value (the red line above the green line but still close). 

This was due to Google's stock prices being relatively stable (not volatile) during this period. The same prediction for more volatile stocks may not be applicable.

Actual simulations in the real-world would be more complex, involving more sophisticated models and additional auxillary data.

<details>
  <summary><u>Click me to see the implementation in Python:</u></summary>

```python
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import imageio

# Download historical data for a stock (e.g., S&P 500 index)
ticker = "GOOG"
data = yf.download(ticker, start="2010-01-01", end="2022-01-31")
data["Returns"] = data["Adj Close"].pct_change()

# Calculate the mean and standard deviation of daily returns
mean_return = data[data.index.to_series().between("2020-02-01", "2021-01-31")][
    "Returns"
].mean()
std_return = data[data.index.to_series().between("2020-02-01", "2021-01-31")][
    "Returns"
].std()

# Monte Carlo simulation parameters
num_simulations = 1000
num_days = 252  # Number of trading days in a year
initial_investment = 10000  # Initial investment amount

# Initialize an array to store the simulated portfolio values
simulated_portfolios = np.zeros((num_simulations, num_days))

for i in range(num_simulations):
    # Generate daily returns using a normal distribution
    daily_returns = np.random.normal(mean_return, std_return, num_days) + 1

    # Simulate the portfolio value over time
    portfolio_value = np.zeros(num_days)
    portfolio_value[0] = initial_investment

    for t in range(1, num_days):
        portfolio_value[t] = portfolio_value[t - 1] * daily_returns[t - 1]

    simulated_portfolios[i] = portfolio_value

# Calculate the mean and 5th/95th percentiles of the final portfolio values
final_values = simulated_portfolios[:, -1]
mean_final_value = np.mean(final_values)
percentile_5th = np.percentile(final_values, 5)
percentile_95th = np.percentile(final_values, 95)

actual_portfolio_value = np.zeros(num_days)
actual_portfolio_value[0] = initial_investment
actual_daily_returns = data[data.index.to_series().between("2021-02-02", "2022-01-31")][
    "Returns"
]

for t in range(1, num_days):
    actual_portfolio_value[t] = actual_portfolio_value[t - 1] * (
        1 + actual_daily_returns.iloc[t - 1]
    )


# Prepare data for Plotly
days = np.arange(num_days)
simulated_data = pd.DataFrame(simulated_portfolios, columns=[f"Day {i}" for i in days])
simulated_data["Simulation"] = np.arange(num_simulations)

# Melt the DataFrame to long format for Plotly
long_format = simulated_data.melt(
    id_vars="Simulation", var_name="Day", value_name="Portfolio Value"
)

# Convert 'Day' to an integer for plotting
long_format["Day"] = long_format["Day"].str.extract("(\d+)").astype(int)


mean_value = long_format.groupby("Day")["Portfolio Value"].mean()
percentile_5th = long_format.groupby("Day").agg(
    min_val=("Portfolio Value", "min"),
    percentile_05=("Portfolio Value", lambda x: x.quantile(0.05)),
)
percentile_95th = long_format.groupby("Day").agg(
    min_val=("Portfolio Value", "min"),
    percentile_95=("Portfolio Value", lambda x: x.quantile(0.95)),
)

# Create the initial plot with the first frame
fig = go.Figure()

# Add traces for each simulation
for sim in long_format["Simulation"].unique():
    sim_data = long_format[long_format["Simulation"] == sim]
    fig.add_trace(
        go.Scatter(
            x=sim_data["Day"],
            y=sim_data["Portfolio Value"],
            mode="lines",
            name=f"Simulation {sim}",
            line=dict(width=0.5),
            marker=dict(color="gray", opacity=0.1),
        )
    )

# Add the 5th percentile line
fig.add_trace(
    go.Scatter(
        x=days,
        y=percentile_5th["percentile_05"],
        mode="lines",
        name="5th Percentile",
        line=dict(color="blue", dash="dash"),
    )
)

# Add the 95th percentile line
fig.add_trace(
    go.Scatter(
        x=days,
        y=percentile_95th["percentile_95"],
        mode="lines",
        name="95th Percentile",
        line=dict(color="blue", dash="dash"),
    )
)

# Add the filled area between the 5th and 95th percentiles
fig.add_trace(
    go.Scatter(
        x=days,
        y=percentile_95th["percentile_95"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        fill=None,
    )
)
fig.add_trace(
    go.Scatter(
        x=days[:i],
        y=percentile_5th["percentile_05"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        fill="tonexty",
        fillcolor="rgba(173, 216, 230, 0.3)",
        name="5th-95th Percentile",
    )
)

fig.add_trace(
    go.Scatter(
        x=days,
        y=mean_value,
        mode="lines",
        name="Average",
        line=dict(width=0.5),
        marker=dict(color="red"),
    )
)

fig.add_trace(
    go.Scatter(
        x=days,
        y=actual_portfolio_value,
        mode="lines",
        name="Actual Values",
        line=dict(width=0.5),
        marker=dict(color="green"),
    )
)


# Update layout with animation settings
fig.update_layout(
    title="Monte Carlo Simulation of Portfolio Values",
    title_x=0.5,
    xaxis=dict(title="Day", range=[0, max(days)]),
    yaxis=dict(title="Portfolio Value", range=[0, simulated_portfolios.max()]),
    width=900,
    height=600,
)
fig.show()

```
</details>

