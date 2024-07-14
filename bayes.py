import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import numpy as np

# Load the data
data = pd.read_csv('data.csv')

# Convert all columns to numeric, coercing non-numeric values to NaN
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Create a list to hold the yearly data
yearly_data = [data[str(year)].dropna().values for year in range(1994, 2016)]

with pm.Model() as model:
    # Prior for the mean return
    mu = pm.Normal('mu', mu=0, sd=10)
    
    # Prior for the standard deviation of the return
    sigma = pm.HalfNormal('sigma', sd=10)
    
    # Likelihood for each year's return
    for i, year in enumerate(range(1994, 2016)):
        # Adjust the standard deviation for the number of instances
        adjusted_sigma = sigma / tt.sqrt(len(yearly_data[i]))
        pm.Normal(f'year_{year}', mu=mu, sd=adjusted_sigma, observed=yearly_data[i])
    
    # Sample from the posterior
    trace = pm.sample(2000)

    # Print out whether the investment is improving or failing
    mean_mu = np.mean(trace['mu'])
    print(f"Mean of mu: {mean_mu}")
    if mean_mu > 0:
        print("Investment is improving.")
    else:
        print("Investment is failing.")

    # Plot the posterior within the model context
    pm.plot_posterior(trace)
