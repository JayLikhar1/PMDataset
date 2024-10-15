import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

#Load the Data

data = pd.read_csv('predictive_maintenance_dataset.csv')
print(data)

# Describing the Data
print(data.describe())

# Creating First Regression 
# Defining the Dependant and Independant Variables

y = data['failure']
x1 = data[['metric1', 'metric2', 'metric3', 'metric4', 'metric5', 'metric6', 'metric7', 'metric8', 'metric9']]

# Exploration of the data...Plotting of each metrics and failure
#for i in x1.columns:
    #plt.scatter(data[i], y)
    #plt.xlabel(i)
    #plt.ylabel('failure')
    #plt.title(f'Scatter plot of {i} vs failure')
    #plt.show()


# Applying the Statsmodels
# Add a constant term to the independent variables

from statsmodels.tools import add_constant
import statsmodels.api as sm
print(sm)
x = sm.add_constant(x1)

# Fit the model
model = sm.OLS(y, x).fit()

# Display the results
print(model.summary())

# Assuming you've selected the significant metrics
X_significant = x[['metric2', 'metric3', 'metric4', 'metric7', 'metric8']]
X_significant = sm.add_constant(X_significant)  # Adds a constant term to the predictor


# Fit your model
model.fit(X_significant, y)

# Make predictions
yhat = model.predict(X_significant)  # or for the test set

fig = plt.plot(x1, yhat, lw = 4, C = 'Orange' , label = 'regression Line' )
plt.xlabel( data[['metric1', 'metric2', 'metric3', 'metric4', 'metric5', 'metric6', 'metric7', 'metric8', 'metric9']]
)
plt.ylabel('failure')
plt.show()
