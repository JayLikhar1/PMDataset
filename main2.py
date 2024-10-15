import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

# Load the Data
data = pd.read_csv('predictive_maintenance_dataset.csv')
print(data)

# Describing the Data
print(data.describe())

# Creating First Regression 
# Defining the Dependent and Independent Variables
y = data['failure']
x1 = data[['metric1', 'metric2', 'metric3', 'metric4', 'metric5', 'metric6', 'metric7', 'metric8', 'metric9']]

# Exploration of the data...Plotting of each metric and failure
# for i in x1.columns:
#     plt.scatter(data[i], y)
#     plt.xlabel(i)
#     plt.ylabel('failure')
#     plt.title(f'Scatter plot of {i} vs failure')
#     plt.show()

# Applying the Statsmodels
# Add a constant term to the independent variables
x = sm.add_constant(x1)

# Fit the model
model = sm.OLS(y, x).fit()

# Display the results
print(model.summary())

# Assuming you've selected the significant metrics
X_significant = x[['const', 'metric2', 'metric3', 'metric4', 'metric7', 'metric8']]  # Note: Add 'const' since we added it earlier

# Fit your model again with significant metrics
model_significant = sm.OLS(y, X_significant).fit()

# Make predictions
yhat = model_significant.predict(X_significant)

# Plotting the regression line
plt.figure(figsize=(10, 6))
plt.scatter(y, yhat, color='blue', label='Predicted vs Actual')
plt.plot(y, yhat, lw=4, color='orange', label='Regression Line')
plt.xlabel('Actual Failure')
plt.ylabel('Predicted Failure')
plt.title('Predicted vs Actual Failure')
plt.legend()
plt.show()
