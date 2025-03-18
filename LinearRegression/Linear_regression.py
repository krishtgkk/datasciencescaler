# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

# Load sample data
df = sns.load_dataset('mpg')
df = df.dropna()  # Remove missing values for simplicity

# Define independent (X) and dependent (y) variables
X = df[['horsepower', 'weight', 'acceleration']]  # Features
y = df['mpg']  # Target

# Add a constant to X for statsmodels (required for intercept)
X_const = sm.add_constant(X)

# Fit linear regression model using statsmodels
model = sm.OLS(y, X_const).fit()
print(model.summary())

# Fit linear regression model using scikit-learn (for predictions)
lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)

# Residuals (errors)
residuals = y - y_pred

# --------------------------
# Check Linear Regression Assumptions
# --------------------------

# 1. **Linearity**: Check if the relationship between features and target is linear
sns.pairplot(df, x_vars=['horsepower', 'weight', 'acceleration'], y_vars=['mpg'], kind='scatter')
plt.show()

# 2. **Multicollinearity**: Check for correlation between independent variables
# Calculate VIF (Variance Inflation Factor)
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print("VIF:\n", vif_data)

# 3. **Homoscedasticity**: Check if residuals have constant variance
# Plot residuals vs predicted values
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.show()

# Breusch-Pagan test for homoscedasticity
bp_test = het_breuschpagan(residuals, X_const)
print(f"Breusch-Pagan Test p-value: {bp_test[1]}")

# 4. **Normality of Residuals**: Check if residuals are normally distributed
# Plot histogram of residuals
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.show()

# Q-Q plot of residuals
sm.qqplot(residuals, line='s')
plt.title("Q-Q Plot of Residuals")
plt.show()

# 5. **Independence of Residuals**: Check for autocorrelation
# Durbin-Watson test
dw_test = durbin_watson(residuals)
print(f"Durbin-Watson Test Statistic: {dw_test} (Values close to 2 indicate no autocorrelation)")

# 6. **Outliers**: Check for influential points
# Cook's distance
influence = model.get_influence()
cooks_distance = influence.cooks_distance[0]
plt.stem(np.arange(len(cooks_distance)), cooks_distance, markerfmt=",")
plt.title("Cook's Distance")



# Calculate R2-square and Adjusted R2-Square for a different example

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Sample data
# Let's create a simple dataset with one feature and a target variable
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Convert to DataFrame for better visualization (optional)
data = pd.DataFrame(np.hstack([X, y]), columns=['Feature', 'Target'])
print(data.head())

# Split the data into feature (X) and target (y)
X = data[['Feature']]
y = data['Target']

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate R-squared
r2 = r2_score(y, y_pred)

# Calculate Adjusted R-squared
n = X.shape[0]  # Number of observations
p = X.shape[1]  # Number of features
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Output the results
print(f"R-squared: {r2:.4f}")
print(f"Adjusted R-squared: {adjusted_r2:.4f}")

# Coefficients and intercept
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
plt.show()
