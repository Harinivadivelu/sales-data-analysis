import pandas as pd
 # Load the dataset while treating all columns as strings
df = pd.read_csv("Walmart.csv", dtype=str)
 
 # Display the first 5 rows
print(df.head())
print(df.dtypes)
print(df.describe())
df['Date'] = pd.to_datetime(df['Date'],
format='%d-%m-%Y')
print(df.duplicated().sum())
df=df.drop_duplicates()
import pandas as pd
 
 # Load dataset
df = pd.read_csv("Walmart.csv")
 
 # Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
 
 # Convert numeric columns from object to float
numeric_cols = ['Weekly_Sales', 'Fuel_Price', 'CPI', 'Unemployment']
df[numeric_cols] = df[numeric_cols].astype(float)
 
 # Check data types after conversion
print(df.dtypes)
print(df.isnull().sum())
print(df.dtypes)
print(df.describe())
print(df.isnull().sum())
import matplotlib.pyplot as plt
 
df.groupby("Date")["Weekly_Sales"].sum().plot(figsize=(12,5), title="Total Weekly Sales Over Time")
plt.xlabel("Date") 
plt.ylabel("Total Weekly Sales")
plt.show()
import seaborn as sns
 
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
top_stores = df.groupby("Store")["Weekly_Sales"].sum().sort_values(ascending=False).head(5)
print(top_stores)

top_stores.plot(kind='bar', title="Top 5 Stores by Sales", color='skyblue')
plt.show()
sns.boxplot(x=df["Holiday_Flag"], y=df["Weekly_Sales"])
plt.title("Holiday vs Non-Holiday Sales")
plt.show()
print(df.isnull().sum())
df.fillna(df.mean(),inplace=True)
df.dropna(inplace=True)
print(df.isnull().sum())  # Should return 0 for all columns
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.boxplot(x=df['Store'], y=df['Weekly_Sales'])
plt.xlabel("Store")
plt.ylabel("Weekly Sales")
plt.title("Distribution of Weekly Sales Across Stores")
plt.xticks(rotation=90)  # Rotate store labels for better readability
plt.show()
df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date is in datetime format
df.sort_values('Date', inplace=True)  # Sort by date

df['Weekly_Sales_MA'] = df['Weekly_Sales'].rolling(window=10).mean()  # 10-week moving average

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Weekly_Sales'], label='Actual Sales', alpha=0.5)
plt.plot(df['Date'], df['Weekly_Sales_MA'], label='10-Week Moving Avg', color='red')
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.title("Sales Trend with Moving Average")
plt.legend()
plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose

df.set_index('Date', inplace=True)  # Set Date as index
result = seasonal_decompose(df['Weekly_Sales'], model='additive', period=52)  # Weekly seasonality

result.plot()
plt.show()
df['Weekday'] = df['Date'].dt.dayofweek  # 0 = Monday, 6 = Sunday
df['Is_Weekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)  # 1 for Saturday/Sunday
df['Year'] = df['Date'].dt.year
df['Sales_YoY'] = df.groupby('Year')['Weekly_Sales'].pct_change()
df['Sales_Lag_1W'] = df['Weekly_Sales'].shift(1)  # Sales from last week
df['Sales_Lag_4W'] = df['Weekly_Sales'].shift(4)  # Sales from 4 weeks ago
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()
features = ['Fuel_Price', 'CPI', 'Unemployment', 'Is_Weekend', 'Sales_Lag_1W', 'Sales_Lag_4W']
target = 'Weekly_Sales'
from sklearn.model_selection import train_test_split

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred_rf))
from sklearn.metrics import r2_score

print("R2 Score (Linear Regression):", r2_score(y_test, y_pred))
print("R2 Score (Random Forest):", r2_score(y_test, y_pred_rf))
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Extract feature importances
feature_importances = rf_model.feature_importances_

# Create a DataFrame for visualization
feat_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
feat_importance_df = feat_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_importance_df, palette='viridis')
plt.title("Feature Importance in Sales Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Predict on test data
y_pred = rf_model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
from sklearn.model_selection import GridSearchCV

# Define hyperparameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearch
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Hyperparameters:", grid_search.best_params_)
best_rf = RandomForestRegressor(**grid_search.best_params_, random_state=42)
best_rf.fit(X_train, y_train)

# Evaluate again
y_pred_best = best_rf.predict(X_test)
print(f"New R² Score: {r2_score(y_test, y_pred_best):.2f}")
import joblib

# Save the trained model
joblib.dump(best_rf, "walmart_sales_model.pkl")

# Load the model (to test)
loaded_model = joblib.load("walmart_sales_model.pkl")
import joblib

# Save the trained model
joblib.dump(best_rf, "walmart_sales_model.pkl")

# Load the model (to test)
loaded_model = joblib.load("walmart_sales_model.pkl")
import matplotlib.pyplot as plt
import numpy as np

# Get feature importances from the trained model
feature_importances = rf_model.feature_importances_
feature_names = X_train.columns

# Sort features by importance
sorted_idx = np.argsort(feature_importances)

# Plot feature importance
plt.figure(figsize=(10,6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Sales Prediction")
plt.show()
# Print best parameters found
print("Best Hyperparameters:", grid_search.best_params_)

# Train a new RandomForestRegressor with the best parameters
best_rf = RandomForestRegressor(
    n_estimators=grid_search.best_params_['n_estimators'],
    max_depth=grid_search.best_params_['max_depth'],
    min_samples_split=grid_search.best_params_['min_samples_split'],
    min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
    random_state=42
)

# Fit the optimized model
best_rf.fit(X_train, y_train)

# Predict again with the optimized model
y_pred_best = best_rf.predict(X_test)

# Evaluate performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred_best)
mse = mean_squared_error(y_test, y_pred_best)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_best)

print(f"Optimized Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
import matplotlib.pyplot as plt  
import seaborn as sns  

# Group by store and sum weekly sales
store_sales = df.groupby("Store")["Weekly_Sales"].sum().reset_index()

# Sort stores by total sales
store_sales = store_sales.sort_values(by="Weekly_Sales", ascending=False)

# Plot the sales per store
plt.figure(figsize=(12,6))
sns.barplot(x="Store", y="Weekly_Sales", data=store_sales, palette="viridis")
plt.xlabel("Store ID")
plt.ylabel("Total Sales")
plt.title("Total Sales per Store")
plt.xticks(rotation=90)
plt.show()
# Convert Date column to datetime format (if not already done)
df['Date'] = pd.to_datetime(df['Date'])

# Extract month and year
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

# Group sales by month
monthly_sales = df.groupby("Month")["Weekly_Sales"].mean().reset_index()

# Plot sales trends by month
plt.figure(figsize=(12,6))
sns.lineplot(x="Month", y="Weekly_Sales", data=monthly_sales, marker="o")
plt.xlabel("Month")
plt.ylabel("Average Weekly Sales")
plt.title("Average Sales Per Month")
plt.xticks(range(1, 13))
plt.grid()
import pandas as pd  

# Convert Date column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Sort data by date
df = df.sort_values(by="Date")

# Set Date as index (important for time series models)
df.set_index("Date", inplace=True)

# Check the dataset
df.head()
from statsmodels.tsa.arima.model import ARIMA  
import matplotlib.pyplot as plt  

# Fit ARIMA model
model = ARIMA(df["Weekly_Sales"], order=(5,1,0))  # (p,d,q) values need tuning
model_fit = model.fit()

# Forecast next 12 weeks
forecast = model_fit.forecast(steps=12)

# Plot actual vs forecasted sales
plt.figure(figsize=(12,6))
plt.plot(df.index, df["Weekly_Sales"], label="Actual Sales", color="blue")
plt.plot(pd.date_range(df.index[-1], periods=12, freq="W"), forecast, label="Forecasted Sales", color="red")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.title("Sales Forecasting using ARIMA")
plt.legend()
plt.show()
import matplotlib.pyplot as plt  

# Resample weekly sales to monthly for trend analysis
df_monthly = df["Weekly_Sales"].resample("M").sum()

# Plot sales trend
plt.figure(figsize=(12,6))
plt.plot(df_monthly, marker="o", linestyle="-", color="blue")
plt.title("Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.grid()
plt.show()
import seaborn as sns  

# Compute correlation matrix
correlation_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Between Sales & Other Factors")
plt.show()

