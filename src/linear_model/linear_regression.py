import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import joblib

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the preprocessed data directory relative to the script's location
data_dir = os.path.join(script_dir, 'data')

# Define the path to the models directory relative to the parent directory of the script's location
models_dir = os.path.join(os.path.dirname(script_dir), 'models')

# Create the models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# Load the preprocessed data
X_train = np.load(f'{data_dir}/X_train_final.npy')
X_val = np.load(f'{data_dir}/X_val_final.npy')
X_test = np.load(f'{data_dir}/X_test_final.npy')

y_train_log = np.load(f'{data_dir}/y_train_log.npy')  # Log-transformed version!
y_val_log = np.load(f'{data_dir}/y_val_log.npy')
y_test_log = np.load(f'{data_dir}/y_test_log.npy')

# Check for negative or zero values in y_train_log, y_val_log, y_test_log
if (y_train_log <= 0).any() or (y_val_log <= 0).any() or (y_test_log <= 0).any():
    print("Error: Log-transformed target values contain non-positive numbers.")
else:
    print("Log transformation is correct.")

# Combine validation and test sets
X_combined_test = np.concatenate((X_val, X_test), axis=0)
y_combined_test_log = np.concatenate((y_val_log, y_test_log), axis=0)

# Initialize the Linear Regression model
linear_model = LinearRegression()

# Fit the linear regression model on the training data
linear_model.fit(X_train, y_train_log)

# Save the trained linear regression model to a file in the models directory
linear_model_filename = os.path.join(models_dir, 'linear_regression_model.joblib')
joblib.dump(linear_model, linear_model_filename)
print(f'Linear Regression Model saved to {linear_model_filename}')

# Make predictions on the combined test set (log-transformed) using linear regression
y_combined_test_pred_log_linear = linear_model.predict(X_combined_test)

# Convert log-transformed predictions to original scale for linear regression
y_combined_test_pred_linear = np.expm1(y_combined_test_pred_log_linear)

# Calculate performance metrics for combined test set in original scale for linear regression
mse_combined_test_linear = mean_squared_error(np.expm1(y_combined_test_log), y_combined_test_pred_linear)
r2_combined_test_linear = r2_score(np.expm1(y_combined_test_log), y_combined_test_pred_linear)

print(f'Linear Regression - Combined Test MSE (original scale): {mse_combined_test_linear}')
print(f'Linear Regression - Combined Test R^2 (original scale): {r2_combined_test_linear}')

# Make predictions on the training set (log-transformed) using linear regression
y_train_pred_log_linear = linear_model.predict(X_train)

# Convert log-transformed predictions to original scale for linear regression
y_train_pred_linear = np.expm1(y_train_pred_log_linear)

# Calculate performance metrics for training set in original scale for linear regression
mse_train_linear = mean_squared_error(np.expm1(y_train_log), y_train_pred_linear)
r2_train_linear = r2_score(np.expm1(y_train_log), y_train_pred_linear)

print(f'Linear Regression - Training MSE (original scale): {mse_train_linear}')
print(f'Linear Regression - Training R^2 (original scale): {r2_train_linear}')

# Initialize the Ridge Regression model
ridge_model = Ridge(alpha=1.0)

# Fit the ridge regression model on the training data
ridge_model.fit(X_train, y_train_log)

# Save the trained ridge regression model to a file in the models directory
ridge_model_filename = os.path.join(models_dir, 'ridge_regression_model.joblib')
joblib.dump(ridge_model, ridge_model_filename)
print(f'Ridge Regression Model saved to {ridge_model_filename}')

# Make predictions on the combined test set (log-transformed) using ridge regression
y_combined_test_pred_log_ridge = ridge_model.predict(X_combined_test)

# Convert log-transformed predictions to original scale for ridge regression
y_combined_test_pred_ridge = np.expm1(y_combined_test_pred_log_ridge)

# Calculate performance metrics for combined test set in original scale for ridge regression
mse_combined_test_ridge = mean_squared_error(np.expm1(y_combined_test_log), y_combined_test_pred_ridge)
r2_combined_test_ridge = r2_score(np.expm1(y_combined_test_log), y_combined_test_pred_ridge)

print(f'Ridge Regression - Combined Test MSE (original scale): {mse_combined_test_ridge}')
print(f'Ridge Regression - Combined Test R^2 (original scale): {r2_combined_test_ridge}')

# Make predictions on the training set (log-transformed) using ridge regression
y_train_pred_log_ridge = ridge_model.predict(X_train)

# Convert log-transformed predictions to original scale for ridge regression
y_train_pred_ridge = np.expm1(y_train_pred_log_ridge)

# Calculate performance metrics for training set in original scale for ridge regression
mse_train_ridge = mean_squared_error(np.expm1(y_train_log), y_train_pred_ridge)
r2_train_ridge = r2_score(np.expm1(y_train_log), y_train_pred_ridge)

print(f'Ridge Regression - Training MSE (original scale): {mse_train_ridge}')
print(f'Ridge Regression - Training R^2 (original scale): {r2_train_ridge}')

# Save predictions and metrics to files
np.save(f'{data_dir}/y_combined_test_pred_linear.npy', y_combined_test_pred_linear)
np.save(f'{data_dir}/y_train_pred_linear.npy', y_train_pred_linear)

np.save(f'{data_dir}/y_combined_test_pred_ridge.npy', y_combined_test_pred_ridge)
np.save(f'{data_dir}/y_train_pred_ridge.npy', y_train_pred_ridge)

metrics = {
    'linear_regression': {
        'combined_test_mse_original': mse_combined_test_linear,
        'combined_test_r2_original': r2_combined_test_linear,
        'training_mse_original': mse_train_linear,
        'training_r2_original': r2_train_linear
    },
    'ridge_regression': {
        'combined_test_mse_original': mse_combined_test_ridge,
        'combined_test_r2_original': r2_combined_test_ridge,
        'training_mse_original': mse_train_ridge,
        'training_r2_original': r2_train_ridge
    }
}

with open(f'{data_dir}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Convert log-transformed target variables to original scale for plotting
y_train_raw = np.expm1(y_train_log)
y_val_raw = np.expm1(y_val_log)
y_test_raw = np.expm1(y_test_log)
y_combined_test_raw = np.expm1(y_combined_test_log)

# Visualize distributions and check for outliers
plt.figure(figsize=(24, 6))

plt.subplot(1, 3, 1)
sns.boxplot(x=y_train_raw)
plt.title('Box Plot of Training Target Values (Original Scale)')
plt.xlabel('Sales Price')

plt.subplot(1, 3, 2)
sns.boxplot(x=y_val_raw)
plt.title('Box Plot of Validation Target Values (Original Scale)')
plt.xlabel('Sales Price')

plt.subplot(1, 3, 3)
sns.boxplot(x=y_test_raw)
plt.title('Box Plot of Test Target Values (Original Scale)')
plt.xlabel('Sales Price')

plt.tight_layout()
plt.show()

# Visualize actual vs predicted values for the combined test set using linear regression
plt.figure(figsize=(10, 6))
plt.scatter(np.expm1(y_combined_test_log), y_combined_test_pred_linear, alpha=0.5, label='Linear Regression')
plt.plot([np.expm1(y_combined_test_log.min()), np.expm1(y_combined_test_log.max())], 
         [np.expm1(y_combined_test_log.min()), np.expm1(y_combined_test_log.max())], 'k--', lw=2, label='Perfect Prediction Line')

# Set logarithmic scale for both axes
plt.xscale('log')
plt.yscale('log')

plt.xlabel('Actual Sales Price (Log Scale)')
plt.ylabel('Predicted Sales Price (Original Scale)')
plt.title('Actual vs Predicted Sales Prices (Combined Test Set) - Linear Regression')
plt.legend()
plt.grid(True)

# Save the plot to a file
plot_file = os.path.join(script_dir, 'linear_combined_test_set_plot.png')
plt.savefig(plot_file)
print(f'Plot saved to {plot_file}')

# Visualize actual vs predicted values for the training set using linear regression
plt.figure(figsize=(10, 6))
plt.scatter(np.expm1(y_train_log), y_train_pred_linear, alpha=0.5, label='Linear Regression')
plt.plot([np.expm1(y_train_log.min()), np.expm1(y_train_log.max())], 
         [np.expm1(y_train_log.min()), np.expm1(y_train_log.max())], 'k--', lw=2, label='Perfect Prediction Line')

# Set logarithmic scale for both axes
plt.xscale('log')
plt.yscale('log')

plt.xlabel('Actual Sales Price (Log Scale)')
plt.ylabel('Predicted Sales Price (Original Scale)')
plt.title('Actual vs Predicted Sales Prices (Training Set) - Linear Regression')
plt.legend()
plt.grid(True)

# Save the plot to a file
plot_file = os.path.join(script_dir, 'linear_training_set_plot.png')
plt.savefig(plot_file)
print(f'Plot saved to {plot_file}')

# Visualize actual vs predicted values for the combined test set using ridge regression
plt.figure(figsize=(10, 6))
plt.scatter(np.expm1(y_combined_test_log), y_combined_test_pred_ridge, alpha=0.5, label='Ridge Regression')
plt.plot([np.expm1(y_combined_test_log.min()), np.expm1(y_combined_test_log.max())], 
         [np.expm1(y_combined_test_log.min()), np.expm1(y_combined_test_log.max())], 'k--', lw=2, label='Perfect Prediction Line')

# Set logarithmic scale for both axes
plt.xscale('log')
plt.yscale('log')

plt.xlabel('Actual Sales Price (Log Scale)')
plt.ylabel('Predicted Sales Price (Original Scale)')
plt.title('Actual vs Predicted Sales Prices (Combined Test Set) - Ridge Regression')
plt.legend()
plt.grid(True)

# Save the plot to a file
plot_file = os.path.join(script_dir, 'ridge_combined_test_set_plot.png')
plt.savefig(plot_file)
print(f'Plot saved to {plot_file}')

# Visualize actual vs predicted values for the training set using ridge regression
plt.figure(figsize=(10, 6))
plt.scatter(np.expm1(y_train_log), y_train_pred_ridge, alpha=0.5, label='Ridge Regression')
plt.plot([np.expm1(y_train_log.min()), np.expm1(y_train_log.max())], 
         [np.expm1(y_train_log.min()), np.expm1(y_train_log.max())], 'k--', lw=2, label='Perfect Prediction Line')

# Set logarithmic scale for both axes
plt.xscale('log')
plt.yscale('log')

plt.xlabel('Actual Sales Price (Log Scale)')
plt.ylabel('Predicted Sales Price (Original Scale)')
plt.title('Actual vs Predicted Sales Prices (Training Set) - Ridge Regression')
plt.legend()
plt.grid(True)

# Save the plot to a file
plot_file = os.path.join(script_dir, 'ridge_training_set_plot.png')
plt.savefig(plot_file)
print(f'Plot saved to {plot_file}')
