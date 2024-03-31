import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

data = {
    'x': [0.71176706, 0.648424558, 0.712163761, 0.627647798],
    'y': [-0.22493821, -0.150333813, -0.037779757, 0.026202763],
    'X': [0.684702188, 0.629909754, 0.693434587, 0.615052886]
}

# Create a DataFrame from your data
df = pd.DataFrame(data)
# Function to plot data and the regression model in 3D
def plot_3d_data_and_model(x_vals, y_vals, X_vals, model, angle):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for actual data points
    ax.scatter(x_vals, y_vals, X_vals, color='red', label='Actual Data')

    # Creating a meshgrid for the surface plot
    x_surf = np.linspace(min(x_vals), max(x_vals), 100)
    y_surf = np.linspace(min(y_vals), max(y_vals), 100)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)

    # Predicting X values across the grid
    X_surf = model.predict(np.array([x_surf.ravel(), y_surf.ravel()]).T).reshape(x_surf.shape)

    # Plotting the surface
    ax.plot_surface(x_surf, y_surf, X_surf, color='blue', alpha=0.3, edgecolor='none')

    # Set labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('X')

    # Set title
    ax.set_title('3D plot of the linear regression model')

    # Setting the angle of the view
    ax.view_init(elev=10., azim=angle)

    # Show plot
    plt.show()

# # Assuming 'model' is already trained LinearRegression model and 'df' is your DataFrame
# Create a Linear Regression model
model = LinearRegression()

# Fit the model with the data - 'x' and 'y' as features, 'X' as the target variable
model.fit(df[['x', 'y']], df['X'])

# Get the coefficients for the predictors (features)
coefficients = model.coef_

# Get the intercept of the model
intercept = model.intercept_

# Assuming 'x' and 'y' are the names of your features, print the formula
print(f"X = {coefficients[0]:.4f}*x + {coefficients[1]:.4f}*y + {intercept:.4f}")

# Predict the 'X' values using the model
X_predicted = model.predict(df[['x', 'y']])

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(df['X'], X_predicted)

# Print the errors
print(f"Mean Squared Error (MSE): {mse}")

for angle in range(0, 360, 60):
    plot_3d_data_and_model(df['x'], df['y'], df['X'], model, angle)
