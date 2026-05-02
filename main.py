import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Ice Cream.csv')
print(df.head())
m = df.shape[0]
print(m)
print(df.isnull().sum())

# Visualize the data
plt.scatter(df['Temperature'], df['Revenue'])
plt.xlabel('Temperature (°C)')
plt.ylabel('Ice Cream Sales')
plt.title('Temperature vs Ice Cream Sales')
plt.show()

# Training the model
x = df['Temperature'].values.reshape(-1, 1)
y = df['Revenue'].values

def predict(x, w, b):
    return w * x + b

# Initialize parameters
w = 0 
b = 0
alpha = 0.001
iterations = 1000

# Cost Function 
def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        total_cost = total_cost + cost 
    total_cost = total_cost / (2 * m)
    return total_cost

# Gradient Descent
def gradient_descent(x, y, w, b, alpha, iterations):
    m = x.shape[0]
    cost_history = []
    for i in range(iterations):
        dw = 0
        db = 0 
        for j in range(m):
            f_wb = w * x[j] + b
            dw = dw + (f_wb - y[j]) * x[j]
            db = db + (f_wb - y[j])
            cost_history.append(compute_cost(x, y, w, b))
        dw = dw / m
        db = db / m
        w = w - alpha * dw
        b = b - alpha * db
        #cost_history.append(compute_cost(x, y, w, b))
    return w, b, cost_history

w, b, cost_history = gradient_descent(x, y, w, b, alpha, iterations)
print(f"Optimized parameters: w = {w}, b = {b}")

w = float(w)
b = float(b)

df_sorted = df.sort_values('Temperature')

x = df_sorted['Temperature'].values
y = df_sorted['Revenue'].values

y_pred = w * x + b

plt.scatter(x, y, color='blue')
plt.plot(x, y_pred, color='red')
plt.xlabel('Temperature (°C)')
plt.ylabel('Ice Cream Sales')
plt.title('Temperature vs Ice Cream Sales')
plt.show()

plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Learning Progress")
plt.show()
