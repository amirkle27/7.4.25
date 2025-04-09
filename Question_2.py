import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([
    [2, 15, 40, 0],
    [5, 16, 45, 1],
    [3, 16, 40, 0],
    [10, 18, 50, 5],
    [7, 17, 45, 3],
    [1, 14, 35, 0],
    [8, 16, 45, 4],
    [4, 15, 40, 1],
    [6, 15, 42, 2],
    [12, 19, 55, 8]
])
y = np.array([15, 25, 18, 45, 35, 12, 38, 22, 30, 60])
columns = ['Experience','Education','WeeklyHours','ManagementExperience']
model = LinearRegression()
model.fit(X, y)

#####1#####

print("\nQuestion 2.1:\n")
print(f"Intercept (β₀): {model.intercept_:.2f}")
print(f"Coefficients (β₁, β₂, β₃, β₄): {model.coef_}")

#####2#####

print("\nQuestion 2.2:\n")

print(f"Prediction equation:")
print(f"Salary = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Experience + {model.coef_[1]:.2f} * Education + {model.coef_[2]:.2f} * WeeklyHours + {model.coef_[3]:.2f} * ManagementExperience")



print(f"Intercept (β₀): {model.intercept_:.2f}")
for i, col in enumerate(X.columns):
    print(f"Coefficient for {col} (β{i+1}): {model.coef_[i]:.2f}")
