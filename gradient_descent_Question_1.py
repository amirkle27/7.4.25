import numpy as np
import matplotlib.pyplot as plt

def gradient_descent_investment_vs_growth(advertising_investment, sales_growth, learning_rate, iterations = 1000):
    m = 0
    b = 0
    n = len(advertising_investment)
    cost_history = []
    for i in range(iterations):
        y_pred = m * advertising_investment + b

        cost = (1 / (2 * n)) * np.sum((y_pred - sales_growth) ** 2)
        cost_history.append(cost)

        m_gradient = (1 / n) * np.sum(advertising_investment * (y_pred - sales_growth))
        b_gradient = (1 / n) * np.sum(y_pred - sales_growth)

        m = m - learning_rate * m_gradient
        b = b - learning_rate * b_gradient
        print(f'm = {m}, b = {b}')

        if i % 100 == 0:
            print(f"Iteration {i}: m = {m:.4f}, b = {b:.4f}, cost = {cost:.4f}")
    print(f"\nFinal parameters: m = {m:.2f}, b = {b:.2f}")
    print(f"Line equation: y = {m:.2f}x + {b:.2f}")
    return cost_history, m, b

learning_rate_000001 = 0.000001
learning_rate_000005 = 0.000005
learning_rate_0001 = 0.0001



#####1#####

print("\nQuestion 1.1:\n")

advertising_investment = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50])
sales_growth = np.array([25, 30, 40, 45, 50, 60, 65, 70, 80])
cost_history, m1, b1 = gradient_descent_investment_vs_growth(advertising_investment, sales_growth, learning_rate = 0.001)

plt.figure(figsize=(10, 6))
plt.scatter(advertising_investment, sales_growth, color='blue', label='Data points')
x_line = np.linspace(0, 70, 100)
y_line = m1 * x_line + b1
plt.plot(x_line, y_line, color='red', label='Regression line')
plt.title('Question 1.1 - Linear Regression - Advertising Investment vs. Sales Growth')
plt.xlabel('Advertising Investment (k$)')
plt.ylabel('Sales Growth (k$)')
plt.legend()
plt.grid(True)
plt.text(-2, 101, f'y = {m1:.2f}x + {b1:.2f}', fontsize=12)
plt.show()

#####2#####

print("\nQuestion 1.2:\n")

investment = [60]
predicted_growth = m1 * investment[0] + b1

print(f'If the company invests a sum of {investment[0]}k$ in advertising, the expected growth in sales is {predicted_growth:.2f}k$')

plt.figure(figsize=(10, 6))
plt.scatter(advertising_investment, sales_growth, color='blue', label='Data points')
plt.scatter(investment, predicted_growth, color = 'green', label = f'Sales Growth Prediction for {investment[0]}k Investment - {predicted_growth:.2f}k')
x_line = np.linspace(0, 70, 100)
y_line = m1 * x_line + b1
plt.plot(x_line, y_line, color='red', label='Regression line')
plt.title('Question 1.2 - Sales Growth Prediction for a 60k$ Investment')
plt.xlabel('Advertising Investment (k$)')
plt.ylabel('Sales Growth (k$)')
plt.legend()
plt.grid(True)
plt.show()
######3#####

print("\nQuestion 1.3:\n")

desired_growth = 100
investment_needed = (desired_growth - b1) / m1

print(f"To get a growth of 100k in sales, approximately {investment_needed:.2f}k$ of advertising investment are needed")

plt.figure(figsize=(10, 6))
plt.scatter(advertising_investment, sales_growth, color='blue', label='Data points')
plt.scatter(investment_needed, desired_growth, color = 'green', label = f"Investment of {investment_needed:.2f}k needed to gain a {desired_growth:.2f}k growth in Sales")



x_line = np.linspace(0, 70, 100)
y_line = m1 * x_line + b1
plt.plot(x_line, y_line, color='red', label='Regression line')
plt.title('Question 1.3 - Investment needed for a 100k$ Sales Growth')
plt.xlabel('Advertising Investment (k$)')
plt.ylabel('Sales Growth (k$)')
plt.legend()
plt.grid(True)
plt.show()
######4#####

print("\nQuestion 1.4:\n")

cost_000001,_,_ = gradient_descent_investment_vs_growth(advertising_investment, sales_growth, learning_rate_000001)
cost_000005,_,_ = gradient_descent_investment_vs_growth(advertising_investment,sales_growth,learning_rate_000005)
cost_0001,_,_ = gradient_descent_investment_vs_growth(advertising_investment,sales_growth,learning_rate_0001)

plt.figure(figsize=(10,6))
plt.plot(cost_000001, label='Learning rate = 0.000001')
plt.plot(cost_000005, label='Learning rate = 0.000005')
plt.plot(cost_0001, label='Learning rate = 0.0001')
plt.title('Question 1.4 - Comparison of Learning Rates - Cost vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)
plt.show()
