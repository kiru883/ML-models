from LinearRegression import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


# linear function
X = 2 * np.random.rand(100 , 1)
y = 4 + 3 * X + np.random.randn(100 , 1)

# linear regression gradient descent and analytical solve
lr_sgd = LinearRegression(optimizer="SGD")
lr_orig = LinearRegression(optimizer="ORIG")

lr_sgd.fit(X.reshape(-1, 1), y.reshape(-1, 1))
pred_y_lr_sgd = lr_sgd.predict(X.reshape(-1, 1))
lr_orig.fit(X.reshape(-1, 1), y.reshape(-1, 1))
pred_y_lr_orig = lr_orig.predict(X.reshape(-1, 1))

# plots
_, ax = plt.subplots(figsize=(16, 12))
ax.plot(X, y, 'o')
ax.plot(X, pred_y_lr_sgd, '-', color='red', label='LinearRegression SGD')
ax.plot(X, pred_y_lr_orig, '-', color='blue', label='LinearRegression ORIG')
ax.legend()
plt.show()
