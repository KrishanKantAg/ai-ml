import copy, math
import numpy as np
import matplotlib.pyplot as plt
import os
import time


plt.style.use(os.path.join(os.path.dirname(__file__), "../../deeplearning.mplstyle"))
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

# some random big number
crazy_number = 10000
crazy_feature = 10000

# House pricing ex with multiple features - price, room, floors, age
# X_train = np.random.rand(crazy_number, crazy_feature)
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])

# House price
y_train = np.random.rand(crazy_number)
y_train = np.array([460, 232, 178])

# --------------------------------------
# Linear Regression Model f_wb = w.x + b
# --------------------------------------

m, n = X_train.shape

# w initial value:
# w_init = np.random.rand(n)
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
b_init = 785.1811367994083

f_wb = np.dot(X_train, w_init) + b_init


# --------------------------------------
# Cost function (Squared error) J_wb = (1/2m) * (sum_1_to_m((f_wb_i - y_i)**2))
# --------------------------------------
def compute_cost(f_wb, y_train):
    error = (f_wb - y_train) ** 2
    return (1 / (2 * m)) * (np.sum(error))


# J_wb = compute_cost(f_wb, y_train)

# print("#" * 50)
# print("Cost Fn")
# print(J_wb)
# print("#" * 50)

# --------------------------------------
# Gradients dJ_dw, and dJ_db
# --------------------------------------


def compute_gradient(f_wb, y, X):
    tic = time.time()
    error = f_wb - y
    error = error.reshape(m, -1)
    dJ_dw = (1 / m) * (np.sum(error * X, axis=0))
    dJ_db = (1 / m) * (np.sum(error))
    toc = time.time()
    # print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")
    return dJ_dw, dJ_db


# dJ_dw, dJ_db = compute_gradient(f_wb, y_train, X_train)


def run_gradient_descend(itr, X_train, y_train, w_init, b_init, alpha):
    w = w_init
    b = b_init
    J_history = []
    for i in range(itr):
        f_wb = np.dot(X_train, w) + b
        cost_at_i_itr = compute_cost(f_wb, y_train)
        if i < 100_000:
            J_history.append(cost_at_i_itr)
        if i % math.ceil(itr / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        dJ_dw, dJ_db = compute_gradient(f_wb, y_train, X_train)
        w -= (alpha) * dJ_dw
        b -= (alpha) * dJ_db
    return w, b, J_history


# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.0
# some gradient descent settings
iterations = 1_000
alpha = 5.0e-7
# run gradient descent
w_final, b_final, J_history = run_gradient_descend(
    iterations, X_train, y_train, initial_w, initial_b, alpha
)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

for i in range(m):
    print(
        f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}"
    )
