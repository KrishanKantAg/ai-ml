# explore the impact of the learning rate alpha on gradient descent
# improve performance of gradient descent by feature scaling using z-score normalization
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.preprocessing import scale

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from lab_utils_common import dlc
from lab_utils_multi import (
    norm_plot,
    plot_cost_i_w,
    plt_equal_scale,
    run_gradient_descent,
)

np.set_printoptions(precision=2)

data_file_path = os.path.join(os.path.dirname(__file__), "../../data/house.txt")
data = np.loadtxt(data_file_path, delimiter=",", skiprows=1)
X_train = data[:, :-1]
X_features = ["size(sqft)", "bedrooms", "floors", "age"]
y_train = data[:, -1]

# ------------------------------------------------------------------------------------------------------
# Try to plot the target and seperate features to understand the feel of how different features affect the target
# ------------------------------------------------------------------------------------------------------
# fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
# for i in range(len(ax)):
#     ax[i].scatter(X_train[:, i], y_train)
#     ax[i].set_xlabel(X_features[i])
# ax[0].set_ylabel("Price (1000's)")
# plt.show()

## Seeing the plot we'll get the feel that floor and bedrooms doesn't affect the prices as much
## A clear pattern can be observed for age and size, when size 📈 price 📈, whilst age 📈 price 📉

# ------------------------------------------------------------------------------------------------------
# Now we'll try to run gradient descend and see how the cost graph looks wrt to itr numbers
# ------------------------------------------------------------------------------------------------------
# alpha = 9.9e-7
# alpha = 9e-7
# alpha = 1e-7
# _, _, hist = run_gradient_descent(X_train, y_train, 1000, alpha)
# plot_cost_i_w(X_train, y_train, hist)


# Finding the z-score FS
# A good alternative to this is scikit learn
# from sklearn.preprocessing import scale
# scale(X_orig, axis=0, with_mean=True, with_std=True, copy=True)
def z_score_fs(X_train):
    mu = np.mean(X_train, axis=0)
    # sigma = np.sqrt((np.sum((X_train - mu) ** 2, axis=0)) * (1 / m))
    sigma = np.std(X_train, axis=0)
    X_norm = (X_train - mu) / sigma
    return (X_norm, mu, sigma)


(X_scaled, mu, sigma) = z_score_fs(X_train)

X_scale_ski = scale(X_train, axis=0, with_mean=True, with_std=True, copy=True)

print("=" * 50)
print(f"Mean of the normal feature 0: {np.mean(X_train[:, 0])}")
print(f"Variance of the normal feature 0: {np.std(X_train[:, 0])}")

print("=" * 50)
print(
    f"Mean of the scaled feature 0 using z-score implemented: {np.mean(X_scaled[:, 0])}"
)
print(
    f"Variance of the scaled feature 0 using z-score implemented: {np.std(X_scaled[:, 0])}"
)
print("=" * 50)

print(f"Mean of the scaled feature 0 using scikit: {np.mean(X_scale_ski[:, 0])}")
print(f"Variance of the scaled feature 0 using scikit: {np.std(X_scale_ski[:, 0])}")
print("=" * 50)

# fig, ax = plt.subplots(1, 2, figsize=(12, 3))
# ax[0].scatter(X_train[:, 0], X_train[:, 3])
# ax[0].set_xlabel(X_features[0])
# ax[0].set_ylabel(X_features[3])
# ax[0].set_title("unnormalized")
# ax[0].axis("equal")

# ax[1].scatter(X_scaled[:, 0], X_scaled[:, 3])
# ax[1].set_xlabel(X_features[0])
# ax[0].set_ylabel(X_features[3])
# ax[1].set_title(r"Z-score normalized")
# ax[1].axis("equal")
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# fig.suptitle("distribution of features before, during, after normalization")
# plt.show()

print(f"X_mu = {mu}, \nX_sigma = {sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_scaled,axis=0)}")

# fig, ax = plt.subplots(1, 4, figsize=(12, 3))
# for i in range(len(ax)):
#     norm_plot(
#         ax[i],
#         X_train[:, i],
#     )
#     ax[i].set_xlabel(X_features[i])
# ax[0].set_ylabel("count")
# fig.suptitle("distribution of features before normalization")
# plt.show()
# fig, ax = plt.subplots(1, 4, figsize=(12, 3))
# for i in range(len(ax)):
#     norm_plot(
#         ax[i],
#         X_scaled[:, i],
#     )
#     ax[i].set_xlabel(X_features[i])
# ax[0].set_ylabel("count")
# fig.suptitle("distribution of features after normalization")

# plt.show()

alpha = 1.0e-1
w_norm, b_norm, hist = run_gradient_descent(X_scaled, y_train, 1000, alpha)
# plot_cost_i_w(X_train, y_train, hist)

# m = X_scaled.shape[0]
# yp = np.zeros(m)
# for i in range(m):
#     yp[i] = np.dot(X_scaled[i], w_norm) + b_norm

#     # plot predictions and targets versus original features
# fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
# for i in range(len(ax)):
#     ax[i].scatter(X_train[:, i], y_train, label="target")
#     ax[i].set_xlabel(X_features[i])
#     ax[i].scatter(X_train[:, i], yp, color=dlc["dlorange"], label="predict")
# ax[0].set_ylabel("Price")
# ax[0].legend()
# fig.suptitle("target versus prediction using z-score normalized model")
# plt.show()

plt_equal_scale(X_train, X_scaled, y_train)
