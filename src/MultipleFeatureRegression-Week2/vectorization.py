import numpy as np
import time

# a = np.array([1, 2])
# b = np.array([1, 2, 3])

# d = np.random.random_sample(4)


# print(d, d.dtype, d.shape)


# # NumPy routines which allocate memory and fill arrays with value
# # Here the arugment is not positional, but rather the shape
# a = np.zeros(4)
# print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# a = np.zeros((4,))
# print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# a = np.random.random_sample(4)
# print(
#     f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}"
# )

# # NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
# # Here the arugments cannot be tuple/shape, instead inferred as positional args
# a = np.arange(2.0)
# print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# a = np.random.rand(4, 2)
# print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# # NumPy routines which allocate memory and fill with user specified values
# a = np.array([5, 4, 3, 2])
# print(
#     f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}"
# )
# a = np.array([5.0, 4, 3, 2])
# print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# # INDEXING
# # vector indexing operations on 1-D vectors
# a = np.arange(10)
# print(a)

# # access an element
# print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

# # access the last element, negative indexes count from the end
# print(f"a[-1] = {a[-1]}")

# # indexs must be within the range of the vector or they will produce and error
# try:
#     c = a[10]
# except Exception as e:
#     print("The error message you'll see is:")
#     print(e)

# # Slicing creates an array of indices using a set of three values (start:stop:step).
# # A subset of values is also valid. Its use is best explained by example
# # vector slicing operations
# a = np.arange(10)
# print(f"a         = {a}")

# # access 5 consecutive elements (start:stop:step)
# c = a[2:7:1]
# print("a[2:7:1] = ", c)

# # access 3 elements separated by two
# c = a[2:7:2]
# print("a[2:7:2] = ", c)

# # access all elements index 3 and above
# c = a[3:]
# print("a[3:]    = ", c)

# # access all elements below index 3
# c = a[:3]
# print("a[:3]    = ", c)

# # access all elements
# c = a[:]
# print("a[:]     = ", c)


# def my_dot(a, b):
#     """
#     Compute the dot product of two vectors

#      Args:
#        a (ndarray (n,)):  input vector
#        b (ndarray (n,)):  input vector with same dimension as a

#      Returns:
#        x (scalar):
#     """
#     x = 0
#     for i in range(a.shape[0]):
#         x = x + a[i] * b[i]
#     return x


# # Comparsion of vector vs loop
# np.random.seed(1)
# a = np.random.rand(100_000_000)
# b = np.random.rand(100_000_000)

# tic = time.time()  # capture start time
# c = np.dot(a, b)
# toc = time.time()  # capture end time

# print(f"np.dot(a, b) =  {c:.4f}")
# print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

# tic = time.time()  # capture start time
# c = my_dot(a, b)
# toc = time.time()  # capture end time

# print(f"my_dot(a, b) =  {c:.4f}")
# print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

# del a
# del b
