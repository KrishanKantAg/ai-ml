#!/usr/bin/env python3
"""
Test script to verify AI/ML environment setup
Run this to check if all packages are working correctly
"""

# import sys
# print(f"Python version: {sys.version}")

# # Test core packages
# try:
#     import numpy as np
#     print(f"NumPy version: {np.__version__}")
    
#     import pandas as pd
#     print(f"Pandas version: {pd.__version__}")
    
#     print("✅ All packages imported successfully!")
    
#     # Simple test
#     data = np.array([1, 2, 3, 4, 5])
#     print(f"Sample data: {data}")
#     print(f"Mean: {np.mean(data)}")
    
# except ImportError as e:
#     print(f"❌ Error importing package: {e}")
import sys
import os
# Add src directory to Python path
# The __file__ gives the path of the current file
# The os.path.dirname(__file__) gives the folder where this file's present
# Instead of directly calculating the path, we prefer using os.path.join() to ensure cross-platform compatibility, for windows, Linux and macOS
# sys.path is a list of directories that the interpreter will search for modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from lab_utils_common import dlcolors

n_bin = 10
dlcm = LinearSegmentedColormap.from_list('dl_map', dlcolors, N=n_bin)

# Create gradient data
gradient = np.linspace(0, 1, 256).reshape(1, -1)

# # Plot
plt.figure(figsize=(6, 1))
plt.imshow(gradient, aspect="auto", cmap=dlcm)
plt.axis("off")
plt.show()

print(dlcm)