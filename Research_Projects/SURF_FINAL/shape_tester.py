import numpy as np

# Path to the .npy file
file_path = '/home/ys92/dataset/ctslice_train.npy'

# Load the .npy file
ct_test_data = np.load(file_path, allow_pickle=True)

# Print the shape and size of the data
print(f"Shape of the CT test data: {ct_test_data.shape}")
print(f"Size of the CT test data: {ct_test_data.size}")

# Additional information
print(f"Number of dimensions: {ct_test_data.ndim}")
print(f"Data type: {ct_test_data.dtype}")

# If the data is too large to print directly, you can inspect the shape of the first element
print(f"Shape of the first element: {ct_test_data[0].shape}")

