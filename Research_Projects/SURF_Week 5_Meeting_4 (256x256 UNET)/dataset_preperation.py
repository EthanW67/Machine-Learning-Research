import os
import cv2
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Function to generate low-resolution and high-resolution image pairs
def generate_image_pairs(dataset_path, low_res_size, high_res_size):
    low_res_images = []
    high_res_images = []
    resized_low_res_images = []

    for filename in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, filename)
        if os.path.isfile(img_path):
            high_res = cv2.imread(img_path)
            high_res = cv2.resize(high_res, high_res_size)
            low_res = cv2.resize(high_res, low_res_size)
            resized_low_res = cv2.resize(low_res, high_res_size)

            high_res_images.append(high_res)
            low_res_images.append(low_res)
            resized_low_res_images.append(resized_low_res)

    low_res_images = np.array(low_res_images)
    high_res_images = np.array(high_res_images)
    resized_low_res_images = np.array(resized_low_res_images)
    return low_res_images, high_res_images, resized_low_res_images


# Paths to the dataset folders
train_path = r'C:\Users\ethan.DESKTOP-TL348AV\OneDrive\Desktop\Pytorch\SURF Research\SURF Week5_Meeting 4\data\training_dataset'
test_path = r'C:\Users\ethan.DESKTOP-TL348AV\OneDrive\Desktop\Pytorch\SURF Research\SURF Week5_Meeting 4\data\testing_dataset'

# Output paths for the .npy files
train_output_path = r'C:\Users\ethan.DESKTOP-TL348AV\PycharmProjects\resized_data\train_pairs.npy'
test_output_path = r'C:\Users\ethan.DESKTOP-TL348AV\PycharmProjects\resized_data\test_pairs.npy'

# Parameters for resizing
low_res_size = (64, 64)
high_res_size = (256, 256)

# Ensure the directory exists
os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
os.makedirs(os.path.dirname(test_output_path), exist_ok=True)

# Generate and save train pairs
train_low_res, train_high_res, train_resized_low_res = generate_image_pairs(train_path, low_res_size, high_res_size)
with open(train_output_path, 'wb') as f:
    pickle.dump({'low_res': train_low_res, 'high_res': train_high_res, 'resized_low_res': train_resized_low_res}, f, protocol=4)

# Generate and save test pairs
test_low_res, test_high_res, test_resized_low_res = generate_image_pairs(test_path, low_res_size, high_res_size)
with open(test_output_path, 'wb') as f:
    pickle.dump({'low_res': test_low_res, 'high_res': test_high_res, 'resized_low_res': test_resized_low_res}, f, protocol=4)

"""
# Generate and save train pairs
train_low_res, train_high_res = generate_image_pairs(train_path, low_res_size, high_res_size)
np.save(train_output_path, {'low_res': train_low_res, 'high_res': train_high_res})

# Generate and save test pairs
test_low_res, test_high_res = generate_image_pairs(test_path, low_res_size, high_res_size)
np.save(test_output_path, {'low_res': test_low_res, 'high_res': test_high_res})
"""
print("Low-resolution and high-resolution image pairs saved successfully.")



