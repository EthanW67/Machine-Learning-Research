import os
import shutil
import random

# Path to the dataset folder
dataset_path = r'C:\Users\ethan.DESKTOP-TL348AV\OneDrive\Desktop\Pytorch\SURF Research\SURF Week5_Meeting 4\data\celeba_hq_256'
train_path = r'C:\Users\ethan.DESKTOP-TL348AV\OneDrive\Desktop\Pytorch\SURF Research\SURF Week5_Meeting 4\data\training_dataset'
test_path = r'C:\Users\ethan.DESKTOP-TL348AV\OneDrive\Desktop\Pytorch\SURF Research\SURF Week5_Meeting 4\data\testing_dataset'

# Create train and test directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# List all files in the dataset directory
all_files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]

# Shuffle the files
random.shuffle(all_files)

# Define the split ratio (e.g., 80% train, 20% test)
split_ratio = 0.8
split_index = int(len(all_files) * split_ratio)

# Split the files into train and test sets
train_files = all_files[:split_index]
test_files = all_files[split_index:]

# Move the files to the respective directories
for f in train_files:
    shutil.move(os.path.join(dataset_path, f), os.path.join(train_path, f))

for f in test_files:
    shutil.move(os.path.join(dataset_path, f), os.path.join(test_path, f))

print("Dataset split complete.")
print(f"Train set: {len(train_files)} images")
print(f"Test set: {len(test_files)} images")


import os
import cv2
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



# Custom Dataset class
class ImagePairDataset(Dataset):
    def __init__(self, npy_file):
        data = np.load(npy_file, allow_pickle=True).item()
        self.low_res = data['low_res']
        self.high_res = data['high_res']
        self.transforms = transforms

    def __len__(self):
        return len(self.low_res)

    def __getitem__(self, idx):
        low_res_img = self.low_res[idx]
        high_res_img = self.high_res[idx]

        low_res_img = torch.from_numpy(low_res_img).float() / 255.0
        high_res_img = torch.from_numpy(high_res_img).float() / 255.0

        low_res_img = low_res_img.permute(2, 0, 1)
        high_res_img = high_res_img.permute(2, 0, 1)

        if self.transforms:
            low_res_img = self.transforms(low_res_img)
            high_res_img = self.transforms(high_res_img)

        return low_res_img, high_res_img

transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Paths to the .npy files
train_data_path = r'C:\Users\ethan.DESKTOP-TL348AV\PycharmProjects\resized_data\train_pairs.npy'
test_data_path = r'C:\Users\ethan.DESKTOP-TL348AV\PycharmProjects\resized_data\test_pairs.npy'


# Create datasets and dataloaders
train_dataset = ImagePairDataset(train_data_path, transform = transforms)
test_dataset = ImagePairDataset(test_data_path, transform = transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Example usage of the dataloaders
for low_res, high_res in train_loader:
    print("Train batch - Low Res:", low_res.shape, "High Res:", high_res.shape)
    break

for low_res, high_res in test_loader:
    print("Test batch - Low Res:", low_res.shape, "High Res:", high_res.shape)
    break

