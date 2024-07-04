import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Create Dataset
#------------------------------------------------------------------------------------------------------------
# Custom Dataset class
class ImagePairDataset(Dataset):
    def __init__(self, npy_file, transforms=None):
        data = np.load(npy_file, allow_pickle=True)
        self.low_res = data['low_res']
        self.high_res = data['high_res']
        self.resized_low_res = data['resized_low_res']
        self.transforms = transforms

    def __len__(self):
        return len(self.low_res)

    def __getitem__(self, idx):
        low_res_img = self.low_res[idx]
        high_res_img = self.high_res[idx]
        resized_low_res_img = self.resized_low_res[idx]

        # Convert images to RGB if they are in BGR format
        if low_res_img.shape[2] == 3:  # Check if the image has 3 channels
            low_res_img = cv2.cvtColor(low_res_img, cv2.COLOR_BGR2RGB)
        if high_res_img.shape[2] == 3:  # Check if the image has 3 channels
            high_res_img = cv2.cvtColor(high_res_img, cv2.COLOR_BGR2RGB)
        if resized_low_res_img.shape[2] == 3:  # Check if the image has 3 channels
            resized_low_res_img = cv2.cvtColor(resized_low_res_img, cv2.COLOR_BGR2RGB)

        # Ensure images are 3-channel (RGB)
        if len(low_res_img.shape) == 2:  # If the image is grayscale
            low_res_img = cv2.cvtColor(low_res_img, cv2.COLOR_GRAY2RGB)
        if len(high_res_img.shape) == 2:  # If the image is grayscale
            high_res_img = cv2.cvtColor(high_res_img, cv2.COLOR_GRAY2RGB)
        if len(resized_low_res_img.shape) == 2:  # If the image is grayscale
            resized_low_res_img = cv2.cvtColor(resized_low_res_img, cv2.COLOR_GRAY2RGB)

        low_res_img = torch.from_numpy(low_res_img).float() / 255.0
        high_res_img = torch.from_numpy(high_res_img).float() / 255.0
        resized_low_res_img = torch.from_numpy(resized_low_res_img).float() / 255.0

        low_res_img = low_res_img.permute(2, 0, 1)
        high_res_img = high_res_img.permute(2, 0, 1)
        resized_low_res_img = resized_low_res_img.permute(2, 0, 1)

        if self.transforms:
            low_res_img = self.transforms(low_res_img)
            high_res_img = self.transforms(high_res_img)
            resized_low_res_img = self.transforms(resized_low_res_img)

        return low_res_img, high_res_img, resized_low_res_img

"""
        # Convert to PIL Images for efficient processing
        low_res_img = Image.fromarray(low_res_img)
        high_res_img = Image.fromarray(high_res_img)
        resized_low_res_img = Image.fromarray(resized_low_res_img)

       # Apply transformations if any
        if self.transforms:
            low_res_img = self.transforms(low_res_img)
            high_res_img = self.transforms(high_res_img)
            resized_low_res_img = self.transforms(resized_low_res_img)
        else:
            low_res_img = T.ToTensor()(low_res_img)
            high_res_img = T.ToTensor()(high_res_img)
            resized_low_res_img = T.ToTensor()(resized_low_res_img)

        return low_res_img, high_res_img, resized_low_res_img
"""

# Function to unnormalize and convert tensor to numpy array for displaying
def unnormalize(tensor, mean, std):
    tensor = tensor.clone()  # Clone the tensor to avoid modifying the original
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def show_images(low_res, high_res, resized_low_res, result):
    low_res = low_res.cpu().permute(1, 2, 0).numpy() * 255.0
    high_res = high_res.cpu().permute(1, 2, 0).numpy() * 255.0
    resized_low_res = resized_low_res.cpu().permute(1, 2, 0).numpy() * 255.0
    result = result.cpu().permute(1, 2, 0).numpy() * 255.0

    low_res = low_res.astype(np.uint8)
    high_res = high_res.astype(np.uint8)
    resized_low_res = resized_low_res.astype(np.uint8)
    result = result.astype(np.uint8)


    plt.figure(figsize=(12, 6))

    plt.subplot(1, 4, 1)
    plt.title("Low Resolution")
    plt.imshow(low_res)

    plt.subplot(1, 4, 2)
    plt.title("High Resolution")
    plt.imshow(high_res)

    plt.subplot(1, 4, 3)
    plt.title("Resized Low Resolution")
    plt.imshow(resized_low_res)

    plt.subplot(1, 4, 4)
    plt.title("Model Output")
    plt.imshow(result)

    plt.show()

transforms = T.Compose([
    T.RandomHorizontalFlip(),
    #T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':

    # if os.name == 'nt':
    #   import multiprocessing
    #   multiprocessing.set_start_method('spawn', force=True)

    # Paths to the .npy files
    train_data_path = r'C:\Users\ethan.DESKTOP-TL348AV\SURF Summer Research\SURF Research\SURF Week5_Meeting 4\data\train_pairs.npy'
    test_data_path = r'C:\Users\ethan.DESKTOP-TL348AV\SURF Summer Research\SURF Research\SURF Week5_Meeting 4\data\test_pairs.npy'

    # Create datasets and dataloaders
    train_dataset = ImagePairDataset(train_data_path, transforms=transforms)
    test_dataset = ImagePairDataset(test_data_path, transforms=transforms)

    train_loader = DataLoader(train_dataset, batch_size=14, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=14, shuffle=False, num_workers=0)
"""
    # Example usage of the dataloaders
    for low_res, high_res, resized_low_res in train_loader:
        print("Train batch - Low Res:", low_res.shape, "High Res:", high_res.shape, "Resized Low Res", resized_low_res.shape)

        # Unnormalize the images
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        low_res = unnormalize(low_res[1].clone(), mean, std)
        high_res = unnormalize(high_res[1].clone(), mean, std)
        resized_low_res = unnormalize(resized_low_res[1].clone(), mean, std)


        # Display the first image in the batch
        show_images(low_res, high_res, resized_low_res)
        break

    for low_res, high_res, resized_low_res in test_loader:
        print("Test batch - Low Res:", low_res.shape, "High Res:", high_res.shape, "Resized Low Res", resized_low_res.shape)

        # Unnormalize the images
        low_res = unnormalize(low_res[0].clone(), mean, std)
        high_res = unnormalize(high_res[0].clone(), mean, std)
        resized_low_res = unnormalize(resized_low_res[0].clone(), mean, std)

        # Display the first image in the batch
        show_images(low_res, high_res, resized_low_res)
        break
"""

#------------------------------------------------------------------------------------------
# Create CelebA UNet Model
import torch
import torch.nn as nn
import torch.optim as optim

class CelebA_UNetModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(CelebA_UNetModel, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dropout4 = nn.Dropout2d(p=dropout_rate)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        #self.bn3 = nn.BatchNorm2d(128)
        self.bottleneck2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        #self.bn4 = nn.BatchNorm2d(64)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dropout5 = nn.Dropout2d(p=dropout_rate)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        #self.bn5 = nn.BatchNorm2d(32)
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dropout6 = nn.Dropout2d(p=dropout_rate)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        #self.bn5 = nn.BatchNorm2d(32)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dropout7 = nn.Dropout2d(p=dropout_rate)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv15 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        #self.bn5 = nn.BatchNorm2d(32)
        self.conv16 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dropout8 = nn.Dropout2d(p=dropout_rate)

        # Output Layer
        self.outconv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        # ReLU Layer
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encode
        x = self.relu(self.conv1(x))
        #x = self.bn1(x)
        x = self.relu(self.conv2(x))
        x = self.dropout1(x)
        skip_connection1 = x
        x = self.pool1(x)

        x = self.relu(self.conv3(x))
        #x = self.bn1(x)
        x = self.relu(self.conv4(x))
        x = self.dropout2(x)
        skip_connection2 = x
        x = self.pool2(x)

        x = self.relu(self.conv5(x))
        #x = self.bn1(x)
        x = self.relu(self.conv6(x))
        x = self.dropout3(x)
        skip_connection3 = x
        x = self.pool3(x)

        x = self.relu(self.conv7(x))
        #x = self.bn1(x)
        x = self.relu(self.conv8(x))
        x = self.dropout4(x)
        skip_connection4 = x
        x = self.pool4(x)

        # Bottleneck
        x = self.relu(self.bottleneck(x))
        #x = self.bn3(x)
        x = self.relu(self.bottleneck2(x))

        # Decoder
        x = self.upconv1(x)
        x = self.relu(self.conv9(torch.cat((skip_connection4, x), 1))) # Concatenate after upsampling
        #x = self.bn4(x)
        x = self.relu(self.conv10(x))
        x = self.dropout5(x)

        x = self.upconv2(x)
        x = self.relu(self.conv11(torch.cat((skip_connection3, x), 1))) # Concatenate after upsampling
        #x = self.bn4(x)
        x = self.relu(self.conv12(x))
        x = self.dropout6(x)

        x = self.upconv3(x)
        x = self.relu(self.conv13(torch.cat((skip_connection2, x), 1))) # Concatenate after upsampling
        #x = self.bn4(x)
        x = self.relu(self.conv14(x))
        x = self.dropout7(x)

        x = self.upconv4(x)
        x = self.relu(self.conv15(torch.cat((skip_connection1, x), 1))) # Concatenate after upsampling
        #x = self.bn4(x)
        x = self.relu(self.conv16(x))
        x = self.dropout8(x)

        # Output Layer
        x = self.outconv(x)
        return x

#--------------------------------------------------------------------------------------------------------------------
# Train Function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CelebA_UNetModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Ensure the environment variable for memory management is set
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Create a directory to save models
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

def clear_cuda_cache():
    torch.cuda.empty_cache()

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filename, model, optimizer):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded. Epoch: {epoch}, Loss: {loss}")
        return epoch, loss
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, None

def train_and_evaluate(model, device, train_loader, test_loader, criterion, optimizer, start_epoch, epochs, save_frequency=5):
    for epoch in range(start_epoch, epochs):
        #Training
        model.train()
        total_loss = 0.0
        total_batches = len(train_loader)
        for batch_idx, (low_res, high_res, resized_low_res) in enumerate(train_loader):
            input_images = resized_low_res.to(device)
            high_res_image = high_res.to(device)

            optimizer.zero_grad()
            outputs = model(input_images)
            loss = criterion(outputs, high_res_image)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()  # This updates the model's weights

            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Batch Loss: {loss.item():.4f}, Average Loss: {total_loss / (batch_idx + 1):.4f}')

            # Clear cache occasionally to manage memory
            if (batch_idx + 1) % 100 == 0:
                clear_cuda_cache()

        average_train_loss = total_loss / total_batches
        print(f'Epoch {epoch + 1} completed, Average Training Loss: {average_train_loss:.4f}')

        # Evaluation phase
        low_res, high_res, resized_low_res, output = evaluate(model, device, test_loader, criterion)
        #evaluate(model, device, test_loader, criterion)

        # Compute SSIM and PSNR for the epoch
        ssim_values = []
        psnr_values = []
        for i in range(output.shape[0]):
            output_image = outputs[i].detach().cpu().numpy().transpose(1, 2, 0)
            high_res_image_np = high_res_image[i].detach().cpu().numpy().transpose(1, 2, 0)
            ssim_value = ssim(high_res_image_np, output_image, win_size=3, channel_axis=2, data_range=1.0)
            psnr_value = psnr(high_res_image_np, output_image, data_range=1.0)
            ssim_values.append(ssim_value)
            psnr_values.append(psnr_value)

        avg_ssim = np.mean(ssim_values)
        avg_psnr = np.mean(psnr_values)
        print(f'Epoch {epoch + 1}, SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}')

        # Save the model every `save_frequency` epochs
        if (epoch + 1) % save_frequency == 0:
            checkpoint_filename = os.path.join(save_dir, f"CelebA_model_with_dropout_epoch_{epoch + 1}.pth.tar")
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': average_train_loss,
            }, filename=checkpoint_filename)

        # Unnormalize the images for visualization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        low_res = unnormalize(low_res, mean, std)
        high_res = unnormalize(high_res, mean, std)
        resized_low_res = unnormalize(resized_low_res, mean, std)
        output = unnormalize(output, mean, std)

        # Display the images
        show_images(high_res, low_res, resized_low_res, output)
def evaluate(model, device, data_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_batches = len(data_loader)
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch_idx, (low_res, high_res, resized_low_res) in enumerate(data_loader):
            input_images = resized_low_res.to(device)
            high_res_image = high_res.to(device)

            outputs = model(input_images)
            loss = criterion(outputs, high_res_image)

            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f'Validation Batch {batch_idx + 1}, Batch Loss: {loss.item():.4f}')


            # Clear cache occasionally to manage memory
            if (batch_idx + 1) % 100 == 0:
                clear_cuda_cache()


    average_loss = total_loss / total_batches
    print(f'Validation Average Loss: {average_loss:.4f}')

    # Return a batch of images for visualization
    return low_res[0], high_res[0], resized_low_res[0], outputs[0]

# Load the checkpoint
checkpoint_path = "checkpoints/CelebA_model_with_dropout_epoch_18.pth.tar"  # Replace with the path to your checkpoint
start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer)

# Train and evaluate the model
train_and_evaluate(model, device, train_loader, test_loader, criterion, optimizer, start_epoch, epochs=20, save_frequency=1)











def train(model, device, train_loader, criterion, optimizer,epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (low_res, high_res, resized_low_res) in enumerate(train_loader):
            input_images = resized_low_res.to(device)
            high_res_image = high_res.to(device)

            optimizer.zero_grad()
            outputs = model(input_images)
            loss = criterion(outputs, high_res_image)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Average Loss: {total_loss / 10:.4f}')
                total_loss = 0.0

            clear_cuda_cache()

#train(model, device, train_loader, criterion, optimizer, epochs=1)





"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
device = "cuda" if torch.cuda.is_available() else "cpu"
psnr = PeakSignalNoiseRatio()
psnr.to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
ssim.to(device)

# Load Dataset
#----------------------------------------------------------------------------------------


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the transformations
original_transforms = transforms.Compose([
    transforms.ToTensor(), # Convert to tensor
    transforms.Resize((256, 256)),
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image
])

transform_down = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Resize((64, 64)),  # Resize to 7x7
])

transform_up = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize back to 28x28
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image

])

# Regular Dataset
train_dataset = datasets.CelebA(root='./data', train=True, download=True, transform=original_transforms)
test_dataset = datasets.CelebA(root='./data', train=False, download=True, transform=original_transforms)

# Load MNIST dataset and apply the downscale transformation
train_dataset_down = datasets.CelebA(root='./data', train=True, download=True, transform=transform_down)
test_dataset_down = datasets.CelebA(root='./data', train=False, download=True, transform=transform_down)

# Apply the upscale transformation on the already downscaled datasets
train_dataset_up = datasets.CelebA(root='./data', train=True, download=True, transform=transforms.Compose([transform_down, transform_up]))
test_dataset_up = datasets.CelebA(root='./data', train=False, download=True, transform=transforms.Compose([transform_down, transform_up]))

batch_size = 32

# Original Dataset
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Create DataLoader instances for the downscaled version
train_loader_down = DataLoader(dataset=train_dataset_down, batch_size=batch_size, shuffle=False)
test_loader_down = DataLoader(dataset=test_dataset_down, batch_size=batch_size, shuffle=False)

# Create DataLoader instances for the upscaled version from downscaled images
train_loader_up = DataLoader(dataset=train_dataset_up, batch_size=batch_size, shuffle=False)
test_loader_up = DataLoader(dataset=test_dataset_up, batch_size=batch_size, shuffle=False)

# Data loading code
traindir = 'your download image directory path'
testdir = 'your download image directory path'

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(traindir, transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])),
    batch_size=3, shuffle=True)
"""