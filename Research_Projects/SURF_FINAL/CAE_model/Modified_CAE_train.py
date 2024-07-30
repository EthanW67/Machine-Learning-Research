import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import torchvision.transforms as T
from tqdm import tqdm

"""
Define the CAE model
"""

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(ConvolutionalAutoencoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.relu = nn.ReLU()

        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.pool3(x)

        x = self.relu(self.upconv1(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.upconv2(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.upconv3(x))
        x = self.conv9(x)
        return x

"""
Check if the system has GPU, else use CPU
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

print("Start collecting training data!")

"""
Create Dataloaders

The original CT slice images are with size 256 * 256 (please double check it), we will downsample them to 64 * 64 in the training loop
"""

xray_ct_traindata = []
traindata = np.load('/home/ys92/dataset/ctslice_train.npy', allow_pickle=True)
print(len(traindata))

# Use tqdm to show progress while collecting training data
for i in tqdm(range(len(traindata)), desc="Collecting training data"):
    xray_ct_traindata.append(np.expand_dims(traindata[i], axis=0))

trainloader = torch.utils.data.DataLoader(xray_ct_traindata, shuffle=True, batch_size=32, num_workers=1)

"""
Train the Model
"""
model = ConvolutionalAutoencoder(in_channels=1, out_channels=1).to(device)
model.to(device)

"""
Optimizer and Criterion
"""
lr = 3e-5
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()  # Mean Squared Error Loss

num_epochs = 100

print(num_epochs)
print("Start training model!")

"""
Checkpoint Save/Load functions
"""

def save_checkpoint(model, optimizer, epoch, loss, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filename, model, optimizer):
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded: start from epoch {epoch + 1} with loss {loss}")
        return epoch + 1, loss  # Resume from the next epoch
    else:
        print(f"No checkpoint found at {filename}, starting from scratch.")
        return 0, None  # Start from scratch

"""
Two transforms:

We want to do super resolution: 

from 64 * 64 --> 256 * 256
"""

transform1 = T.Resize((64, 64), antialias=True)  # Explicitly set antialias
transform2 = T.Resize((256, 256), antialias=True)  # Explicitly set antialias

# Load the pre-trained model if it exists
checkpoint_path = '/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/Modified_CAE_savemodels/checkpoint.pth'
start_epoch, last_loss = load_checkpoint(checkpoint_path, model, optimizer)

# Override start_epoch if specified in arguments
parser = argparse.ArgumentParser(description="Train CAE Model")
parser.add_argument("--start-epoch", type=int, default=start_epoch, help="Specify the epoch to start training from")
args = parser.parse_args()

start_epoch = args.start_epoch

"""
Start Training
"""

for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    epoch_loss = 0
    
    with tqdm(trainloader, unit="batch") as tepoch:
        for i, data in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")

            ct_train_slice = data.to(device)  # ground truth shape (32, 1, 256, 256)
            ct_train_slice = ct_train_slice.reshape(32, 1, 256, 256)
            ct_train_slice_64 = transform1(ct_train_slice)  # downsample first
            ct_train_slice_256 = transform2(ct_train_slice_64)  # upsample back to 256x256
            
            # Forward pass
            output = model(ct_train_slice_256)
            loss = criterion(output, ct_train_slice)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())
            epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(trainloader)}")

    # Save the model and optimizer state at the end of every epoch
    save_checkpoint(model, optimizer, epoch, loss.item(), checkpoint_path)

        # Save additional checkpoint every 15 epochs with a filename indicating the epoch
    if (epoch + 1) % 5 == 0:
        epoch_checkpoint_path = f"{checkpoint_path}_epoch_{epoch+1}.pth"
        save_checkpoint(model, optimizer, epoch, loss.item(), epoch_checkpoint_path)