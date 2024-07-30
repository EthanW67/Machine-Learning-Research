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
Define the U-Net model.
"""

# change checkpoint_path to desired epoch

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        # Encoder
        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = CBR(64, 128)
        self.enc4 = CBR(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc5 = CBR(128, 256)
        self.enc6 = CBR(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc7 = CBR(256, 512)
        self.enc8 = CBR(512, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck1 = CBR(512, 1024)
        self.bottleneck2 = CBR(1024, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.dec4b = CBR(512, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = CBR(512, 256)
        self.dec3b = CBR(256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CBR(256, 128)
        self.dec2b = CBR(128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = CBR(128, 64)
        self.dec1b = CBR(64, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc1 = self.enc2(enc1)
        pool1 = self.pool1(enc1)

        enc2 = self.enc3(pool1)
        enc2 = self.enc4(enc2)
        pool2 = self.pool2(enc2)

        enc3 = self.enc5(pool2)
        enc3 = self.enc6(enc3)
        pool3 = self.pool3(enc3)

        enc4 = self.enc7(pool3)
        enc4 = self.enc8(enc4)
        pool4 = self.pool4(enc4)

        # Bottleneck
        middle = self.bottleneck1(pool4)
        middle = self.bottleneck2(middle)

        # Decoder
        up4 = self.up4(middle)
        cat4 = torch.cat([enc4, up4], dim=1)
        dec4 = self.dec4(cat4)
        dec4 = self.dec4b(dec4)

        up3 = self.up3(dec4)
        cat3 = torch.cat([enc3, up3], dim=1)
        dec3 = self.dec3(cat3)
        dec3 = self.dec3b(dec3)

        up2 = self.up2(dec3)
        cat2 = torch.cat([enc2, up2], dim=1)
        dec2 = self.dec2(cat2)
        dec2 = self.dec2b(dec2)

        up1 = self.up1(dec2)
        cat1 = torch.cat([enc1, up1], dim=1)
        dec1 = self.dec1(cat1)
        dec1 = self.dec1b(dec1)

        return self.final(dec1)

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

# for i in range(len(traindata)):
#     xray_ct_traindata.append(np.expand_dims(traindata[i], axis=0))

# Use tqdm to show progress while collecting training data
for i in tqdm(range(len(traindata)), desc="Collecting training data"):
    xray_ct_traindata.append(np.expand_dims(traindata[i], axis=0))

trainloader = torch.utils.data.DataLoader(xray_ct_traindata, shuffle=True, batch_size=32, num_workers=1)

"""
Train the Model
"""
model = UNet(in_channels=1, out_channels=1).to(device)
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
checkpoint_path = '/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/U-Net_savemodels/checkpoint_PSNR_SSIM.pth'
start_epoch, last_loss = load_checkpoint(checkpoint_path, model, optimizer)

# Override start_epoch if specified in arguments
parser = argparse.ArgumentParser(description="Train U-Net Model")
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
