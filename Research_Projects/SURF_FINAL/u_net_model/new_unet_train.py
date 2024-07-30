import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as T

# Check if the system has GPU, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

print("Start collecting training data!")

# Create Dataloaders
xray_ct_traindata = []
traindata = np.load('/home/ys92/dataset/ctslice_train.npy', allow_pickle=True)
print(len(traindata))

for i in tqdm(range(len(traindata)), desc="Collecting training data"):
    xray_ct_traindata.append(np.expand_dims(traindata[i], axis=0))

trainloader = torch.utils.data.DataLoader(xray_ct_traindata, shuffle=True, batch_size=32, num_workers=1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

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
        self.enc1 = nn.Sequential(CBR(in_channels, 64), ResidualBlock(64))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(CBR(64, 128), ResidualBlock(128))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = nn.Sequential(CBR(128, 256), ResidualBlock(256))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = nn.Sequential(CBR(256, 512), ResidualBlock(512))
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(CBR(512, 1024), ResidualBlock(1024))

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.dec4 = nn.Sequential(CBR(1024, 512), ResidualBlock(512))

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec3 = nn.Sequential(CBR(512, 256), ResidualBlock(256))

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec2 = nn.Sequential(CBR(256, 128), ResidualBlock(128))

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec1 = nn.Sequential(CBR(128, 64), ResidualBlock(64))

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)

        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)

        # Bottleneck
        middle = self.bottleneck(pool4)

        # Decoder
        up4 = self.up4(middle)
        att4 = self.att4(g=up4, x=enc4)
        cat4 = torch.cat([att4, up4], dim=1)
        dec4 = self.dec4(cat4)

        up3 = self.up3(dec4)
        att3 = self.att3(g=up3, x=enc3)
        cat3 = torch.cat([att3, up3], dim=1)
        dec3 = self.dec3(cat3)

        up2 = self.up2(dec3)
        att2 = self.att2(g=up2, x=enc2)
        cat2 = torch.cat([att2, up2], dim=1)
        dec2 = self.dec2(cat2)

        up1 = self.up1(dec2)
        att1 = self.att1(g=up1, x=enc1)
        cat1 = torch.cat([att1, up1], dim=1)
        dec1 = self.dec1(cat1)

        return self.final(dec1)


# Train the Model
# Assuming the UNet class is defined as shown previously
model = UNet(in_channels=1, out_channels=1)

# Move the model to the desired device (e.g., CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Optimizer and Criterion
lr = 3e-5
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()  # MAE LOSS

num_epochs = 100

print(num_epochs)
print("Start training model!")

# Save Model function
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

# Load the pre-trained model if it exists
start_epoch = 0
model_path = '/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/new_unet_savemodels/checkpoint.pth'

if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}.")
else:
    print(f"Pre-trained model not found. Training from scratch.")

# Two transforms for super resolution: from 64 * 64 --> 256 * 256
transform1 = T.Resize((64, 64), antialias=True)  # Explicitly set antialias
transform2 = T.Resize((256, 256), antialias=True)  # Explicitly set antialias

# Start Training
for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    epoch_loss = 0

    with tqdm(trainloader, unit="batch") as tepoch:
        for i, data in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")

            ct_train_slice = data.to(device)  # ground truth shape (32, 1, 256, 256)
            ct_train_slice = ct_train_slice.reshape(32, 1, 256, 256)
            ct_train_slice_64 = transform1(ct_train_slice)  # downsample first
            ct_train_slice_256 = transform2(ct_train_slice_64)  # input, or conditional input, because we need to make shape consistent with output
            
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

    save_checkpoint(model, optimizer, epoch, model_path)

        # Save additional checkpoint every 15 epochs with a filename indicating the epoch
    epoch_checkpoint_path = f"{model_path}_epoch_{epoch+1}.pth"
    save_checkpoint(model, optimizer, epoch, epoch_checkpoint_path)

