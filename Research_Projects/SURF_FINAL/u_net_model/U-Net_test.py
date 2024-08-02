import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
"""
Checks these factors:
checkpoint = torch.load('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/U-Net_savemodels/checkpoint_PSNR_SSIM.pth')
testdata = np.load('/home/ys92/dataset/ctslice_val.npy', allow_pickle=True)
plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/U-Net_test_figs/gtcttest_{}_original.png'.format(i), dpi=300) # ground truth figure
plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/U-Net_test_figs/average_psnr.png', dpi=300)

"""


"""
Check if the system has GPU, else use CPU
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print("Start collecting training data!")

"""
Create Dataloaders
"""
xray_ct_testdata = []
testdata = np.load('/home/ys92/dataset/ctslice_val.npy', allow_pickle=True)
print(len(testdata))
for i in range(len(testdata)):
    if testdata[i].shape == (256, 256):
        xray_ct_testdata.append(np.expand_dims(testdata[i], axis=0))

testloader = DataLoader(xray_ct_testdata, shuffle=False, batch_size=1, num_workers=1)

"""
Define the U-Net model
"""
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
Initialize the model and optimizer
"""
model = UNet(in_channels=1, out_channels=1)
model.to(device)

optimizer = torch.optim.Adam(model.parameters())

# Load the checkpoint and extract the model state dictionary
checkpoint = torch.load('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/U-Net_savemodels/checkpoint_PSNR_SSIM.pth')
# checkpoint = torch.load('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/U-Net_savemodels/checkpoint_PSNR_SSIM.pth_epoch_15.pth')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print(f"Loaded model and optimizer from epoch {epoch} with loss {loss}")

"""
Two transforms:

We want to do super resolution: 

from 64 * 64 --> 256 * 256
"""
transform1 = T.Resize((64, 64), antialias=True)
transform2 = T.Resize((256, 256), antialias=True)




#Testing Process

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Custom functions to calculate PSNR and SSIM
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = img1.var()
    sigma2 = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    return ssim

# Main code
iteration_to_save = 6300  # Change this to the desired iteration
total_psnr = 0
total_ssim = 0
num_samples = 0

with torch.no_grad():
    for i, data in tqdm(enumerate(testloader, 0), total=len(testloader), desc="Processing"):
        ct_test_slice = data.to(device)  # output ground truth (1, 1, 256, 256)
        ct_test_slice = ct_test_slice.reshape(1, 1, 256, 256)

        ct_test_slice_64 = transform1(ct_test_slice)  # downsample first
        ct_test_slice_256 = transform2(ct_test_slice_64)  # input, or conditional input, because we need make shape consistent with output

        # Inference
        predicted_ct_slice = model(ct_test_slice_256)

        # Calculate PSNR and SSIM
        gt_np = ct_test_slice.cpu().numpy().reshape(256, 256)
        pred_np = predicted_ct_slice.cpu().numpy().reshape(256, 256)

        psnr = calculate_psnr(gt_np, pred_np)
        ssim = calculate_ssim(gt_np, pred_np)

        total_psnr += psnr
        total_ssim += ssim
        num_samples += 1

        # Save images for the specified iteration
        if i == iteration_to_save:

            plt.figure(figsize=(6, 6))
            plt.imshow(ct_test_slice.reshape(256, 256).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            # plt.title('Original CT Slice (256x256)')
            plt.axis('off')
            plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/U-Net_test_figs/original_CT_Slice_{}_original.png'.format(i), dpi=600, bbox_inches='tight', pad_inches=0) # ground truth figure

            plt.figure(figsize=(6, 6))
            plt.imshow(ct_test_slice_64.reshape(64, 64).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            # plt.title('Downsampled CT Slice (64x64)')
            plt.axis('off')
            plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/U-Net_test_figs/downsampled_CT_Slice_{}_64.png'.format(i), dpi=600, bbox_inches='tight', pad_inches=0) # downsampled figure

            plt.figure(figsize=(6, 6))
            plt.imshow(ct_test_slice_256.reshape(256, 256).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            # plt.title('Upsampled CT Slice (256x256)')
            plt.axis('off')
            plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/U-Net_test_figs/upsampled_CT_Slice_{}_256.png'.format(i), dpi=600, bbox_inches='tight', pad_inches=0) # upsampled figure

            plt.figure(figsize=(6, 6))
            plt.imshow(predicted_ct_slice.reshape(256, 256).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            #plt.title('Predicted CT Slice (256x256)')
            plt.axis('off')
            plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/U-Net_test_figs/predict_CT_Slice_{}.png'.format(i), dpi=600, bbox_inches='tight', pad_inches=0) # prediction figure



            # plt.figure(figsize=(6, 6))
            # plt.imshow(predicted_ct_slice.reshape(256, 256).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            # plt.title('dpi = 600')
            # plt.axis('off')
            # plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/new_unet_test_figs/dpi_600_{}.png'.format(i), dpi=600, bbox_inches='tight', pad_inches=0) # prediction figure



    # Calculate average PSNR and SSIM
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    # Print average PSNR and SSIM
    print(f'Average PSNR: {avg_psnr:.4f}')
    print(f'Average SSIM: {avg_ssim:.4f}')

    # Save average PSNR and SSIM as images
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, f'Average PSNR: {avg_psnr:.4f}', fontsize=15, ha='center')
    ax.axis('off')
    plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/U-Net_test_figs/average_psnr.png', dpi=300)

    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, f'Average SSIM: {avg_ssim:.4f}', fontsize=15, ha='center')
    ax.axis('off')
    plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/U-Net_test_figs/average_ssim.png', dpi=300)

