import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

# Check if the system has GPU, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print("Start collecting testing data!")

xray_ct_testdata = []
testdata = np.load('/home/ys92/dataset/ctslice_val.npy', allow_pickle=True)
print(len(testdata))
for i in range(len(testdata)):
    if testdata[i].shape == (256, 256):
        xray_ct_testdata.append(np.expand_dims(testdata[i], axis=0))

testloader = DataLoader(xray_ct_testdata, shuffle=False, batch_size=1, num_workers=1)

# Define the UNet model (same as in training script)
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

# Initialize the model
model = UNet(in_channels=1, out_channels=1)
model.to(device)

# Load the checkpoint and extract the model state dictionary
checkpoint_path = '/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/new_unet_savemodels/checkpoint.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded model from checkpoint")




# Two transforms: from 64 * 64 --> 256 * 256
transform1 = T.Resize((64, 64), antialias=True)
transform2 = T.Resize((256, 256), antialias=True)

# transform1 = T.Resize((64, 64))
# transform2 = T.Resize((256, 256))

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
iteration_to_save = 50  # Change this to the desired iteration
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
            plt.imshow(ct_test_slice.reshape(256, 256).cpu().numpy(), cmap='gray')
            plt.title('Original CT Slice (256x256)')
            plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/new_unet_test_figs/gtcttest_{}_original.png'.format(i), dpi=300) # ground truth figure

            plt.imshow(ct_test_slice_64.reshape(64, 64).cpu().numpy(), cmap='gray')
            plt.title('Downsampled CT Slice (64x64)')
            plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/new_unet_test_figs/gtcttest_{}_64.png'.format(i), dpi=300) # downsampled figure

            plt.imshow(ct_test_slice_256.reshape(256, 256).cpu().numpy(), cmap='gray')
            plt.title('Upsampled CT Slice (256x256)')
            plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/new_unet_test_figs/gtcttest_{}_256.png'.format(i), dpi=300) # upsampled figure

            plt.imshow(predicted_ct_slice.reshape(256, 256).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            plt.title('Predicted CT Slice (256x256)')
            plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/new_unet_test_figs/predictcttest_{}.png'.format(i), dpi=300) # prediction figure

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
    plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/new_unet_test_figs/average_psnr.png', dpi=300)

    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, f'Average SSIM: {avg_ssim:.4f}', fontsize=15, ha='center')
    ax.axis('off')
    plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/new_unet_test_figs/average_ssim.png', dpi=300)
