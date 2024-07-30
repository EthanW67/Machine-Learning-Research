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
Initialize the model and optimizer
"""
model = ConvolutionalAutoencoder(in_channels=1, out_channels=1)
model.to(device)

optimizer = torch.optim.Adam(model.parameters())

# Load the checkpoint and extract the model state dictionary
checkpoint = torch.load('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/Modified_CAE_savemodels/modified_checkpoint.pth')
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
transform1 = T.Resize((64, 64))
transform2 = T.Resize((256, 256))



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
            plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/Modified_CAE_test_figs/gtcttest_{}_original.png'.format(i), dpi=300) # ground truth figure

            plt.imshow(ct_test_slice_64.reshape(64, 64).cpu().numpy(), cmap='gray')
            plt.title('Downsampled CT Slice (64x64)')
            plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/Modified_CAE_test_figs/gtcttest_{}_64.png'.format(i), dpi=300) # downsampled figure

            plt.imshow(ct_test_slice_256.reshape(256, 256).cpu().numpy(), cmap='gray')
            plt.title('Upsampled CT Slice (256x256)')
            plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/Modified_CAE_test_figs/gtcttest_{}_256.png'.format(i), dpi=300) # upsampled figure

            plt.imshow(predicted_ct_slice.reshape(256, 256).cpu().numpy(), cmap='gray')
            plt.title('Predicted CT Slice (256x256)')
            plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/Modified_CAE_test_figs/predictcttest_{}.png'.format(i), dpi=300) # prediction figure

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
    plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/Modified_CAE_test_figs/average_psnr.png', dpi=300)

    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, f'Average SSIM: {avg_ssim:.4f}', fontsize=15, ha='center')
    ax.axis('off')
    plt.savefig('/home/ys92/EthanSURF/SURF_Research/SURF_Final/cond_diffu_CTs/Modified_CAE_test_figs/average_ssim.png', dpi=300)

