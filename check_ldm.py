import os
import numpy as np
from torchvision import utils as vutils
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt
from ldm.data.mscoco_dataloader import get_dataloader
from ldm.loss.vqperceptual import VQLPIPSWithDiscriminator
from ldm.model.ldm import *
from torchvision import transforms

T = 300
BATCH_SIZE = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
num_residual = 3
ldm = model = LDM(n_e=4096*2, e_dim=256, beta=0.25, input_channel=256, num_res=num_residual)
ldm.to(device)
ldm.init_from_ckpt('checkpoints_ldm/ldm_epoch_79.pt')
ldm.eval()

img_size = 64
img = torch.randn((1, 256, img_size, img_size), device=device)
num_images = 10
stepsize = int(T / num_images)
text = 'a bowl of cats.'

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

for i in range(0, T)[::-1]:
    t = torch.full((1,), i, device=device, dtype=torch.long)
    img = ldm.ddpm.sample_timestep(img, t)
    if i % stepsize == 0:
        dec = ldm.backward(img, text)
        dec = torch.clamp(dec, -1.0, 1.0)
        plt.subplot(1, num_images, int(i / stepsize) + 1)
        show_tensor_image(dec.detach().cpu())
    plt.savefig('output.png')
