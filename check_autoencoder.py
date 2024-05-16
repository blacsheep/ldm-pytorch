import os
import numpy as np
from torchvision import utils as vutils
from tqdm import tqdm
import sys

from ldm.data.mscoco_dataloader import get_dataloader
from ldm.loss.vqperceptual import VQLPIPSWithDiscriminator
from ldm.model.ldm import *

BATCH_SIZE = 2
data_path = '/home/lxia/ldm/data/dataset/{00010..00014}.tar'
dataloader = get_dataloader(data_path, batch_size=BATCH_SIZE)
device = "cuda" if torch.cuda.is_available() else "cpu"
num_residual = 3
vq = VQModel(n_e=4096*2, e_dim=256, beta=0.25, input_channel=256, num_res=num_residual)
vq.to(device)

vq.init_from_ckpt('checkpoints/vq_epoch_14.pt')
vq.eval()
with tqdm(dataloader) as pbar:
    for i, (_, (txt, img)) in enumerate(zip(pbar, dataloader)):
        img = img.to(device)
        decoded_img, codebook_loss = vq(img)
        if i % 50 == 0:
            with torch.no_grad():
                real_fake_images = torch.cat((img[:4], decoded_img[:4]))
                vutils.save_image(real_fake_images, os.path.join("check_performance", f"{i}.jpg"), nrow=4)

        pbar.update(0)
