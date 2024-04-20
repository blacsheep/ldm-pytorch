import os

import numpy as np
from torchvision import utils as vutils
from tqdm import tqdm

import sys

sys.path.append("/home/lxia")

from ldm.data.mscoco_dataloader import get_dataloader
from ldm.loss.vqperceptual import VQLPIPSWithDiscriminator
from ldm.model.ldm import *


# https://github.com/dome272/VQGAN-pytorch/blob/main/training_vqgan.py
class TrainVQ:
    def __init__(self, vq, dataloader, num_epochs, lr=1e-6, betas=(0.5, 0.9), device='cuda'):
        if device == 'cuda':
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vq = vq.to(self.device)
        self.dataloader = dataloader
        self.lr = lr
        self.betas = betas
        self.vq_perceptual_disc = VQLPIPSWithDiscriminator().to(self.device)
        self.num_epochs = num_epochs

    def get_optimizer_autoencoder(self):
        autoencoder = self.vq
        lr = self.lr
        betas = self.betas
        optimizer = torch.optim.Adam(list(autoencoder.encoder.parameters()) +
                                     list(autoencoder.decoder.parameters()) +
                                     list(autoencoder.vq.parameters()) +
                                     list(autoencoder.quant_conv.parameters()) +
                                     list(autoencoder.post_quant_conv.parameters()),
                                     lr=lr, betas=betas
                                     )
        return optimizer

    def get_optimizer_discriminator(self):
        discriminator = self.vq_perceptual_disc.discriminator
        lr = self.lr
        betas = self.betas
        optimizer = torch.optim.Adam(list(discriminator.parameters()),
                                     lr=lr, betas=betas
                                     )
        return optimizer

    def train(self):
        opt_autoencoder = self.get_optimizer_autoencoder()
        opt_discriminator = self.get_optimizer_discriminator()
        counter = 0
        for epoch in range(self.num_epochs):
            with tqdm(self.dataloader) as pbar:
                for i, (_, (txt, img)) in enumerate(zip(pbar, self.dataloader)):
                    counter += len(img)
                    img = img.to(self.device)
                    decoded_img, codebook_loss = self.vq(img)
                    vq_loss, d_loss = self.vq_perceptual_disc(codebook_loss=codebook_loss,
                                                              inputs=img,
                                                              reconstructions=decoded_img,
                                                              global_step=counter,
                                                              last_layer=self.vq.get_last_layer()
                                                              )
                    opt_autoencoder.zero_grad()
                    opt_discriminator.zero_grad()

                    # not to free the grads after `backward` is called
                    vq_loss.backward(retain_graph=True)
                    d_loss.backward()

                    opt_autoencoder.step()
                    opt_discriminator.step()

                    if i % 1000 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat((img[:4], decoded_img.add(1).mul(0.5)[:4]))
                            vutils.save_image(real_fake_images, os.path.join("results", f"{epoch}_{i}.jpg"), nrow=4)

                    pbar.set_postfix(
                        vq_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        d_Loss=np.round(d_loss.cpu().detach().numpy().item(), 3)
                    )
                    pbar.update(0)
                torch.save(self.vq.state_dict(), os.path.join("checkpoints", f"vq_epoch_{epoch}.pt"))


if __name__ == '__main__':
    BATCH_SIZE = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = '/home/lxia/ldm/data/dataset/{00000..00059}.tar'
    dataloader = get_dataloader(data_path, batch_size=BATCH_SIZE)

    num_residual = 2
    vq = VQModel(n_e=4096, e_dim=256, beta=0.25, input_channel=256, num_res=num_residual)
    num_epochs = 20
    trainer = TrainVQ(vq=vq, dataloader=dataloader, num_epochs=num_epochs)

    trainer.train()
