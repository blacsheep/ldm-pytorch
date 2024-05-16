from torch.optim import Adam
from tqdm import tqdm
from ldm.model.ldm import *
from ldm.data.mscoco_dataloader import get_dataloader
import os
import torch


if __name__ == '__main__':
    # Params
    IMG_SIZE = 256
    BATCH_SIZE = 12
    T = 300
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataloader
    data_path = '/home/lxia/ldm/ldm/data/dataset/{00010..00014}.tar'
    dataloader = get_dataloader(data_path, batch_size=BATCH_SIZE)

    # model
    num_residual = 3
    model = LDM(n_e=4096*2, e_dim=256, beta=0.25, input_channel=256, num_res=num_residual)
    model.to(device)
    optimizer = Adam(list(model.unet.parameters()) +
                     list(model.cross_attn.parameters()),
                     lr=0.001)
    epochs = 100

    for epoch in range(epochs):
        with tqdm(dataloader) as pbar:
            for i, (_, (txt, img)) in enumerate(zip(pbar, dataloader)):
                optimizer.zero_grad()
                img = img.to(device)
                t = torch.randint(0, T, (len(txt),), device=device).long()
                noise_pred, loss = model(img, t, txt)

                loss.backward()
                optimizer.step()
            torch.save(model.state_dict(), os.path.join("checkpoints_ldm", f"ldm_epoch_{epoch}.pt"))
