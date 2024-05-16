from ldm.model.VQ import VQModel
from ldm.model.txt2embedding import BERTEmbedding
from ldm.model.attention import CrossAttention
from ldm.model.ddpm import DDPM
from torch import nn
import torch.nn.functional as F
import torch

class LDM(nn.Module):
    def __init__(self, n_e, e_dim, beta, input_channel, num_res, text_embedding_dim=768, f=8):
        super().__init__()
        # conditional stage
        self.bert_embedding = BERTEmbedding().eval()
        self.preset_cond_stage()
        # first stage model
        self.VQLayer = VQModel(n_e, e_dim, beta, input_channel, num_res, f).eval()
        self.load_first_stage()
        # unet and cross attention
        self.ddpm = DDPM()
        self.unet = self.ddpm.unet
        self.cross_attn = CrossAttention(input_channel, text_embedding_dim)

    def load_first_stage(self, path='checkpoints/vq_epoch_14.pt'):
        self.VQLayer.init_from_ckpt(path)
        for param in self.VQLayer.parameters():
            param.requires_grad = False

    def preset_cond_stage(self):
        for param in self.bert_embedding.parameters():
            param.requires_grad = False

    def forward(self, img, timestep, text):
        # 生成text embedding用来做cross attention
        embedding = self.bert_embedding(text)
        # 压缩到latent space
        quant, _, (_, _, _) = self.VQLayer.encode(img)
        # cross attention
        # quant: b c h w
        b, c, h, w = quant.shape
        quant = quant.permute(0, 2, 3, 1)
        quant = quant.view(b, h*w, c)
        latent_vec_with_attn = self.cross_attn(quant, embedding)
        latent_vec_with_attn = latent_vec_with_attn.permute(0, 2, 1).view(b, c, h, w)
        # unet procedure
        noised_img, noice = self.ddpm.forward_diffusion_sample(latent_vec_with_attn, timestep)
        output_ = self.unet(noised_img, timestep)
        mse_loss = F.mse_loss(output_, noice)
        return output_, mse_loss

    @torch.no_grad()
    def backward(self, img, text):
        embedding = self.bert_embedding(text)
        # 压缩到latent space
        b, c, h, w = img.shape
        quant = img.permute(0, 2, 3, 1)
        quant = quant.view(b, h*w, c)
        latent_vec_with_attn = self.cross_attn(quant, embedding)
        latent_vec_with_attn = latent_vec_with_attn.permute(0, 2, 1).view(b, c, h, w)
        decoded_img = self.VQLayer.decode(latent_vec_with_attn)
        return decoded_img
    def init_from_ckpt(self, path, ignore_keys=None, only_model=False):
        if ignore_keys is None:
            ignore_keys = []
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]