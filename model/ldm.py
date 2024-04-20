from ldm.model.VQ import *
from ldm.model.txt2embedding import *
from ldm.model.unet import *


class LDM(nn.Module):
    def __init__(self, n_e, e_dim, beta, input_channel, num_res, text_embedding_dim=768, f=8):
        super().__init__()
        self.bert_embedding = BERTEmbedding()
        self.VQLayer = self.VQModel(n_e, e_dim, beta, input_channel, num_res, f)
        self.unet = SimpleUnet()
        self.cross_attn = CrossAttention(input_channel // f, text_embedding_dim)

    def forward(self, x, timestep, text):
        # 生成text embedding用来做cross attention
        embedding = self.bert_embedding(text)
        # 压缩到latent space
        quant, diff, (_, _, ind) = self.VQLayer.encode(embedding)
        # cross attention
        latent_vec_with_attn = self.cross_attn(quant, embedding)
        # unet procedure
        unet_output = self.unet(latent_vec_with_attn, timestep)
        # decode the generated latent vector as final result
        result = self.VQLayer.decode(unet_output)
        return result, diff
