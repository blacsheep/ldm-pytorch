from einops import rearrange
import math
from ldm.model.attention import *
import torch.nn.functional as F

class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


# https://zh.d2l.ai/chapter_convolutional-modern/resnet.html
# the most basic ResNet structure
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.gn1 = Normalize(num_channels)
        self.gn2 = Normalize(num_channels)

    def forward(self, X):
        Y = F.relu(self.gn1(self.conv1(X)))
        Y = self.gn2((self.conv2(Y)))
        if self.conv3:
            X = self.conv3(X)

        if X.shape != Y.shape:
            return F.relu(Y)
        Y += X
        return F.relu(Y)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Encoder(nn.Module):
    def __init__(self, input_channels, number_res, f=8, attn_res=32):
        """
        Initialize the class with the given input channels, number of residuals, and down sampling factor f.
        Parameters:
            input_channels (int): The number of input channels.
            number_res (int): The number of residuals.
            f (int, optional): Down sampling factor with a default value of 8.
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = input_channels // f
        self.level = math.log2(f)
        if self.level != int(self.level):
            raise ValueError('Down Sampling factor Wrong!')
        self.level = int(self.level)
        self.block = nn.ModuleList()

        self.rgb2input_channel = nn.Conv2d(3, input_channels, 3, 1, 1)
        self.block.append(self.rgb2input_channel)
        # down sampling
        # 每一个level维度除以2
        curr_output = self.input_channels
        for i_level in range(self.level):
            curr_f = 2 ** i_level
            curr_input = self.input_channels // curr_f
            curr_output = curr_input // 2
            self.res_sequence = nn.ModuleList()
            # 每次除以2连接n_res个残差块
            for i in range(number_res):
                if i == 0:
                    self.res_sequence.append(Residual(curr_input, curr_output))
                else:
                    self.res_sequence.append(Residual(curr_output, curr_output))
                # self-attention 收尾
                if curr_output == attn_res:
                    self.attn = SelfAttention(curr_output)
                    self.res_sequence.append(self.attn)

            self.block.append(nn.Sequential(*self.res_sequence))
            if i_level != self.level - 1:
                self.block.append(DownSampleBlock(curr_output))
        # mid
        self.mid = nn.ModuleList()
        self.mid.append(Residual(curr_output, curr_output))
        self.mid.append(SelfAttention(curr_output))
        self.mid.append(Residual(curr_output, curr_output))

        # end
        self.end = nn.ModuleList()
        self.end.append(Normalize(curr_output))
        self.end.append(Swish())
        self.conv_out = torch.nn.Conv2d(curr_output, curr_output,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.end.append(self.conv_out)

    def forward(self, x):
        for layer in self.block:
            x = layer(x)

        for layer in self.mid:
            x = layer(x)

        for layer in self.end:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_channels, number_res, f=8, attn_res=32):
        """
        Initialize the class with the given input channels, number of residuals, and down sampling factor f.
        Parameters:
            input_channels (int): The number of input channels.
            number_res (int): The number of residuals.
            f (int, optional): Down sampling factor with a default value of 8.
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = input_channels // f
        self.level = math.log2(f)
        if self.level != int(self.level):
            raise ValueError('Down Sampling factor Wrong!')
        self.level = int(self.level)
        self.block = nn.ModuleList()
        # 每一个level维度除以2
        curr_output = self.output_channels

        # mid
        self.mid = nn.ModuleList()
        conv_in = torch.nn.Conv2d(curr_output, curr_output,
                                                          kernel_size=3,
                                                          stride=1,
                                                          padding=1)
        self.mid.append(conv_in)
        self.mid.append(Residual(curr_output, curr_output))
        self.mid.append(SelfAttention(curr_output))
        self.mid.append(Residual(curr_output, curr_output))

        # upsampling
        for i_level in range(self.level):
            curr_f = 2 ** i_level
            curr_input = self.output_channels * curr_f
            curr_output = curr_input * 2
            self.res_sequence = nn.ModuleList()
            # 每次除以2连接n_res个残差块
            for i in range(number_res):
                if i == 0:
                    self.res_sequence.append(Residual(curr_input, curr_output))
                else:
                    self.res_sequence.append(Residual(curr_output, curr_output))
                # self-attention 收尾
                if curr_output == attn_res:
                    self.attn = SelfAttention(curr_output)
                    self.res_sequence.append(self.attn)
            self.block.append(nn.Sequential(*self.res_sequence))
            if i_level != self.level - 1:
                self.block.append(UpSampleBlock(curr_output))

        # end
        self.end = nn.ModuleList()
        self.end.append(Normalize(curr_output))
        self.end.append(Swish())
        self.conv_out = nn.Conv2d(curr_output, 3, 3, 1, 1)
        self.end.append(self.conv_out)

    def forward(self, x):
        # mid -> upsampling -> end
        for layer in self.mid:
            x = layer(x)

        for layer in self.block:
            x = layer(x)

        for layer in self.end:
            x = layer(x)
        return x


# https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py#L213
class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding(buggy version)
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)


# https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py#L14
class VQModel(nn.Module):
    def __init__(self, n_e, e_dim, beta, input_channel, num_res, f=8):
        super().__init__()
        # essential blocks
        self.encoder = Encoder(input_channel, num_res, f)
        self.decoder = Decoder(input_channel, num_res, f)
        self.vq = VectorQuantizer2(n_e, e_dim, beta)

        # conv layer after quant
        self.quant_conv = torch.nn.Conv2d(input_channel // f, e_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(e_dim, input_channel // f, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        # z_q, loss, (perplexity, min_encodings, min_encoding_indices)
        return self.vq(h)

    def decode(self, x):
        quant = self.post_quant_conv(x)
        dec = self.decoder(quant)
        return dec

    def forward(self, x):
        quant, diff, (_, _, ind) = self.encode(x)
        dec = self.decode(quant)
        return dec, diff

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def init_from_ckpt(self, path, ignore_keys=None):
        self.load_state_dict(torch.load(path))
        print(f"Restored from {path}")