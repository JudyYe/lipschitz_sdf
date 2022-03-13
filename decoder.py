
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import pytorch_lightning as pl
import wandb
import seaborn as sb

from lip_weight_norm import lip_weight_norm


class SdfDec(pl.LightningModule):
    def __init__(self, alpha=1, *args, **kargs) -> None:
        super().__init__()
        self.dec = Decoder(*args, **kargs)
        self.alpha = alpha
        self.weight_norm = self.dec.weight_norm

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer    

    def forward(self, x, z):
        y = self.dec(torch.cat([z, x], -1))
        return y
        
    def _step(self, train_batch, batch_idx):
        x1, z1, y1  = train_batch
        
        y1_hat = self(x1, z1)
        
        losses = {}
        sdf_loss = F.mse_loss(y1_hat, y1)
        losses['sdf'] = sdf_loss
        lip_reg = self.alpha * self.calc_lip()
        losses['lip'] = lip_reg

        loss = 0
        for k, v in losses.items():
            loss = v + loss
        return loss, losses


    def training_step(self, train_batch, batch_idx):
        loss, losses = self._step(train_batch, batch_idx)
        losses = {'train_%s' % k: v for k, v in losses.items() }

        # self.log('train_loss', loss, on_step=True)
        # self.log_dict(losses, on_step=True)
        try:
            self.logger.experiment.log(losses, step=self.global_step)
            self.logger.experiment.log({'train_loss': loss}, step=self.global_step)
        except:
            print('cannot log')
            pass
        return loss
    
    def validation_step(self, train_batch, batch_idx):
        x1, z1, y1  = train_batch

        loss, losses = self._step(train_batch, batch_idx)
        losses = {'val_%s' % k: v for k, v in losses.items() }

        # self.log('val_loss', loss, on_step=True)
        # self.log_dict(losses)
        self.logger.experiment.log(losses, step=self.global_step)
        self.logger.experiment.log({'val_loss': loss}, step=self.global_step)
        self.vis_z(z1[0:1])
    
    def vis_z(self, zz):
        """zz: (1, D)
        returns: (1, 1, H, W)
        """
        # display image
        device = zz.device
        W = H = 28
        x = torch.linspace(-1, 1, W, device=device)  # W
        y = torch.linspace(-1, 1, H, device=device)
        yy, xx = torch.meshgrid([y, x])  # (H, W? )
        zz = zz.repeat(H*W, 1)

        y = self(torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], -1), zz)
        y = y.reshape(1, 1, H, W)

        images = wandb.Image(make_grid(y), caption='z=%g' % zz[0, 0].item())
        try:
            self.logger.experiment.log({'sample': images}, step=self.global_step)
        except:
            print('did not log ', 'sample')
            pass

        # self.logger.log_image('sample', [make_grid(y)], caption=['z=%g' % zz[0,0].item()])
        # self.logger.exper('samples', [y], caption=['z=%g' % zz[0, 0].item()], step=self.global_step)
        # self.display_images(y)
        return y
        
    def calc_lip(self):
        model = self.dec
        if self.weight_norm == 'lip':
            # weight_c is a scaler, take their production
            # \prod |weight|_\inf
            loss = 1
            for name, param in model.named_parameters():
                if 'lin' in name and 'weight_c' in name:
                    loss = loss * param
        elif self.weight_norm == 'wn':
            # \sum |weight|_p
            loss = 0
            for name, param in model.named_parameters():
                if 'lin' in name and 'weight' in name:
                    loss = loss + torch.norm(param, 'inf')
        elif self.weight_norm == 'none':
            loss = 0
        else:
            raise NotImplementedError(self.weight_norm)
        return loss



    def display_images(self, images):
        """
        Args:
            xy (_type_): (N, 1, H, W)
        """
        images = images.cpu().detach()
        images = make_grid(images)[0].detach().numpy()
        images = sb.heatmap(images, vmin=0)

        # plt.show()
        
        
class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        inp_ch=3,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + inp_ch] + dims + [1]
        self.inp_ch = inp_ch

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= inp_ch

            if weight_norm == 'wn' and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            elif weight_norm == 'lip' and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    lip_weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (weight_norm == 'none')
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout

    # input: N x (L+3)
    def forward(self, input):
        """latent: N, L+3 where z,xyz"""
        xyz = input[:, -self.inp_ch:]

        if input.shape[1] > self.inp_ch and self.latent_dropout:
            latent_vecs = input[:, :-self.inp_ch]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and self.weight_norm == 'none'
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x

