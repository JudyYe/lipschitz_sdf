import numpy as np
import torch
import torch.nn.functional as F
import ipyplot
from torch.utils.data import Dataset
import skfmm
from vis_utils import show_images

class ImageSdf(Dataset):
    def __init__(self, image_list, z_dim=1, nP=400):
        """
        Args:
            image_list (_type_): (N, H, W) of 0, 1
            nP (int, optional): _description_. Defaults to 400.
        """
        self.image_list = image_list = np.array(image_list)
        self.N, self.H, self.W = self.image_list.shape
        self.z_dim = z_dim

        phi = (image_list > 0).astype(np.float32) - 0.5  # outside: -0.5

        # self.edt_list = edt_list = [ndimage.distance_transform_edt(1-m) / max(self.H, self.W) * 2 for m in mask_list]
        edt_list = [-skfmm.distance(m, dx = 2/max(self.H, self.W)) for m in phi]
        self.edt = torch.FloatTensor(edt_list)

        self.nP = nP
        ipyplot.plot_images(image_list, )
        
        show_images(edt_list, True)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        point1,sdf1 = self.sample(idx)
        z1 = torch.FloatTensor(np.array([[idx]])).repeat(self.nP, self.z_dim)
        return point1, z1, sdf1
    
    def sample(self, idx):
        xy = torch.rand([self.nP, 2]) * 2 - 1
        sdf = F.grid_sample(
            self.edt[idx].view(1, 1, self.H, self.W), 
            xy.view(1, 1, self.nP, 2)
        ).view(self.nP, 1)
        return xy, sdf        