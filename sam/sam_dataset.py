import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from typing import List, Tuple


class SamDataset(Dataset):
    def __init__(self, root_dir, mask_dir):
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.num_images = len(os.listdir(self.root_dir))

    def __len__(self):
        return self.num_images

    def read_mask(self, npy_path: str) -> np.array:
        with open(npy_path, 'rb') as f:
            mask = np.load(f)
        return mask

    def choose_rndm_coords(self, mask: np.array) -> List[Tuple[int]]:
        coords_X, coords_Y = np.where(mask > 0)
        i = np.random.choice(range(len(coords_X)))
        return [(coords_X[i], coords_Y[i])]

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, f"{idx}.jpg")
        mask_name = os.path.join(self.mask_dir, f"{idx}.npy")

        image = Image.open(img_name)
        mask = self.read_mask(mask_name)
        foreground_point = self.choose_rndm_coords(mask)

        return image, foreground_point, mask