import torch
from torch.utils.data import Dataset

from config import NULL_CLASS
from training.masking import mixed_mask


class ASCIIDataset(Dataset):
    def __init__(self, data_tensor, labels=None, mask_fn=mixed_mask):
        """
        Args:
            data_tensor: [N, H, W] tensor of character indices (uint8 or long)
            labels: optional [N] tensor of class labels (NULL_CLASS for
                    unlabeled samples). None -> all NULL_CLASS.
            mask_fn: masking strategy — mixed_mask (default; learns free
                     generation, inpainting and upscaling together) or
                     random_mask (the original scheme)
        """
        self.data = data_tensor
        self.labels = labels
        self.mask_fn = mask_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Files store uint8 (vocab fits in a byte — 8x smaller on disk/RAM);
        # the embedding layer needs long indices
        grid = self.data[idx].long()  # [H, W]
        masked_grid, target_grid, mask, ratio = self.mask_fn(grid)
        label = (int(self.labels[idx]) if self.labels is not None
                 else NULL_CLASS)
        return (masked_grid, target_grid, mask,
                torch.tensor(ratio, dtype=torch.float),
                torch.tensor(label, dtype=torch.long))
