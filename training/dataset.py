from torch.utils.data import Dataset

from training.masking import random_mask


class ASCIIDataset(Dataset):
    def __init__(self, data_tensor):
        """
        Args:
            data_tensor: [N, H, W] long tensor of character indices
        """
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Data files store uint8 (the vocab fits in a byte — 8x smaller
        # on disk and in RAM); the embedding layer needs long indices
        grid = self.data[idx].long()  # [H, W]
        masked_grid, target_grid, mask = random_mask(grid)
        return masked_grid, target_grid, mask
