import torch
from torch.utils.data import Dataset

from config import TEXT_EMB_DIM
from training.masking import mixed_mask


class ASCIIDataset(Dataset):
    def __init__(self, data_tensor, caption_ids=None, caption_embs=None,
                 mask_fn=mixed_mask):
        """
        Args:
            data_tensor: [N, H, W] tensor of character indices (uint8 or long)
            caption_ids: optional [N] long tensor indexing into caption_embs
                         (-1 = no caption for that sample)
            caption_embs: optional [num_captions, TEXT_EMB_DIM] float tensor
                          of precomputed text embeddings (see
                          data/text_embed.py)
            mask_fn: masking strategy — mixed_mask (default; learns free
                     generation, inpainting and upscaling together) or
                     random_mask (the original scheme)

        Each item is (masked_grid, target_grid, mask, ratio, cond_emb,
        has_cond). Uncaptioned samples get a zero cond_emb and
        has_cond=False — the training loop turns those into the model's
        learned null embedding via cond_drop.
        """
        self.data = data_tensor
        self.caption_ids = caption_ids
        self.caption_embs = caption_embs
        self.cond_dim = (caption_embs.shape[1] if caption_embs is not None
                         else TEXT_EMB_DIM)
        self.mask_fn = mask_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Files store uint8 (vocab fits in a byte — 8x smaller on disk/RAM);
        # the embedding layer needs long indices
        grid = self.data[idx].long()  # [H, W]
        masked_grid, target_grid, mask, ratio = self.mask_fn(grid)

        cid = int(self.caption_ids[idx]) if self.caption_ids is not None else -1
        if cid >= 0 and self.caption_embs is not None:
            cond_emb = self.caption_embs[cid].float()
            has_cond = True
        else:
            cond_emb = torch.zeros(self.cond_dim)
            has_cond = False
        return (masked_grid, target_grid, mask,
                torch.tensor(ratio, dtype=torch.float),
                cond_emb, torch.tensor(has_cond))
