import torch
from torch.utils.data import Dataset

from config import TEXT_EMB_DIM
from training.masking import mixed_mask


class ASCIIDataset(Dataset):
    def __init__(self, data_tensor, caption_ids=None, caption_tokens=None,
                 caption_masks=None, mask_fn=mixed_mask):
        """
        Args:
            data_tensor: [N, H, W] tensor of character indices (uint8 or long)
            caption_ids: optional [N] long tensor indexing into the caption
                         tables (-1 = no caption for that sample)
            caption_tokens: optional [num_captions, L, TEXT_EMB_DIM] float —
                            per-token text embeddings of the unique captions
                            (see data/text_embed.embed_captions)
            caption_masks: optional [num_captions, L] bool validity masks
                           aligned with caption_tokens
            mask_fn: masking strategy — mixed_mask (default; learns free
                     generation, inpainting and upscaling together) or
                     random_mask (the original scheme)

        Each item is (masked_grid, target_grid, mask, ratio, cond_tokens,
        cond_mask, has_cond). Uncaptioned samples get zero tokens and an
        all-False mask with has_cond=False — the training loop turns those
        into the model's learned null token via cond_drop.
        """
        self.data = data_tensor
        self.caption_ids = caption_ids
        self.caption_tokens = caption_tokens
        self.caption_masks = caption_masks
        if caption_tokens is not None:
            self.ctx_shape = caption_tokens.shape[1:]  # (L, D)
        else:
            self.ctx_shape = (1, TEXT_EMB_DIM)
        self.mask_fn = mask_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Files store uint8 (vocab fits in a byte — 8x smaller on disk/RAM);
        # the embedding layer needs long indices
        grid = self.data[idx].long()  # [H, W]
        masked_grid, target_grid, mask, ratio = self.mask_fn(grid)

        cid = int(self.caption_ids[idx]) if self.caption_ids is not None else -1
        if cid >= 0 and self.caption_tokens is not None:
            cond_tokens = self.caption_tokens[cid].float()
            cond_mask = self.caption_masks[cid]
            has_cond = True
        else:
            cond_tokens = torch.zeros(self.ctx_shape)
            cond_mask = torch.zeros(self.ctx_shape[0], dtype=torch.bool)
            has_cond = False
        return (masked_grid, target_grid, mask,
                torch.tensor(ratio, dtype=torch.float),
                cond_tokens, cond_mask, torch.tensor(has_cond))
