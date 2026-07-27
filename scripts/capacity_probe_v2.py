import sys, torch
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..'))
import training.train as T
T.COND_DROPOUT = 0.0   # patch the module global: every row keeps its caption

from model.ascii_bert import ASCIIBert, MODEL_SIZES

# 1,000 samples with UNIQUE captions: caption -> exactly one grid,
# so a perfect memorizer reaches CE ~ 0 even at 100% mask ratio.
d = torch.load('data/synthetic_v3.pt', map_location='cpu')
seen, keep = set(), []
for i, cid in enumerate(d['caption_ids'].tolist()):
    if cid not in seen:
        seen.add(cid); keep.append(i)
    if len(keep) == 1000:
        break
idx = torch.tensor(keep)
torch.save({'data': d['data'][idx], 'caption_ids': d['caption_ids'][idx],
            'captions': d['captions']}, 'data/probe_unique1k.pt')

model = ASCIIBert(**MODEL_SIZES['base']).to('cuda')
T.train_stage(model, 'data/probe_unique1k.pt', epochs=60, lr=3e-4,
              stage_name='memorize-u', device=torch.device('cuda'),
              ckpt_dir='checkpoints_capacity',
              save_checkpoints=False, epoch_samples=False)
