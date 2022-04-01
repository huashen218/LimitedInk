import torch
import random
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)    
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_tensor(x):
    return x.to(device) if isinstance(x, torch.Tensor) else torch.Tensor(x).to(device)    # convert to type: torch.cuda.FloatTensor


def to_longtensor(x):
    return x.to(device) if isinstance(x, torch.LongTensor) else torch.LongTensor(x).to(device)     # convert to type: torch.cuda.LongTensor


def iterator_toarray_batch(iterator):
    inputs = [tuple(t.cpu().detach().numpy() for t in batch) for batch in iterator]
    true_labels = [batch[1].cpu().detach().numpy().flatten() for batch in iterator]
    return inputs, true_labels