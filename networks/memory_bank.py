import torch
import torch.nn as nn


class MemoryBank(nn.Module):
    def __init__(self, dim, K, n_cls):
        super(MemoryBank, self).__init__()
        
        self.K = K
        
        self.register_buffer("queue", torch.randn(dim, K))
        self.register_buffer("q_label", torch.randint(n_cls, (1, K)))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity
        
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.q_label[:, ptr:ptr + batch_size] = labels.unsqueeze(1).T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
