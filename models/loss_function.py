import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        z = torch.cat([z_i, z_j], dim=0)  
        z = F.normalize(z, dim=1) 
        sim = torch.matmul(z, z.T)  
        sim = sim / self.temperature
        
        # Create mask to exclude self-similarity
        batch_size = z_i.size(0)
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim = sim.masked_fill(mask, float('-inf'))

        # Positive pairs
        targets = torch.arange(batch_size, device=z.device)
        targets = torch.cat([targets + batch_size, targets]) 
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(sim, targets)
        return loss

