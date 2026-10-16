"""Trainable native ROI identity projection; no query/task or GT runtime input."""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


class TemporalAppearance(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(2560),nn.Linear(2560,128),nn.GELU(),nn.Linear(128,64))

    def forward(self,features):
        return F.normalize(self.net(features),dim=-1)


class AppearanceProjector:
    def __init__(self,checkpoint):
        self.checkpoint=str(Path(checkpoint).resolve())
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['architecture']=='native-temporal-identity-2560-128-64-v1'
        self.model=TemporalAppearance();self.model.load_state_dict(saved['state_dict']);self.model.eval()
        self.training_steps=saved['step']

    def __call__(self,vectors):
        # Keep native input copies bounded independently of track population.
        outputs=[]
        with torch.inference_mode():
            for begin in range(0,len(vectors),16):
                inputs=torch.tensor(vectors[begin:begin+16],dtype=torch.float32)
                outputs.append(self.model(inputs))
        return torch.cat(outputs).numpy() if outputs else torch.empty(0,64).numpy()
