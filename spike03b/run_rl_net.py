import torch.nn as nn
def make_actor(i=23, o=3):
    return nn.Sequential(nn.Linear(i, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, o))
