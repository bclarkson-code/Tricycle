"""
Training script for training a SmolGPT model on the complete
works of shakespeare.

The hyperparams for this model are very much a work in progress
"""

from tricycle.configs import SmolGPTConfig
from tricycle.model import GPT

config = SmolGPTConfig()
model = GPT(config)
