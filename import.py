"""
install the librairies
"""

# install pytorch biggan
!pip install pytorch-pretrained-biggan

from pytorch_pretrained_biggan import (BigGAN, one_hot_from_int, truncated_noise_sample,BigGANConfig)
import torchvision.utils as vutils
import numpy as np
import IPython.display
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import os
import torch
from torch import nn
from tqdm.notebook import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import pdb
import numpy as np
import random
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
%matplotlib inline
