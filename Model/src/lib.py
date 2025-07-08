import torch
import copy
import random
import os
import csv
import gc
import warnings
import argparse
import ast
import json
import sys
import pprint
import re
import wandb
import shutil
import itertools
import copy
import ot
import time
import powerlaw
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as tgnn
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import os.path as osp
import scipy.sparse as sp

from itertools import product
from typing import Callable, List, Optional, Union
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import random_split
from torch_geometric.nn import aggr
from torch_geometric.data import Data, HeteroData, InMemoryDataset, download_url, extract_zip, Batch
from torch_geometric.datasets import DBLP, IMDB
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.utils import to_networkx, to_dense_adj
from torch_geometric.transforms import RemoveDuplicatedEdges
from datetime import datetime
from sklearn import preprocessing

warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = [15, 15]
plt.rcParams["figure.autolayout"] = True

torch.set_printoptions(threshold=10000)