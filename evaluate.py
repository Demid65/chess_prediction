import numpy as np

import torch

from utils import extract_metadata, vectorize
from model import EvaluationModel

import warnings
import argparse

warnings.filterwarnings("ignore")

MODEL_PATH = 'model.pt'

# parse the command line arguments
parser = argparse.ArgumentParser(
                    prog='python evaluate.py',
                    description='Evaluates chess positions given Forsyth–Edwards Notation.',
                    epilog='https://github.com/Demid65/chess_prediction')

parser.add_argument('--model', type=str, metavar='MODEL_PATH', dest='MODEL_PATH',
                    help=f'Filename of the saved model. Defaults to {MODEL_PATH}', default=MODEL_PATH)

args = parser.parse_args()

MODEL_PATH = args.MODEL_PATH

fen = input()

model = EvaluationModel()
ckpt = torch.load(MODEL_PATH)
model.load_state_dict(ckpt)

data = vectorize(fen)
data = torch.tensor(data, dtype=torch.float).permute(2, 0, 1).cpu()
data = data[None, :]

meta = extract_metadata(fen)
meta = torch.tensor(meta, dtype=torch.float).cpu()
meta = meta[None, :]

model = model.cpu()

with torch.no_grad():
    model.eval()
    res = model(data, meta).item()
print(res)
