import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import torch.optim as optim
from torch import nn

from utils import extract_metadata, vectorize
from model import EvaluationModel

import warnings
import argparse

warnings.filterwarnings("ignore")

DATASET_PATH = 'dataset.db'
OUTPUT_PATH = 'model.pt'
TRAIN_LR = 0.05
TRAIN_EPOCH = 40
BATCH_SIZE = 512
SAMPLE_SIZE = 850000
PROPORTION = 0.2

# parse the command line arguments
parser = argparse.ArgumentParser(
                    prog='python train.py',
                    description='Runs the training process for chess evaluation model',
                    epilog='https://github.com/Demid65/chess_prediction')

parser.add_argument('--dataset', type=str, metavar='DATASET_PATH', dest='DATASET_PATH',
                    help=f'Path to the dataset. Defaults to {DATASET_PATH}', default=DATASET_PATH) 

parser.add_argument('--save_to', type=str, metavar='OUTPUT_PATH', dest='OUTPUT_PATH',
                    help=f'Filename of to save the model to. Defaults to {OUTPUT_PATH}', default=OUTPUT_PATH) 
                    
parser.add_argument('--sample_size', type=int, metavar='SAMPLE_SIZE', dest='SAMPLE_SIZE',
                    help=f'Filename of the saved model. Defaults to {SAMPLE_SIZE}', default=SAMPLE_SIZE)

parser.add_argument('--proportion', type=float, metavar='PROPORTION', dest='PROPORTION',
                    help=f'Proportion of dataset sample to be used for evaluation. Defaults to {PROPORTION}', default=PROPORTION)

parser.add_argument('--train_lr', type=float, metavar='TRAIN_LR', dest='TRAIN_LR',
                    help=f'Learning rate for the training. Defaults to {TRAIN_LR}', default=TRAIN_LR)

parser.add_argument('--train_epoch', type=int, metavar='TRAIN_EPOCH', dest='TRAIN_EPOCH',
                    help=f'Number of epochs for the training. Defaults to {TRAIN_EPOCH}', default=TRAIN_EPOCH)

parser.add_argument('--batch_size', type=int, metavar='BATCH_SIZE', dest='BATCH_SIZE',
                    help=f'Batch size for training process. Defaults to {BATCH_SIZE}', default=BATCH_SIZE)

args = parser.parse_args()

DATASET_PATH = args.DATASET_PATH
OUTPUT_PATH = args.OUTPUT_PATH
TRAIN_LR = args.TRAIN_LR
TRAIN_EPOCH = args.TRAIN_EPOCH
BATCH_SIZE = args.BATCH_SIZE
SAMPLE_SIZE = args.SAMPLE_SIZE
PROPORTION = args.PROPORTION

def train(
    model,
    optimizer,
    scheduler,
    loss_fn,
    train_loader,
    val_loader,
    epochs=1,
    device="cpu",
    ckpt_path="best.pt",
):
    # best score for checkpointing
    best = 0
    
    # iterating over epochs
    for epoch in range(epochs):
        # training loop description
        train_loop = tqdm(
            enumerate(train_loader, 0), total=len(train_loader), desc=f"Epoch {epoch}"
        )
        model.to(device)
        model.train()
        train_loss = 0.0
        # iterate over dataset 
        for i, data in train_loop:
            inputs, meta, labels = data
            inputs, meta, labels = inputs.to(device), meta.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass and loss calculation
            outputs = model(inputs, meta)
            
            labels = torch.squeeze(labels)
            outputs = torch.squeeze(outputs)
            
            loss = loss_fn(outputs, labels)

            # backward pass
            loss.backward()

            # optimizer run
            optimizer.step()

            train_loss += loss.item()
            train_loop.set_postfix({"loss": train_loss/(i+1)})
        
        # validation
        errors = None
        
        with torch.no_grad():
            eval_loss = 0.0
            model.eval()  # evaluation mode
            val_loop = tqdm(enumerate(val_loader, 0), total=len(val_loader), desc="Val")
            for i, data in val_loop:
                inputs, meta, labels = data
                inputs, meta, labels = inputs.to(device), meta.to(device), labels.to(device)

                outputs = model(inputs, meta)
                labels = torch.squeeze(labels)
                outputs = torch.squeeze(outputs)
                
                loss = loss_fn(outputs, labels)

                eval_loss += loss.item()
                

            score = (i+1) / eval_loss
            print(f'eval_loss: {eval_loss / (i+1)}')

            if score > best:
                torch.save(model.state_dict(), ckpt_path)
                best = score
                
            scheduler.step(eval_loss / (i+1))

print(f'Reading dataset from {DATASET_PATH}')
sql = sqlite3.connect(DATASET_PATH)
df = pd.read_sql_query(f'SELECT * FROM evaluations LIMIT {SAMPLE_SIZE}', sql)

print('Processing dataset')
vecs = []
meta = []
eval = []
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    vecs.append(vectorize(row['fen']))
    meta.append(extract_metadata(row['fen']))
    eval.append(row['eval'])

X = torch.tensor(np.array(vecs), dtype=torch.float).permute(0, 3, 1, 2)
m = torch.tensor(np.array(meta), dtype=torch.float)
y = torch.tensor(np.array(eval), dtype=torch.float)

processed_dataset = TensorDataset(X, m, y)

# set proportion and split dataset into train and validation parts
train_dataset, val_dataset = random_split(processed_dataset, [1-PROPORTION, PROPORTION])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print('Starting training')

model = EvaluationModel()
loss_fn = nn.L1Loss()
device = 'cuda' if torch.cuda.is_available else 'cpu'

optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.25)

train(
    model,
    optimizer,
    scheduler,
    loss_fn,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    epochs=TRAIN_EPOCH,
    ckpt_path=OUTPUT_PATH
)

print('done')
