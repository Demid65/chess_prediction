from urllib.request import urlopen
from io import BytesIO
from gzip import GzipFile
import os
from tqdm import tqdm
from shutil import copyfileobj
import warnings
import argparse

warnings.filterwarnings("ignore")

DATASET_LINK = 'https://storage.googleapis.com/chesspic/datasets/2021-07-31-lichess-evaluations-37MM.db.gz'
EXTRACT_PATH = 'dataset.db'

# parse the command line arguments
parser = argparse.ArgumentParser(
                    prog='python download_dataset.py',
                    description='downloads the dataset for chess evaluation model training.',
                    epilog='https://github.com/Demid65/chess_prediction')
                    
parser.add_argument('--link', type=str, metavar='DATASET_LINK', dest='DATASET_LINK',
                    help=f'Link to the dataset. Defaults to {DATASET_LINK}', default=DATASET_LINK)

parser.add_argument('--extract_to', type=str, metavar='EXTRACT_PATH', dest='EXTRACT_PATH',
                    help=f'File where dataset is extracted to. Defaults to {EXTRACT_PATH}', default=EXTRACT_PATH)

args = parser.parse_args()

DATASET_LINK = args.DATASET_LINK
EXTRACT_PATH = args.EXTRACT_PATH

# Download and unzip the dataset
def download_and_unzip(url, extract_to):
    http_response = urlopen(url)
    zipfile = GzipFile(fileobj=http_response)
    with open(EXTRACT_PATH, 'wb') as f:
        copyfileobj(zipfile, f)
    

print(f'downloading from {DATASET_LINK}')
download_and_unzip(DATASET_LINK, extract_to=EXTRACT_PATH)

print('done')