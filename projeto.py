# Bibliotecas

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, pipeline
from sklearn import preprocessing
from tqdm import tqdm

# Definicao de alguns parametros
MAX_LENGTH = 512

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

BACH_SIZE = 16

# Confihuracao de CPU/GPU

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print("Conferindo a unidade de processamento:", device)

# Leitura da base de dados

df = pd.read_csv('imdb-reviews-pt-br.csv')
print(f"Número de exemplos: {len(df)}")

## Vamos verificar se as categorias estao balanceadas está balanceada:

df["sentiment"].value_counts() # ta bem balanceada

## Fine-tuning

tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

