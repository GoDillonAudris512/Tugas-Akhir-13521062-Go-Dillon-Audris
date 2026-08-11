# Preprocessing code adap   d from:
# Ding et al. (2020), "CogLTX: Applying BERT to Long Texts"
# https://github.com/Sleepychord/CogLTX
# License: MIT

import pickle
import logging
import pandas as pd
import numpy as np
import random

from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer
from sklearn.utils import Bunch
from joblib import dump, load

from external.cogltx.buffer import Buffer
from src.config import DEFAULT_MODEL_NAME

root_dir = Path.cwd()
data_dir = root_dir / 'data' / 'imbalanced'
# data_dir = root_dir / 'data' / 'balanced'
data_dir.mkdir(exist_ok=True)

# Build dataset from raw text files and metadata
def build_dataset(metadata_filename):
    # Read metadata 
    metadata_df = pd.read_csv(data_dir / metadata_filename)
    metadata_df.columns = metadata_df.columns.str.lower()

    # Define target names
    target_names = ["negative", "neutral", "positive"]
    target_to_id = {name: idx for idx, name in enumerate(target_names)}

    # Read text from files and combine with metadata to create dataset (using sklearn.utils.Bunch)
    data = []
    targets = []
    filenames = []
    folds = []
    
    for row in metadata_df.itertuples():
        
        filename = f"{row.ticker[:4]}_MDA_{row.year}.txt"
        file_path = data_dir / 'MDA text' / filename
        
        if file_path.exists():
            text = file_path.read_text(encoding='utf-8')
            
            data.append(text)
            targets.append(target_to_id[row.yoy_close_category])
            filenames.append(filename)
            folds.append(row.ticker_fold)
        else:
            print(f"Warning: File {file_path} does not exist. Skipping this entry.")
    
    dataset = Bunch()
    dataset.data = data
    dataset.target = targets
    dataset.target_names = target_names
    dataset.filename = filenames
    dataset.fold = folds

    return dataset

# Fetch dataset from cache or build it from raw files
def fetch_dataset(cache_filename, metadata_filename, force_reload=False):
    cache_path = data_dir / cache_filename

    if cache_path.exists() and not force_reload:
        print(f"Loading dataset from cache: {cache_path}")
        return load(cache_path)

    print("Reading raw dataset files...")
    dataset = build_dataset(metadata_filename)
    dump(dataset, cache_path)

    return dataset

# Preprocess the dataset, transforming text into tokenized blocks
def preprocess(dataset, dataset_type, fold):
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    count, batches = 0, []
    
    for i in tqdm(range(len(dataset.data))):
        data, label = dataset.data[i], dataset.target[i]
        label_name = dataset.target_names[label]

        qbuf, count = Buffer.split_document_into_blocks([tokenizer.cls_token], tokenizer, count, hard=False, properties=[('label_name', label_name), ('label', label), ('_id', str(i)), ('blk_type', 0), ('filename', dataset.filename[i]), ('fold', dataset.fold[i])])
        dbuf, count = Buffer.split_document_into_blocks(tokenizer.tokenize(data), tokenizer, count, hard=False)
        
        batches.append((qbuf, dbuf))
    
    with open(data_dir / f'mda_{dataset_type}_fold_{fold}.pkl', 'wb') as fout: 
        pickle.dump(batches, fout)
    
    return batches

if __name__ == "__main__":
    # Fetch dataset
    mda_dataset = fetch_dataset('mda_dataset_imbalanced.joblib', 'mda_manual_v20251224_with_fold.csv')

    # Split dataset into train and val sets with respect to their folds
    for fold_num in range(10):
        print(f"Processing fold {fold_num}...")

        train_indices = [i for i, f in enumerate(mda_dataset.fold) if f != fold_num]
        val_indices = [i for i, f in enumerate(mda_dataset.fold) if f == fold_num]

        train_dataset = Bunch(
            data=[mda_dataset.data[i] for i in train_indices],
            target=[mda_dataset.target[i] for i in train_indices],
            target_names=mda_dataset.target_names,
            filename=[mda_dataset.filename[i] for i in train_indices],
            fold=[mda_dataset.fold[i] for i in train_indices],
        )
        val_dataset = Bunch(
            data=[mda_dataset.data[i] for i in val_indices],
            target=[mda_dataset.target[i] for i in val_indices],
            target_names=mda_dataset.target_names,
            filename=[mda_dataset.filename[i] for i in val_indices],
            fold=[mda_dataset.fold[i] for i in val_indices],
        )
        
        # Preprocess train and val datasets
        preprocess(train_dataset, 'train', fold_num)
        preprocess(val_dataset, 'val', fold_num)
