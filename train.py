# English to Italian transalation using HF Opus Books dataset - https://huggingface.co/datasets/Helsinki-NLP/opus_books/viewer/en-it

import torch
import torch.nn as nn
from torch.utils.data import Datset, DataLoader, random_split

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel # in this project we will just do word level instead of BPE and other stuff
from tokenizers.trainers import WordLevelTrainer # this class trains the tokenizer on the dataset, i.e. build the vocabulary given a bunch of sentnces from source language and target language
from tokenizers.pre_tokenizers import Whitespace # split from white space

from torch.utils.tensorboard import SummaryWriter # for visualizing loss during training

from pathlib import Path

# function to get all sentences from the dataset for a given language, this will be used to train the tokenizer
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang] # item['transalation'] is essentally the transalation col in the dataset, which is a dict with keys as lang and its values as sentences

# building our own tokenizer
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # build the tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) # if our trainer encounters a word that is not in the vocabulary, it will replace it with [UNK] token
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2) # give the trainer special tokens which will be used during training, and min_freq means, for a word to be included in the vocabulary, it must appear at least 2 times in the dataset
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer) 
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# load datsset
def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train') # this train split is a feature available in original dataset and kinda is the only split available, so we need to write this to download the "full" dataset, and later we split it ourselves into train and val

    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # split the dataset into train and val
    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size]) # splits ds_raw into two datasets of size train_ds_size and val_ds_size


    # now we need to create the dataset, as in basically the tensors from the dataset that our model could use, done in dataset.py file  

    # now we can can make 2 datasets, one for train and one for val
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # calculating max_len from every sentence in source and target language in this particular dataset split, this will be used to set the seq_len for our model, if max len is say 100, then we can set seq_len to say 120
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))  

    print(f"Max length of source sentences: {max_len_src}")  
    print(f"Max length of target sentences: {max_len_tgt}")  


    # creating dataloaders for train and val datasets
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

# now building model
def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model']) 
    return model

# building training loop
def train_model(config):
    #define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # make sure weight folder is created
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # load dataset
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config,tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # tensorboard (visualize loss)
    writer = SummaryWriter(log_dir=config['experiment_name'])

    # define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # check if there is a checkpoint to load (in case of crashes, our model has ability to load saved weights and continue)
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get
    


