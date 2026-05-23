# English to Italian transalation using HF Opus Books dataset - https://huggingface.co/datasets/Helsinki-NLP/opus_books/viewer/en-it

import torch
import torch.nn as nn
from torch.utils.data import Datset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel # in this project we will just do word level instead of BPE and other stuff
from tokenizers.trainers import WordLevelTrainer # this class trains the tokenizer on the dataset, i.e. build the vocabulary given a bunch of sentnces from source language and target language
from tokenizers.pre_tokenizers import Whitespace # split from white space

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


# creating the dataset, basically the tensors from the dataset that our model could use, in dataset.py file

