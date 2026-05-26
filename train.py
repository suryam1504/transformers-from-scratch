# English to Italian transalation using HF Opus Books dataset - https://huggingface.co/datasets/Helsinki-NLP/opus_books/viewer/en-it

import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from config import get_weights_file_path, get_config

from datasets import load_dataset
from tqdm import tqdm
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
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device) # we will ignore the loss for padding tokens, because they are just there to make the input length equal to seq_len, and they don't carry any meaning, so we don't want our model to learn anything from them
    
    for epoch in range(initial_epoch, config['num_epochs']):
        # model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:

            model.train() # we are putting it here instead of above so that after every validation the model is back into its training mode

            encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch, 1, seq_len, seq_len)

            # run the tensors through transfomer
            encoder_output = model.encode(encoder_input, encoder_mask) # )batch, seq_len, d_model) 
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch, seq_len, tat_vocab_size)

            label = batch['label'].to(device) # (batch, seq_len)

            # # (batch, seq_len, tat_vocab_size) -> (batch*seq_len, tat_vocab_size) 
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)) # we need to reshape the output and label to be of shape (batch*seq_len, vocab_size) and (batch*seq_len) respectively, because CrossEntropyLoss expects the input to be of shape (N, C) where N is the number of samples and C is the number of classes, and the target to be of shape (N) where each value is the class index
            batch_iterator.set_postfix({f'loss': f'{loss.item():6.3f}'})

            # log the loss to tensorboard
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.flush()

            # back
            loss.backward()

            # update weights
            optimizer.step()
            optimizer.zero_grad()

            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

            global_step += 1

        # save model at the end of each epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(), # all weights of model
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


# function for greedy decoding, which will be used during validation to visualize the translations our model is generating
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, devices):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # pre compute encoder output and reuse it for every token we get from decoder
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with sos token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(devices) # (1,1) because one is for batch and one is for decoder input
    while True: # keep asking decoder to give tokens until it gives eos token or reaches max_len
        if decoder_input.size(1) == max_len:
            break
        # build mask for target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source).to(devices) 
        # calculate decoder output
        decoder_output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        # get next token
        prob = model.project(decoder_output[:,-1]) # (1, vocab_size) we only care about the last token in the decoder output, because that is the one we will use to predict the next token
        # greedy search, get the token with the highest probability
        _, next_word = torch.max(prob, dim=1) # (1) this will give us the index of the token with the highest probability
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(devices)], dim=1) # (1, seq_len) we will keep adding the predicted tokens to the decoder input until we get eos token or reach max_len
        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0) # return the predicted token id


# to visualize model while it is getting trained
console_width = 80
def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []

    # size of control window (just use default value)
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device) # (1, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (1, 1, 1, seq_len)

            assert encoder_input.size(0) == 1, "Batch size should be 1 for validation" # making sure we are using batch size of 1 for validation, so we can visualize each example separately

            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device) # (seq_len) this will give us the predicted token ids
            source_text = batch['src_text'][0] # original source text
            target_text = batch['tgt_text'][0] # original target text
            model_output_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy()) # decode the predicted token ids to get the predicted text

            # print to console
            print_msg('-'*console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_output_text}')

            if count == num_examples:
                break


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)


