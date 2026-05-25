# creating tensors from dataset that our model will use

from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None: # ds is dataset downloaded from hugginface, tokenizer_src and tokenizer_tgt are the tokenizers for source and target language respectively, src_lang and tgt_lang are the language codes for source and target language respectively, seq_len is the maximum sequence length for our model, we will pad or truncate the sentences to this length
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # get the token ids for special tokens, which will be used during training and inference
        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype = torch.int64) # could have used tokenizer_tgt as well, since both source and target language will have same special tokens
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype = torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype = torch.int64)

    def __len__(self):
        return len(self.ds) # return the number of samples in the dataset
    
    def __getitem__(self, index: Any) -> Any:
        # extracting original pair from hf dataset
        src_target_pair = self.ds[index]

        # now sourfce text and target text
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # converting each text to tokens and tokens to input ids in the vocanbulary
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids # returns a list of ids correspoinding to each token in the text
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # padding to seq_len for shorter sentences
        # calculating how many padding tokens we need to add for encoder side and decoder side to reach seq_len
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # we need to subtract 2 because we will add [SOS] and [EOS] tokens to the input, so we need to reserve space for them
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # we only need to subtract 1 for decoder side because we will only add [SOS] token at the beginning of the input, and not [EOS] token at the end, because during training, we will use teacher forcing, which means we will feed the target sentence as input to the decoder, so we don't need to add [EOS] token at the end of the input, because the model will learn to predict it at the end of the output

        # handling truncation for longer sentences
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")


        # creating 3 tensors, one for encoder input and one for deocder input, and one for output for decoder, we will call it "label" here

        # 1. add SOS and EOS to source text
        encoder_input = torch.cat( # concatenate
            [
                self.sos_token, # add [SOS] token at the beginning of the input
                torch.Tensor(enc_input_tokens, dtype = torch.int64), # convert list of input ids to tensor
                self.eos_token, # add [EOS] token at the end of the input
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64) # add padding tokens at the end of the input to reach seq_len
            ]
        )

        # 2. add SOS to decoder input, but not EOS, because during training, we will use teacher forcing, which means we will feed the target sentence as input to the decoder, so we don't need to add [EOS] token at the end of the input, because the model will learn to predict it at the end of the output
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.Tensor(dec_input_tokens, dtype = torch.int64), 
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )       

        # add EOS to the label (what we expect as output from decoder)
        label = torch.cat(
            [
                torch.Tensor(dec_input_tokens, dtype = torch.int64), 
                self.eos_token, # add [EOS] token at the end of the output, because during training, we will use teacher forcing, which means we will feed the target sentence as input to the decoder, so we need to add [EOS] token at the end of the output, because the model will learn to predict it at the end of the output
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64) 
            ]
        )

        # making sure we reach seq_len
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len) # we added padding tokens, but we dont want it to participare in attention mechanism, so creating a mask which essentially says all tokens in encoder input which are not padding tokens are cool and good to go, and unsqueezing 2 times to make it compatible with the shape thats used in attention mechnanism, the (batch_size, num_heads, seq_len) thing
            # for decoder, we need to avoid padding otkens as usual, but we alos need a casual mask, which is, avoid seeing words which are ahead and only look at previous words
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            "label": label, # (seq_len)
            "src_text": src_text, # original source text, just for reference, not used in training, for visualization
            "tgt_text": tgt_text 
        }

# making upper traingle matrix = -inf so softmax makes it 0 thing
def causal_mask(size): 
    # gives every value in upper triangle and rest all become 0
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int) # (1, seq_len, seq_len)
    # now we make it so that everyhting that is 0 becomes True
    return mask == 0





