# practical ref (this .py file) - https://www.youtube.com/watch?v=ISNdQcPhsts&t=9085s

# theory ref (campusx) - https://www.youtube.com/watch?v=BjRVS2wTtcA&list=PLkBMe2eZMRQ2VKEtoL0GVUrNzEiXfgj07

# Attention is all you need paper - https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf

import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model  # dimension of the embedding vector for each token, 512 in the paper
        self.vocab_size = vocab_size  # number of unique tokens in the dataset
        self.embedding = nn.Embedding(vocab_size, d_model) # see rough.ipynb to understand what nn.Embedding does

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  # scale the embeddings by sqrt(d_model), what section 3.4 of attention is all you need paper says, bcoz internally it trains at O(1/sqrt(d_model)) rate, so we need to scale it up by sqrt(d_model) to make the training more stable.
    
class PositionalEncoding(nn.Module):
    # see rough.ipynb to understand some parts here

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None: # seq_len is the maximum sequence length that the model can handle (context length of a model), we need this to create the positional encodings for all positions up to seq_len. 
    # imagine seq_len as number of words in a sentence, lets assume 3 for this file
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len 
        self.dropout = nn.Dropout(dropout)

        # NOTE: in paper, the formula is:
        # PE(pos, 2i) = sin(pos / (10000^(2i/d_model)))
        # PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))
        # where pos is the position of the token in the sequence (0-indexed), i is the dimension of the embedding vector (0-indexed). However, in following code, we use a very slightly diff formula using exp and log "which has more stable numbers". # whatver that means

        # create a matrix of shape (seq_len, d_model) to hold the positional encodings for all positions up to seq_len
        pe = torch.zeros(seq_len, d_model)

        # create a vector of shape (seq_len,1)
        position = torch.arange(0,seq_len).unsqueeze(1)  # shape: (seq_len, 1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # shape: (d_model/2,), and div_term is basically just denominator of the formula

        # apply sine to even positions
        pe[:, 0::2] = torch.sin(position * div_term)  # shape: (seq_len, d_model/2)

        # apply cosine to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)  # shape: (seq_len, d_model/2)

        pe = pe.unsqueeze(0)  # shape: (1, seq_len, d_model), we add a batch dimension at the beginning, so that we can easily add it to the input embeddings later, which will also have a batch dimension at the beginning. 
        # imagine having a batch as having multiple sentences (in this file, lets imagine the value as 2), so during code they will be there, but to understand, ignore this dimension and see the other two and imagine them as number of words in a sentence and dimension of embedding vector for each word.

        # register pe as a buffer, which means it will be saved and loaded with the model, but it is not a parameter that will be updated during training
        self.register_buffer('pe', pe)

    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # add the positional encodings to the input embeddings, we only take the first x.size(1) positions from pe, because x might have a shorter sequence length than seq_len, and we set requires_grad to False because we don't want to update the positional encodings during training
        return self.dropout(x) # apply dropout to stop overfitting, section 5.4 of paper

        # to understand, assume there is a sentence with 3 tokens of 512 dim each so x has shape (3,512). pe has shape (1,3,512) (1 is batch dim), so x.shape[1] = 3, so pe[:, :x.shape[1], :] is is of shape (1,3,512), so we can add it to x which is of shape (3,512) and the result will be of shape (3,512) which is what we want.  

# Building encoder

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied, the gamma 
        self.bias = nn.Parameter(torch.zeros(1)) # additive, the beta

    # see rough.ipynb for understanding following
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # shape: (batch_size, seq_len, 1)
        std = x.std(dim=-1, keepdim=True) # shape: (batch_size, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    # we know that input emb and output emb are 512 each, and from paper, inner layer has dimension d_ff = 2048 neurons

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # first linear layer to expand the dimension from d_model to d_ff, the W1 and B1 from paper
        self.dropout = nn.Dropout(dropout) 
        self.linear2 = nn.Linear(d_ff, d_model) # second linear layer to project back to d_model, the W2 and B2 from paper

    def forward(self, x):
        x = self.linear1(x) # shape: (batch_size, seq_len, d_ff), from (batch_size, seq_len, d_model)
        x = torch.relu(x) # apply ReLU activation function
        x = self.dropout(x) 
        x = self.linear2(x) # shape: (batch_size, seq_len, d_model)
        return x
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None: # h is number of heads, 8 in the paper, so each head will have dimension d_k = d_model/h = 512/8 = 64
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h" # making sure d_model is dividible by h

        # we will split the d_model dimension into h heads, so each head will have dimension d_k = d_model/h = 512/8 = 64 in the paper
        self.d_k = d_model // h

        # we need to create linear layers for query, key and value for each head, so we create a single linear layer for each of them that will project the input from d_model to d_model (which is same as h * d_k), and then we will split the output into h heads later. 
        self.W_q = nn.Linear(d_model, d_model) # shape: (d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model) # shape: (d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model) # shape: (d_model, d_model)

        # we also need a final linear layer to project the concatenated output of all heads back to d_model
        self.W_o = nn.Linear(d_model, d_model) # shape: (d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1] # dimension of each head, 64 in the paper

        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1) # shape: (batch_size, h, seq_len, seq_len), we apply softmax to the last dimension which is the seq_len dimension of the key, so that we get the attention weights for each token in the sequence

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores # first part is weighted sum of value vectors,the last part is just for visualization


    # see rough.ipynb
    def forward(self, q, k, v, mask): # mask is for decoder, doing masked multi head attentiona and setting the upper triangle values from Q*K^T/root(d_k) before doing softmax

    # here, in encoder, for the first time, q = k = v = x (the first input emb), as we move on the architecture, they will the output of the previous layer, and they have shape (batch_size, seq_len, d_model). we just name it differently bcoz in decoder we do cross attention and q comes from the previous layer of decoder and k and v come from the output of encoder, so they are different, and this one function handles all scenarios.

        # following code is for the linear projections of query, key and value (the dot product of q * W_q, etc.), we are going from (batch_size, seq_len, d_model) to (batch_size, seq_len, d_model) for each of them, and then we will split them into h heads later. we are doing this for all heads together to be more efficient, instead of doing it separately for each head.
        query = self.W_q(q) # going from (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model) = (2, 3, 512)
        key = self.W_k(k) # going from (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model) = (2, 3, 512)
        value = self.W_v(v) # going from (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model) = (2, 3, 512)

        # now we need to split the query, key and value into h heads, so we will reshape them to (batch_size, seq_len, h, d_k) and then transpose to (batch_size, h, seq_len, d_k) so that we can do the attention calculation for each head separately.

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) # shape: (batch_size, h, seq_len, d_k) = (2, 8, 3, 64)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2) # shape: (batch_size, h, seq_len, d_k) = (2, 8, 3, 64)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2) # shape: (batch_size, h, seq_len, d_k) = (2, 8, 3, 64)

        # SOOOO essentially, there aren't 8 different sets of W_q, W_k, W_v that we define. We divided 3 words of 512 dim each to 8 heads which has 3 words each of each 64 dim, so its like we take input vector say 3*512, dot product with W_q of 512*512 to get query evctor of 3*512, take 8 cols of 3*64 each
        # We have ONE W_q (512×512) that projects 3 words from 512-dim to 512-dim, and then we SPLIT that 512-dim output into 8 heads of 64-dim each, so each head sees all 3 words but only a 64-dim slice of the representation.


        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout) # returns weighted sum of value vectors, shape of x: (batch_size, h, seq_len, d_k) = (2, 8, 3, 64)

        # now concat and multiply by W_o to get back to (batch_size, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # shape: (batch_size, seq_len, d_model) = (2, 3, 512)
        return self.W_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNormalization() 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # x is the input tensor (either embeddings for layer 1, or previous layer's output for subsequent layers)
        # sublayer is a function (either multiheadattention block or feedforward block, the 2 blocks which are inside the encoder block) that will be applied to x
        return x + self.dropout(sublayer(self.norm(x))) 

        # in the paper, order is: SUBLAYER → ADD → NORM (Post-LN)
        # here, order is: NORM → SUBLAYER → DROPOUT → ADD (Pre-LN)
        # Pre-LN is a common modern variation — it normalizes the input BEFORE the sublayer (attention or feedforward) rather than after the residual add.
        # this improves training stability because gradients flow more cleanly through the residual path, and avoids the need for careful learning rate warmup.
        # most modern LLMs (GPT-2, GPT-3 etc.) use Pre-LN over the original paper's Post-LN.

# Now creating Encoder Block which has 2 blocks inside, multiheadattention block (multiheadatttention + add and norm) and feedforward block (feedforward + add and norm)

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # we have 2 residual connections, one for each block inside the encoder block

    def forward(self, x, src_mask): # source mask is to prevent any interaction between padding tokens and actual tokens, we apply it to the input of the ancoder

        # 1. multiheadattention block with residual connection
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # for self attention, q=k=v=x (in deocder, q comes from decoder and k and v from encoder)
        # 2. feedforward block with residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
# noe we can multiple encoder blocks, 6 in the paper

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None: # layers = encoder blocks 
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization() 

    def forward(self, x, src_mask):
        # passing the input through each encoder block one by one, where the output of each block becomes the input to the next, and normalize at the end
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


## DECODER

# decoder block