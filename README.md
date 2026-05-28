# Transformers From Scratch

A learning project where I implemented a Transformer model in PyTorch for English to Italian translation using the Hugging Face Opus Books dataset.

## What I built

- Token + positional embeddings
- Multi-head self-attention
- Encoder and decoder stacks
- Feed-forward blocks, residual connections, and layer normalization
- Training + checkpoint saving/loading
- Greedy decoding for inference
- Attention map visualization in notebooks

## Files

- `model.py`: Transformer architecture from scratch
- `dataset.py`: Dataset preprocessing, padding, masks, causal mask
- `train.py`: Training loop, validation preview, checkpointing
- `config.py`: Hyperparameters and file paths
- `rough.ipynb`: Concept experiments and tensor-shape checks
- `inference.ipynb`: Inference playground
- `attention_visual.ipynb`: Attention heatmap visualization

## Dataset

- Hugging Face dataset: `Helsinki-NLP/opus_books`
- Language pair: English (`en`) -> Italian (`it`)

## Quick run

```bash
source .venv/bin/activate
python train.py
```

## Notes

- Checkpoints are saved in `weights/`
- TensorBoard logs are written to `runs/`
- Resume training by setting `preload` in `config.py` (for example, `"04"`)

Prediction quality note: The translations are still quite rough. Full training looked like it would take days on my local CPU-only system (no GPU), so I reduced training from 20 epochs to 5 midway to keep it practical.
