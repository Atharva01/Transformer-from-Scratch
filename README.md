# Transformer from Scratch

A PyTorch implementation of the Transformer architecture for neural machine translation, built from the ground up following the "Attention is All You Need" paper.

## Overview

This project implements a complete Transformer model for sequence-to-sequence translation tasks. The default configuration translates from English (en) to Dutch (nl) using the OPUS Books dataset.

## Features

- **Complete Transformer Architecture**: Multi-head attention, positional encoding, encoder-decoder structure
- **Custom Components**: All transformer components built from scratch using PyTorch
- **Bilingual Translation**: Supports language pair translation with configurable source and target languages
- **Training Pipeline**: Full training loop with TensorBoard integration for monitoring
- **Tokenization**: WordLevel tokenizer with special tokens (SOS, EOS, PAD, UNK)

## Project Structure

```
├── model.py       # Transformer architecture implementation
├── dataset.py     # Bilingual dataset and data processing
├── train.py       # Training loop and model initialization
└── config.py      # Configuration parameters
```

## Key Components

- **Input Embeddings & Positional Encoding**: Convert tokens to embeddings with position information
- **Multi-Head Attention**: Self-attention and cross-attention mechanisms
- **Encoder**: Stack of encoder blocks with self-attention and feed-forward layers
- **Decoder**: Stack of decoder blocks with masked self-attention and cross-attention
- **Feed-Forward Network**: Position-wise fully connected layers
- **Layer Normalization & Residual Connections**: For stable training

## Configuration

Default hyperparameters (configurable in `config.py`):

- Model dimension: 512
- Sequence length: 350
- Batch size: 8
- Learning rate: 1e-4
- Number of epochs: 20
- Number of layers: 6
- Number of attention heads: 8

## Usage

```bash
python train.py
```

The model will:

1. Download and prepare the OPUS Books dataset
2. Build or load tokenizers for source and target languages
3. Train the transformer model
4. Save model checkpoints in the `weights/` directory
5. Log training metrics to TensorBoard

## Requirements

- PyTorch
- Hugging Face Datasets
- Tokenizers
- TensorBoard
- tqdm

## Model Checkpoints

Model weights are saved after each epoch in the format: `weights/tmodel_XX.pt`
