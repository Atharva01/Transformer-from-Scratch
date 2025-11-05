from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import random_split, DataLoader, Dataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import BilingualDataset
from model import build_transformer
from config import *
from tqdm import tqdm
import warnings


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset(
        'opus_books', f'{config['lang_src']}-{config['lang_tgt']}', split="train")

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Filter out sentences that are too long
    max_len = config['seq_len'] - 2  # Account for SOS and EOS tokens
    filtered_ds = []
    for item in ds_raw:
        src_ids = tokenizer_src.encode(
            item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(
            item['translation'][config['lang_tgt']]).ids
        if len(src_ids) <= max_len and len(tgt_ids) <= max_len - 1:  # Target needs room for EOS only
            filtered_ds.append(item)

    print(
        f"Filtered dataset size: {len(filtered_ds)} (from {len(ds_raw)} original samples)")

    # Keep 90 % for training and 10% for validation
    train_ds_size = int(0.9 * len(filtered_ds))
    val_ds_size = len(filtered_ds) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(
        filtered_ds, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt,
                                config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt,
                              config['lang_src'], config['lang_tgt'], config['seq_len'])

    train_dataloader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len,
                              config['seq_len'], config['seq_len'], config['d_model'])
    return model


def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(
        config)
    model = get_model(config, tokenizer_src.get_vocab_size(),
                      tokenizer_tgt.get_vocab_size())

    model = model.cuda()
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading Model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id(
        '[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(
            train_dataloader, desc=f'Processing epoch {epoch}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # Run the tensors through the Transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask)
            # (B, seq_len, tgt_vocab_size)
            proj_out = model.project(decoder_output)

            label = batch['label'].to(device)  # (B,seq_len)

            # B,seq_len, tgt_vocab_size) ---> (B * seq-len, tgt_vocab_size)
            loss = loss_fn(
                proj_out.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"Loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            # backpropagate
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

         # Save the model
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_filename
        )


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
