import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import string
import re
from tqdm import tqdm
import math
import ssl
import inspect
from torch.utils.data.dataloader import default_collate
import argparse
import os

# Disable SSL cert verification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass  # Legacy Python, ignore
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

#use_train_val_split:bool = True
use_train_val_split:bool = False

if use_train_val_split == True:
    TRANSF_MAX_VOCAB_SIZE = 100000 #TODO Tune this value
    TRANSF_PATIENCE       = 10
else:
    TRANSF_MAX_VOCAB_SIZE = 118000 #TODO Tune this value
    TRANSF_PATIENCE       = 6

TRANSF_BATCH_SIZE = 16     #TODO Tune this value
TRANSF_HID_DIM    = 256    #TODO Tune this value
TRANSF_EMB_DIM    = 192    #TODO Tune this value
TRANSF_MAX_LEN    = 512    #TODO Tune this value
TRANSF_NUM_LAYERS = 3
TRANSF_DROPOUT    = 0.1
TRANSF_NUM_EPOCHS = 35
TRANSF_LR         = 5e-5
TRANSF_NUM_HEADS  = 8


if use_train_val_split == True:
    LSTM_MAX_VOCAB_SIZE   = 40000 #TODO Tune this value
    LSTM_PATIENCE         = 10
else:
    LSTM_MAX_VOCAB_SIZE   = 60000 #TODO Tune this value
    LSTM_PATIENCE         = 6

LSTM_HID_DIM      = 192    #TODO Tune this value
LSTM_EMB_DIM      = 128    #TODO Tune this value
LSTM_MAX_LEN      = 400    #TODO Tune this value
LSTM_NUM_LAYERS   = 3
LSTM_NUM_EPOCHS   = 35

LSTM_WT_DECAY     = 1e-3

#LSTM_BATCH_SIZE   =  8     #TODO Tune this value
LSTM_BATCH_SIZE   = 16     #TODO Tune this value
#LSTM_BATCH_SIZE   = 32     #TODO Tune this value
#LSTM_BATCH_SIZE   = 64     #TODO Tune this value

#LSTM_DROPOUT      = 0.2
#LSTM_DROPOUT      = 0.25
LSTM_DROPOUT      = 0.3
#LSTM_DROPOUT      = 0.325
#LSTM_DROPOUT      = 0.35

LSTM_LR           = 3e-4
LSTM_DIR          = './'

## ##################################################### ##
## Final                                                ##
## ##################################################### ##
def preprocess_text(text):
    """
    Clean and tokenize text
    """

    #print(f"DEBUG: Enter {inspect.currentframe().f_code.co_name}", flush=True, end= "\t####: ", flush=True)
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(f'[{string.punctuation}]', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Replace all HTML <br>, <br/>, or <br /> tags with a single space
        text = re.sub(r"<br\s*/?>", " ", text)

        # Tokenize
        tokens = word_tokenize(text)

        #print(f"\tDEBUG: Leave - 1 {inspect.currentframe().f_code.co_name}", flush=True)
        return tokens

    #print(f"\tDEBUG: Leave - 2 {inspect.currentframe().f_code.co_name}", flush=True)
    return []

## ##################################################### ##
## Class Vocabulary                                      ##
## ##################################################### ##
class Vocabulary:
    """
    Build a vocabulary from the word count
    """

    ## ################################################# ##
    ##                                                   ##
    ## ################################################# ##
    def __init__(self, max_size):
        func = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
        #print(f"\nDEBUG: Enter {func}", flush=True)

        self.max_size = max_size
        # Add <cls> token for transformer classification
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<cls>"}
        self.size = 3  # Start with pad, unk, and cls tokens
        self.word_count = {}
        print(f"\tDEBUG: Leave{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}",
              flush=True)


    ## ################################################# ##
    ##                                                   ##
    ## ################################################# ##
    def add_word(self, word):
        func = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
        #print(f"\nDEBUG: Enter {func}", flush=True)

        if word in self.word_count:
            self.word_count[word] += 1
        else:
            self.word_count[word] = 1
        #print(f"\tDEBUG: Leave{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}",
        #      flush=True)


    ## ################################################# ##
    ##                                                   ##
    ## ################################################# ##
    def build_vocab(self):
        func = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
        #print(f"\nDEBUG: Enter {func}", flush=True)

        #sort words in the descending order of frequency
        sorted_words = sorted(self.word_count.items(), key=lambda x: x[1], reverse = True)

        # We can add up to max_size - 3 new words as we are given 3 words to begin with
        for word , _ in sorted_words:
            if self.size >= self.max_size:
                break

            if word not in self.word2idx:
                self.word2idx[word] = self.size
                self.idx2word[self.size] = word
                self.size += 1

        print(f"\tDEBUG: Leave{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}",
              flush=True)

    ## ################################################# ##
    ##                                                   ##
    ## ################################################  ##
    def text_to_indices(self, tokens, max_len, model_type='lstm'):
        func = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
        #print(f"\nDEBUG: Enter {func}", flush=True)

        indices = []

        if model_type == 'transformer':
            indices.append(self.word2idx["<cls>"])

        for token in tokens:
            if len(indices) >= max_len:
                break

            if token in self.word2idx:
                indices.append(self.word2idx[token])
            elif model_type == 'lstm':
                indices.append(self.word2idx["<unk>"])
            # Transformer skips unknowns

        # pad if too short
        if len(indices) < max_len:
            indices += [self.word2idx["<pad>"]] * (max_len - len(indices))

        #print(f"\tDEBUG: Leave{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}",
        #      flush=True)
        return indices



## ##################################################### ##
## Class IMDBDataset                                     ##
## ##################################################### ##
class IMDBDataset(Dataset):
    """
    A dataset for the IMDB dataset
    """

    def __init__(self, dataframe, vocabulary, max_len, is_training=True, model_type='lstm'):
        func = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
        #print(f"\nDEBUG: Enter {func}", flush=True)

        self.vocab = vocabulary
        self.max_len = max_len
        self.model_type = model_type
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['label'].tolist()

        self.indexed_texts = []
        self.attention_masks = []

        unk_idx = self.vocab.word2idx["<unk>"]
        total_tokens = 0
        total_unks = 0

        for text in self.texts:
            tokens = preprocess_text(text)
            indices = self.vocab.text_to_indices(tokens, max_len=self.max_len, model_type=self.model_type)
            self.indexed_texts.append(torch.tensor(indices, dtype=torch.long))

            # Count unknowns for diagnostics
            total_tokens += len(indices)
            total_unks += sum(1 for idx in indices if idx == unk_idx)

            if self.model_type == 'transformer':
                pad_idx = self.vocab.word2idx["<pad>"]
                mask = [1 if idx != pad_idx else 0 for idx in indices]
                self.attention_masks.append(torch.tensor(mask, dtype=torch.long))

        print(f"UNK coverage: {total_unks}/{total_tokens} tokens ({(total_unks / total_tokens) * 100:.2f}%) were <unk>",
              flush=True)
        print(f"\tDEBUG: Leave {self.__class__.__name__}.{inspect.currentframe().f_code.co_name}", flush=True)

    ## ################################################# ##
    ##                                                   ##
    ## ################################################# ##
    def __len__(self):
        func = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
        #print(f"\nDEBUG: Enter {func}", flush=True)

        return len(self.indexed_texts)


    ## ################################################# ##
    ##                                                   ##
    ## ################################################# ##
    def __getitem__(self, idx):
        func = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
        #print(f"\nDEBUG: Enter {func}", flush=True)

        tokens = preprocess_text(self.texts[idx])
        label = self.labels[idx]

        indices = self.vocab.text_to_indices(tokens, self.max_len, self.model_type)
        input_tensor = torch.tensor(indices, dtype=torch.long)
        label_tensor = torch.tensor([label], dtype=torch.float)

        if self.model_type == 'lstm':
            assert label_tensor.shape == (1,), f"Label shape is wrong: {label_tensor.shape}"
            assert label_tensor.item() in [0.0, 1.0], f"Invalid label: {label_tensor.item()}"
            return (input_tensor, label_tensor)
        else:
            attention_mask = (input_tensor != self.vocab.word2idx['<pad>']).long()
            return input_tensor, attention_mask, label_tensor


## ##################################################### ##
## LSTM model                                            ##
## ##################################################### ##
class LSTM(nn.Module):
    def __init__(self,
                 vocab_size=LSTM_MAX_VOCAB_SIZE,
                 embedding_dim=LSTM_EMB_DIM,
                 hidden_dim=LSTM_HID_DIM,
                 output_dim=1,
                 num_layers=LSTM_NUM_LAYERS,
                 dropout=LSTM_DROPOUT
                ):

        func = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
        #print(f"\nDEBUG: Enter {func}", flush=True)

        self.bidirectional = True
        super(LSTM, self).__init__()


        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
                            input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=(dropout if num_layers > 1 else 0),
                            bidirectional=self.bidirectional
                        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        #self.fc = nn.Linear(hidden_dim, output_dim)

        print(f"\tDEBUG: Leave{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}",
              flush=True)


    def forward(self, x):
        func = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
        #print(f"\nDEBUG: Enter {func}", flush=True)

        # If input is 1D, unsqueeze batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)  # shape: [1, seq_len]

        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)

        # hidden shape: [2*num_layers, batch, hidden_dim]
        # Take final forward and backward hidden states and concatenate
        forward_hidden = hidden[-2, :, :]  # [batch, hidden_dim]
        backward_hidden = hidden[-1, :, :]  # [batch, hidden_dim]
        last_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)  # [batch, 2*hidden_dim]

        out = self.dropout(last_hidden)
        logits = self.fc(out)

        #print(f"\tDEBUG: Leave{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}", flush=True)
        return logits


## ##################################################### ##
## Positional Encoding for Transformer                   ##
## ##################################################### ##
class PositionalEncoding(nn.Module):
    def __init__(self,
                 embedding_dim=TRANSF_EMB_DIM,
                 max_len=TRANSF_MAX_LEN):

        super(PositionalEncoding, self).__init__()

        func = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
        #print(f"\nDEBUG: Enter {func}", flush=True)

        # Create position encoding matrix
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd

        pe = pe.unsqueeze(0)  # [1, max_len, embedding_dim]
        self.register_buffer("pe", pe)

        #print(f"\tDEBUG: Leave{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}", flush=True)

    def forward(self, x=None):
        # x: [batch_size, seq_len, embedding_dim]
        x = x + self.pe[:, :x.size(1), :]

        #print(f"\tDEBUG: Leave{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}", flush=True)
        return x


## ##################################################### ##
## Transformer Encoder                                   ##
## ##################################################### ##
class TransformerEncoder(nn.Module):
    def __init__(self,
                 vocab_size=TRANSF_MAX_VOCAB_SIZE,
                 embedding_dim=TRANSF_EMB_DIM,
                 num_heads=TRANSF_NUM_HEADS,
                 hidden_dim=TRANSF_HID_DIM,
                 num_layers=TRANSF_NUM_LAYERS,
                 dropout=TRANSF_DROPOUT,
                 max_len=TRANSF_MAX_LEN):

        super(TransformerEncoder, self).__init__()

        func = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
        #print(f"\nDEBUG: Enter {func}", flush=True)

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, 1)  # Binary classification


    def forward(self, input_ids=None, attention_mask=None):

        func = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
        #print(f"\nDEBUG: Enter {func}", flush=True)

        # Handle single sample input
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        src_key_padding_mask = attention_mask == 0
        encoder_output = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        cls_output = encoder_output[:, 0, :]
        logits = self.fc(cls_output)

        return logits


## ##################################################### ##
##                                                       ##
## ##################################################### ##
def load_and_preprocess_data(data_path, data_type='train', model_type='lstm', shared_vocab=None):
    """
    Load and preprocess the IMDB dataset

    Args:
        data_path: Path to the data files
        data_type: Type of data to load ('train' or 'test')
        model_type: Type of model ('lstm' or 'transformer')
        shared_vocab: Optional vocabulary to use (for test data)

    Returns:
        data_loader: DataLoader for the specified data type
        vocab: Vocabulary object (only returned for train data)
    """
    print(f"\nDEBUG: Enter {inspect.currentframe().f_code.co_name}", flush=True)


    df = pd.read_parquet(data_path)

    if data_type == 'train':
        vocab = Vocabulary(max_size = (LSTM_MAX_VOCAB_SIZE if model_type == 'lstm' else TRANSF_MAX_VOCAB_SIZE))
        for text in df['text']:
            tokens = preprocess_text(text)
            for token in tokens:
                vocab.add_word(token)
        vocab.build_vocab()

    else:
        vocab = shared_vocab
        if vocab is None:
            raise ValueError("shared_vocab must be provided for test/val data")

    # Create Dataset
    dataset = IMDBDataset(df,
                          vocabulary=vocab,
                          max_len = (LSTM_MAX_LEN if model_type == 'lstm' else TRANSF_MAX_LEN),
                          is_training=(data_type=='train'),
                          model_type=model_type)

    # Create DataLoader
    shuffle = True if data_type == 'train' else False

    data_loader = DataLoader(dataset,
                             batch_size = (LSTM_BATCH_SIZE if model_type == 'lstm' else TRANSF_BATCH_SIZE),
                             shuffle=shuffle)
    if data_type == 'train':
        retval = data_loader, vocab
    else:
        retval = data_loader

    print(f"\tDEBUG: Leave {inspect.currentframe().f_code.co_name}", flush=True)
    return retval



## ##################################################### ##
##                                                       ##
## ##################################################### ##
def train(model, iterator, optimizer, criterion, device, model_type='lstm'):
    print(f"DEBUG: Enter {inspect.currentframe().f_code.co_name} for {model_type}", flush=True)

    model.train()

    epoch_loss = 0
    correct = 0
    total = 0

    for batch in iterator:
        #print(f"DEBUG: Batch")
        #print(f"DEBUG: Type of batch: {type(batch)}")
        #print(f"DEBUG: Length of batch: {len(batch)}")
        #print(f"DEBUG: Types of items in batch: {[type(x) for x in batch]}")

        #if isinstance(batch, (list, tuple)):
        #    for j, item in enumerate(batch):
        #        if isinstance(item, torch.Tensor):
        #            print(f"DEBUG: Shape of item {j}: {item.shape}")
        optimizer.zero_grad()

        if model_type == 'transformer':
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            #labels = labels.to(device)
            labels = labels.to(device).float()

            logits = model(input_ids, attention_mask)

        else:  # LSTM
            inputs, labels = batch    #TODO
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            logits = model(inputs)

        loss = criterion(logits, labels)
        loss.backward()
        if  model_type == 'lstm':
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        epoch_loss += loss.item()

        # Compute accuracy
        probs = torch.sigmoid(logits)

        preds = (probs >= 0.5).long()
        #correct += (preds == labels).sum().item()
        correct += (preds.squeeze() == labels.squeeze().long()).sum().item()

        total += labels.size(0)

    avg_loss = epoch_loss / len(iterator)
    accuracy = correct / total

    print(f"\tDEBUG: Leave {inspect.currentframe().f_code.co_name} for {model_type}", flush=True)
    return avg_loss, accuracy


## ##################################################### ##
##                                                       ##
## ##################################################### ##
def evaluate(model, iterator, criterion, device, model_type='lstm'):
    print(f"DEBUG: Enter {inspect.currentframe().f_code.co_name}", flush=True)

    model.eval()

    epoch_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in iterator:
            if model_type == 'transformer':
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device).float()

                logits = model(input_ids, attention_mask)

            else:  # LSTM
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.to(device).float()

                logits = model(inputs)

            loss = criterion(logits, labels)
            epoch_loss += loss.item()

            probs = torch.sigmoid(logits)

            preds = (probs >= 0.5).long()
            #correct += (preds == labels).sum().item()
            correct += (preds.squeeze() == labels.squeeze().long()).sum().item()

            total += labels.size(0)

    avg_loss = epoch_loss / len(iterator)
    accuracy = correct / total

    print(f"\tDEBUG: Leave {inspect.currentframe().f_code.co_name}", flush=True)
    return avg_loss, accuracy

## #################################################### ##
##                                                      ##
## #################################################### ##
def prepare_data_and_train(model_type='lstm', data_path="train.parquet", num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = (LSTM_NUM_EPOCHS if model_type == 'lstm' else TRANSF_NUM_EPOCHS)
    # Step 1: Load Data
    if use_train_val_split == True:
        df = pd.read_parquet(data_path)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

        # Save splits
        train_df.to_parquet("train_split.parquet", index=False)
        val_df.to_parquet("val_split.parquet", index=False)

        train_loader, vocab = load_and_preprocess_data("train_split.parquet",
                                                       data_type='train',
                                                       model_type=model_type)

        val_loader = load_and_preprocess_data("val_split.parquet",
                                              data_type='test',
                                              model_type=model_type,
                                              shared_vocab=vocab)
    else:
        train_loader, vocab = load_and_preprocess_data(data_path=data_path,
                                                   data_type='train',
                                                   model_type=model_type)


    # Step 3: Init model
    if model_type == 'lstm':
        model = LSTM(vocab_size=vocab.size,
                     embedding_dim=LSTM_EMB_DIM,
                     hidden_dim=LSTM_HID_DIM,
                     output_dim=1,
                     num_layers=LSTM_NUM_LAYERS,
                     dropout=LSTM_DROPOUT
                 ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LSTM_LR, weight_decay=LSTM_WT_DECAY)

    else:
        model = TransformerEncoder(vocab_size=vocab.size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=TRANSF_LR)

    # Step 4: Loss & Optimizer
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=0.5,
                                                           patience=5)

    # Step 5: Training loop with early stopping
    save_path = f"{model_type}.pt"
    epochs_without_improvement = 0
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}", flush=True)
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, model_type)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}", flush=True)

        if use_train_val_split == True:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device, model_type)
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}", flush=True)

        if use_train_val_split == True:
            current_acc = val_acc
        else:
            current_acc = train_acc

        # Scheduler step
        scheduler.step(current_acc)
        for param_group in optimizer.param_groups:
            print(f"Current LR: {param_group['lr']:.6f}")

        global LSTM_DIR

        # Check for improvement
        if current_acc > best_acc:
            best_acc = current_acc


            epochs_without_improvement = 0
            if model_type == 'lstm':
                #epoch_model_path = f"{model_type}_{epoch+1}.pt"
                LSTM_DIR = f"lstm_ED{LSTM_EMB_DIM}_HD{LSTM_HID_DIM}_BS{LSTM_BATCH_SIZE}_DO{LSTM_DROPOUT}_LR{LSTM_LR}"
                os.makedirs(LSTM_DIR, exist_ok=True)
                epoch_model_path = f"{LSTM_DIR}/{model_type}_{epoch+1}__acc{current_acc:.4f}.pt"
                torch.save(model.state_dict(), epoch_model_path, _use_new_zipfile_serialization=True)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path} (current acc = {current_acc:.4f})", flush=True)
        else:
            epochs_without_improvement += 1
            print(f" No improvement for {epochs_without_improvement} epoch(s)", flush=True)

        # Early stopping
        if epochs_without_improvement >= (LSTM_PATIENCE if model_type == 'lstm' else TRANSF_PATIENCE):
            print("Early stopping: Training accuracy plateaued", flush=True)
            break

        if  best_acc > 0.98:
            print("Early stopping: Training accuracy > 0.97", flush=True)
            break

    print(f"\nBest Validation accuracy: {best_acc:.4f}", flush=True)



def main():

    prepare_data_and_train(model_type='lstm', num_epochs=LSTM_NUM_EPOCHS)

    # Train Transformer model and save as transformer.pt
    prepare_data_and_train(model_type='transformer', num_epochs=TRANSF_NUM_EPOCHS)

if __name__ == "__main__":
    main()
