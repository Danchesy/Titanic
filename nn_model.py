import json
import os
import re

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.base import BaseEstimator, TransformerMixin
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import build_preprocessor, log, model_filename, run_method


class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_seq_len = 10):
        self.max_seq_len = max_seq_len
        self.vocab = {}

    def _clean_and_tokenize(self, text):
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def fit(self, X):
        unique_words = set()
        for name in X['Name']:
            words = self._clean_and_tokenize(name)
            unique_words.update(words)

        self.vocab = {word: idx + 2 for idx, word in enumerate(unique_words)}
        self.vocab["<PAD>"] = 0
        self.vocab["<UNK>"] = 1

        return self
    
    def transform(self, X):
        X1 = X.copy()

        encoded_names = []
        for name in X1['Name']:
            words = self._clean_and_tokenize(name)
            ids = [self.vocab.get(word, 1) for word in words]

            if len(ids) < self.max_seq_len:
                ids = ids + [0] * (self.max_seq_len - len(ids))
            else:
                ids = ids[:self.max_seq_len]
                
            encoded_names.append(ids)

        cols = [f"Name token {i}" for i in range(self.max_seq_len)]
        name_df = pd.DataFrame(data=encoded_names, columns=cols, index=X1.index)


        return name_df


class TitanicNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len, num_tabular_features, dropout_rate):
        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embedding_dim, 
            padding_idx=0
        )
        
        total_input_dim = num_tabular_features + (max_seq_len * embedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_tab, x_name):
        # [Batch, 10] -> [Batch, 10, 16]
        x_emb = self.embedding(x_name)
        
        # [Batch, 10, 16] -> [Batch, 160]
        x_emb_flat = x_emb.view(x_emb.size(0), -1)
        
        x_combined = torch.cat([x_tab, x_emb_flat], dim=1)
        
        return self.mlp(x_combined)


def _nn_predict(model, X, X_name_tensor, batch_size=32):
    model.eval()
    preds = []

    dataset = TensorDataset(X, X_name_tensor)
    loader = DataLoader(
                    dataset=dataset, 
                    batch_size=batch_size)

    with torch.no_grad():
        for batch_tab, batch_name in loader: 
            outputs = model(batch_tab, batch_name)
            preds.append(outputs.squeeze().numpy())

    return np.concatenate(preds)


def save_nn_submission(
    model: nn.Module,
    X_submit: pd.DataFrame,
    cfg,
    tok,
    preproc,
    submission_name: str = "NN_Model"):

    X_name_tensor = torch.tensor(tok.transform(X_submit).values, dtype=torch.long)
    X = torch.tensor(preproc.transform(X_submit), dtype=torch.float32)

    predictions = _nn_predict(model, X, X_name_tensor, batch_size=32)

    submission = pd.DataFrame({
        'PassengerId': X_submit.index, 
        'Survived': (predictions > 0.5).astype(int)
    })

    submission_path = os.path.join(
        cfg.data.submission_path, 
        f"{submission_name}_submission.csv"
    )
    submission.to_csv(submission_path, index=False)

    log(f"Submission saved: {submission_path}", cfg.logging.console)
    return submission_path


def nn_train_epoch(model, train_loader, loss_fn, optimizer, max_grad_norm=None):
    model.train()
    epoch_loss = 0.0

    for batch_tab, batch_name, batch_y in train_loader:
        optimizer.zero_grad()
        
        outputs = model(batch_tab, batch_name)
                   
        loss = loss_fn(outputs.squeeze(), batch_y)
        loss.backward()

        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        
        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


def nn_eval(model, loss_fn, X_val, X_val_name_tensor, y_val):
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val, X_val_name_tensor)
        val_loss = loss_fn(val_outputs.squeeze(), y_val)
        val_accuracy = ((val_outputs.squeeze() > 0.5).float() == y_val).float().mean()

    return val_loss, val_accuracy    


def nn_train_pipeline(X_train, y_train, X_val, y_val, cfg, logger=None): 
    console = cfg.logging.console

    preproc = build_preprocessor(cfg=cfg,
                        model_cfg=cfg.model.nn_model,
                        is_scale=cfg.model.nn_model.is_scale, 
                        is_cat=cfg.model.nn_model.is_cat)


    tok = Tokenizer(max_seq_len=cfg.model.nn_model.max_seq_len)
    tok.fit(X_train)
    X_train_name = tok.transform(X_train)
    X_val_name = tok.transform(X_val)

    vocab_size = len(tok.vocab)
    embedding_dim = cfg.model.nn_model.embedding_dim


    X_train_name_tensor = torch.tensor(X_train_name.values, dtype=torch.long)
    X_val_name_tensor = torch.tensor(X_val_name.values, dtype=torch.long)

    X_train_tab = torch.tensor(preproc.fit_transform(X_train), dtype=torch.float32)
    X_val_tab = torch.tensor(preproc.transform(X_val), dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

    model = TitanicNet(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_seq_len=cfg.model.nn_model.max_seq_len,
        num_tabular_features=X_train_tab.shape[1],
        dropout_rate=cfg.model.nn_model.dropout_rate
    )

    train_dataset = TensorDataset(X_train_tab, X_train_name_tensor, y_train_tensor)
    train_loader = DataLoader(
                    dataset=train_dataset, 
                    batch_size=cfg.model.nn_model.batch_size,
                    shuffle = cfg.model.nn_model.shuffle)

    optimizer = hydra.utils.instantiate(cfg.model.nn_model.optimizer, params=model.parameters())
    loss_fn = nn.BCELoss()
    scheduler = hydra.utils.instantiate(cfg.model.nn_model.scheduler, optimizer=optimizer)
    epochs = cfg.model.nn_model.epochs

    best_loss = float('inf')
    patience = cfg.model.nn_model.patience
    patience_counter = 0

    for epoch in range(epochs):
        avg_train_loss = nn_train_epoch(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer)
        val_loss, val_accuracy = nn_eval(model, loss_fn=loss_fn, X_val=X_val_tab, X_val_name_tensor=X_val_name_tensor, y_val=y_val_tensor)

        if val_loss.item() < best_loss - cfg.model.nn_model.min_delta:
            best_loss = val_loss.item()
            patience_counter = 0

            model_name = model.__class__.__name__
            filename = model_filename(cfg, model_name, "states", val_accuracy, extension='pt')
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_accuracy': val_accuracy, 
                'tokenizer_vocab': tok.vocab,
                'tokenizer_max_seq_len': tok.max_seq_len,
            }
            
            torch.save(checkpoint, filename)

            if logger is not None:
                logger.log_pipeline(filename)
            log(f"Model states saved as: {filename}", console)
        else: 
            patience_counter += 1

        scheduler.step(val_loss)

        if patience_counter >= patience:
            log(f"Early stopping triggered at epoch {epoch+1}", console)
            break

        log(f'Epoch {epoch+1}/{epochs}, '
        f'Loss: {avg_train_loss:.4f}, '
        f'Val Loss: {val_loss.item():.4f}, '
        f'Val Accuracy: {val_accuracy.item():.4f}', console)

    return {
        "model": model, 
        "X_val_tab": X_val_tab, 
        "X_val_name_tensor": X_val_name_tensor, 
        "y_val_tensor": y_val_tensor,
        "preproc": preproc,
        "tok": tok,
    }


def add_nn_res(
    accuracy,
    loss,
    predict_time,
    latency_ms,
    model_cfg,
    results = None,
    log_file_path = None):

    experiment_data = {
        "model": 'NN_Model',
        "accuracy": accuracy,
        "loss": loss,
        "hyperparams": OmegaConf.to_container(model_cfg, resolve=True),
        "predict_time_sec": predict_time,
        "latency_ms_per_sample": latency_ms,
    }

    if results is not None:
        results.append(experiment_data)

    # Дописываем в файл ('a' — append)
    # JSON Lines (один эксперимент — одна строчка в файле)
    if log_file_path:
        with open(log_file_path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(experiment_data, ensure_ascii=False) + "\n")

    return experiment_data


def nn_model(X_train, y_train, X_val, y_val, X_submit, cfg, logger=None):
    pipeline_output = run_method(
        obj=nn_train_pipeline,
        method_name="__call__",
        stage="nn_pipeline",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        cfg=cfg,
        logger=logger,
    )

    pipeline_res = pipeline_output["result"]
    model = pipeline_res["model"]
    X_val_tab = pipeline_res["X_val_tab"]
    X_val_name_tensor = pipeline_res["X_val_name_tensor"]
    y_val_tensor = pipeline_res["y_val_tensor"]
    tok = pipeline_res["tok"] 
    preproc = pipeline_res["preproc"] 

    loss_fn = torch.nn.BCELoss()

    eval_output = run_method(
        obj=nn_eval,
        method_name="__call__",
        stage="nn_predict",
        model=model,
        loss_fn=loss_fn,
        X_val=X_val_tab,
        X_val_name_tensor=X_val_name_tensor,
        y_val=y_val_tensor
    )

    val_loss, val_accuracy = eval_output["result"]
    predict_time = eval_output["nn_predict_time_sec"]
    num_samples = X_val_tab.shape[0]
    latency_ms_per_sample = (predict_time * 1000) / num_samples

    add_nn_res(
        accuracy=val_accuracy.item(),
        loss=val_loss.item(),
        predict_time=predict_time,
        latency_ms=latency_ms_per_sample,
        model_cfg=cfg.model.nn_model,
        log_file_path=os.path.join(cfg.data.results_dir, "experiments.jsonl")
    )

    if X_submit is not None and tok is not None and preproc is not None:
        submission_path = save_nn_submission(
            model=model,
            X_submit=X_submit,
            cfg=cfg,
            tok=tok,
            preproc=preproc,
            submission_name="Titanic_NN_Model"
        )
        log(f"Submission saved: {submission_path}", cfg.logging.console)