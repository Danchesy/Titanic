import json
import os
import re
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.base import BaseEstimator, TransformerMixin
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from log_utils import _log
from preprocessing import build_preprocessor
from utils import model_filename, run_method


class Tokenizer(BaseEstimator, TransformerMixin):
    """Простой токенизатор, который преобразует строки с именами в последовательности целых чисел фиксированной длины."""

    def __init__(self, max_seq_len: int = 10) -> None:
        self.max_seq_len: int = max_seq_len
        self.vocab: dict[str, int] = {}

    def _clean_and_tokenize(self, text: str) -> list[str]:
        text = text.lower()
        words = re.findall(r"\b\w+\b", text)
        return words

    def fit(self, X: pd.DataFrame) -> "Tokenizer":
        """Создает словарь из столбца Name матрицы X"""
        unique_words = set()
        for name in X["Name"]:
            words = self._clean_and_tokenize(name)
            unique_words.update(words)

        self.vocab = {word: idx + 2 for idx, word in enumerate(unique_words)}
        self.vocab["<PAD>"] = 0
        self.vocab["<UNK>"] = 1

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Кодирует столбец `Name` в DataFrame фиксированной длины с идентификаторами токенов."""
        X1 = X.copy()

        encoded_names = []
        for name in X1["Name"]:
            words = self._clean_and_tokenize(name)
            ids = [self.vocab.get(word, 1) for word in words]

            if len(ids) < self.max_seq_len:
                ids = ids + [0] * (self.max_seq_len - len(ids))
            else:
                ids = ids[: self.max_seq_len]

            encoded_names.append(ids)

        cols = [f"Name token {i}" for i in range(self.max_seq_len)]
        name_df = pd.DataFrame(data=encoded_names, columns=cols, index=X1.index)

        return name_df


class TitanicNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_seq_len: int,
        num_tabular_features: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )

        total_input_dim = num_tabular_features + (max_seq_len * embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_tab: torch.Tensor, x_name: torch.Tensor) -> torch.Tensor:
        # [Batch, 10] -> [Batch, 10, 16]
        x_emb = self.embedding(x_name)

        # [Batch, 10, 16] -> [Batch, 160]
        x_emb_flat = x_emb.view(x_emb.size(0), -1)

        x_combined = torch.cat([x_tab, x_emb_flat], dim=1)

        return self.mlp(x_combined)


def _nn_predict(model, X, X_name_tensor, batch_size=32):
    """Запускает forward pass модели на батчах данных и возвращает объединенные предсказания.

    Args:
        model: PyTorch `nn.Module` to use for prediction.
        X: Tabular features tensor of shape [N, num_tabular_features].
        X_name_tensor: Name token tensor of shape [N, max_seq_len].
        batch_size: Batch size for DataLoader.

    Returns:
        np.ndarray: 1D array of model outputs (probabilities).
    """
    model.eval()
    preds = []

    dataset = TensorDataset(X, X_name_tensor)
    loader = DataLoader(dataset=dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch_tab, batch_name in loader:
            outputs = model(batch_tab, batch_name)
            preds.append(outputs.squeeze().cpu().numpy())

    return np.concatenate(preds)


def save_nn_submission(
    model: nn.Module,
    X_submit: pd.DataFrame,
    cfg: Any,
    tok: Tokenizer,
    preproc: Any,
    submission_name: str = "NN_Model",
) -> str:

    X_name_tensor = torch.tensor(tok.transform(X_submit).values, dtype=torch.long)
    X = torch.tensor(preproc.transform(X_submit), dtype=torch.float32)

    predictions = _nn_predict(model, X, X_name_tensor, batch_size=32)

    submission = pd.DataFrame(
        {"PassengerId": X_submit.index, "Survived": (predictions > 0.5).astype(int)}
    )

    submission_path = os.path.join(
        cfg.data.submission_path, f"{submission_name}_submission.csv"
    )
    submission.to_csv(submission_path, index=False)

    _log(f"Submission saved: {submission_path}", cfg.logging.console)
    return submission_path


def nn_train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: Any,
    optimizer: Any,
    max_grad_norm: float | None = None,
) -> float:
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


def nn_eval(
    model: nn.Module,
    loss_fn: Any,
    X_val: torch.Tensor,
    X_val_name_tensor: torch.Tensor,
    y_val: torch.Tensor,
    methods: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Валидирует модель на отложенной выборке и вычисляет метрики.

    Returns кортеж `(loss, metric_to_score)`, где `loss` — тензор потерь,
    а `metric_to_score` — словарь с метриками.
    """
    model.eval()

    metric_to_score = {}

    with torch.no_grad():
        val_outputs = model(X_val, X_val_name_tensor)
        val_loss = loss_fn(val_outputs.squeeze(), y_val)

        for name, metric_cfg in methods.items():
            metric = hydra.utils.instantiate(metric_cfg)

            y_pred = (val_outputs.squeeze() > 0.5).float()
            metric_score = metric(y_pred, y_val)

            if torch.is_tensor(metric_score):
                metric_score = metric_score.item()

            metric_to_score[name] = float(metric_score)

    return val_loss, metric_to_score


def nn_train_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: Any,
    logger: Any | None = None,
) -> dict[str, Any]:
    console = cfg.logging.console

    preproc = build_preprocessor(
        cfg=cfg,
        model_cfg=cfg.model.nn_model,
        is_scale=cfg.model.nn_model.is_scale,
        is_cat=cfg.model.nn_model.is_cat,
    )

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
        dropout_rate=cfg.model.nn_model.dropout_rate,
    )

    train_dataset = TensorDataset(X_train_tab, X_train_name_tensor, y_train_tensor)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.model.nn_model.batch_size,
        shuffle=cfg.model.nn_model.shuffle,
    )

    optimizer = hydra.utils.instantiate(
        cfg.model.nn_model.optimizer, params=model.parameters()
    )
    loss_fn = nn.BCELoss()
    scheduler = hydra.utils.instantiate(
        cfg.model.nn_model.scheduler, optimizer=optimizer
    )
    epochs = cfg.model.nn_model.epochs

    best_loss = float("inf")
    patience = cfg.model.nn_model.patience
    patience_counter = 0

    for epoch in range(epochs):
        avg_train_loss = nn_train_epoch(
            model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer
        )
        val_loss, metrics = nn_eval(
            model,
            loss_fn=loss_fn,
            X_val=X_val_tab,
            X_val_name_tensor=X_val_name_tensor,
            y_val=y_val_tensor,
            methods=OmegaConf.to_container(cfg.model.nn_model.metrics, resolve=True),
        )

        if val_loss.item() < best_loss - cfg.model.nn_model.min_delta:
            best_loss = val_loss.item()
            patience_counter = 0

            model_name = model.__class__.__name__
            filename = model_filename(
                cfg, model_name, "states", metrics["accuracy"], extension="pt"
            )
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "val_accuracy": metrics["accuracy"],
                "tokenizer_vocab": tok.vocab,
                "tokenizer_max_seq_len": tok.max_seq_len,
            }

            torch.save(checkpoint, filename)

            if logger is not None:
                logger.log_pipeline(filename)
            _log(f"Model states saved as: {filename}", console)
        else:
            patience_counter += 1

        scheduler.step(val_loss)

        if patience_counter >= patience:
            _log(f"Early stopping triggered at epoch {epoch + 1}", console)
            break

        _log(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss.item():.4f}, "
            f"Val Accuracy: {metrics['accuracy']:.4f}",
            console,
        )

    return {
        "model": model,
        "X_val_tab": X_val_tab,
        "X_val_name_tensor": X_val_name_tensor,
        "y_val_tensor": y_val_tensor,
        "preproc": preproc,
        "tok": tok,
        "path": filename,
    }


def add_nn_res(
    metric_to_score: dict[str, float],
    loss: float,
    tuning_time_sec: float | None,
    predict_time: float | None,
    latency_ms: float | None,
    model_cfg: Any,
    path: str,
    results: list | None = None,
    log_file_path: str | None = None,
) -> dict[str, Any]:
    """Создает и опционально сохраняет запись в формате JSONL, описывающую эксперимент с нейронной сетью.
    
    Args:
        metric_to_score: Dict with metric names and float values.
        loss: Validation loss value.
        tuning_time_sec: Tuning/training time in seconds.
        predict_time: Prediction time in seconds.
        latency_ms: Latency per sample in milliseconds.
        model_cfg: Model configuration (OmegaConf node).
        path: Path to saved model or checkpoint.
        results: Optional list to append the experiment record to.
        log_file_path: Optional path to append the JSONL record on disk.

    Returns:
        Словарь с данными эксперимента, включая метрики, время выполнения и путь к модели.
   """

    experiment_data = {
        "model": "NN_Model",
        "accuracy": metric_to_score.get("accuracy", None),
        "f1_score": metric_to_score.get("f1_score", None),
        "precision": metric_to_score.get("precision", None),
        "recall": metric_to_score.get("recall", None),
        "loss": loss,
        "params": OmegaConf.to_container(model_cfg, resolve=True),
        "tuning_time_sec": tuning_time_sec,
        "predict_time_sec": predict_time,
        "latency_ms_per_sample": latency_ms,
        "path": path,
    }

    if results is not None:
        results.append(experiment_data)

    # JSON Lines (один эксперимент — одна строчка в файле)
    if log_file_path:
        with open(log_file_path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(experiment_data, ensure_ascii=False) + "\n")

    return experiment_data


def nn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_submit: pd.DataFrame | None,
    cfg: Any,
    logger: Any | None = None,
) -> None:
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
    cycle_time = pipeline_output["nn_pipeline_time_sec"]
    model = pipeline_res["model"]
    X_val_tab = pipeline_res["X_val_tab"]
    X_val_name_tensor = pipeline_res["X_val_name_tensor"]
    y_val_tensor = pipeline_res["y_val_tensor"]
    tok = pipeline_res["tok"]
    preproc = pipeline_res["preproc"]
    path = pipeline_res["path"]

    loss_fn = torch.nn.BCELoss()

    eval_output = run_method(
        obj=nn_eval,
        method_name="__call__",
        stage="nn_predict",
        model=model,
        loss_fn=loss_fn,
        X_val=X_val_tab,
        X_val_name_tensor=X_val_name_tensor,
        y_val=y_val_tensor,
        methods=OmegaConf.to_container(cfg.model.nn_model.metrics, resolve=True),
    )

    val_loss, metrics = eval_output["result"]
    predict_time = eval_output["nn_predict_time_sec"]
    tuning_time_sec = cycle_time - predict_time
    num_samples = X_val_tab.shape[0]
    latency_ms_per_sample = (predict_time * 1000) / num_samples

    add_nn_res(
        metric_to_score=metrics,
        loss=val_loss.item(),
        tuning_time_sec=tuning_time_sec,
        predict_time=predict_time,
        latency_ms=latency_ms_per_sample,
        model_cfg=cfg.model.nn_model,
        path=path,
        log_file_path=os.path.join(cfg.data.results_dir, "experiments.jsonl"),
    )

    if X_submit is not None and tok is not None and preproc is not None:
        submission_path = save_nn_submission(
            model=model,
            X_submit=X_submit,
            cfg=cfg,
            tok=tok,
            preproc=preproc,
            submission_name="Titanic_NN_Model",
        )
        _log(f"Submission saved: {submission_path}", cfg.logging.console)
