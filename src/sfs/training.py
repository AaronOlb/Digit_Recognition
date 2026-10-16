

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


from sfs.config import BATCH_SIZE, LEARNING_RATE, LOSS_FN, N_EPOCHS, OPTIMIZER, SEED


@dataclass(frozen=True)
class TrainingData:

    train_features: torch.Tensor
    train_labels: torch.Tensor
    val_features: torch.Tensor
    val_labels: torch.Tensor
    val_index: pd.Index


def create_training_data(
    feature_df: pd.DataFrame,
    output_df: pd.Series | pd.DataFrame,
    train_frac: float = 0.8,
    seed: int = SEED,
) -> TrainingData:

    if isinstance(output_df, pd.DataFrame):
        labels = output_df.iloc[:, 0]
    else:
        labels = output_df

    train_idx, val_idx = train_test_split(
        feature_df.index,
        train_size=train_frac,
        random_state=seed,
        stratify=labels,
    )

    train_features = torch.tensor(
        feature_df.loc[train_idx].to_numpy(),
        dtype=torch.float32
    ) / 255.0

    train_labels = torch.tensor(
        labels.loc[train_idx].to_numpy(),
        dtype=torch.long
    )

    val_features = torch.tensor(
        feature_df.loc[val_idx].to_numpy(),
        dtype=torch.float32
    ) / 255.0

    val_labels = torch.tensor(
        labels.loc[val_idx].to_numpy(),
        dtype=torch.long
    )

    return TrainingData(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        val_index=val_idx,
    )



def train(
    model,
    data,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    predictions_path: str | Path | None = None,
):
    """Return epoch metrics and optionally save final validation predictions to CSV."""

    train_loader = DataLoader(
        TensorDataset(data.train_features, data.train_labels),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(data.val_features, data.val_labels),
        batch_size=batch_size,
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history = []

    for epoch in range(n_epochs):

        model.train()
        train_loss = 0

        for X, y in train_loader:
            optimizer.zero_grad()

            logits = model(X)
            loss = loss_fn(logits, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y) 

        model.eval()
        val_loss = 0
        correct = 0
        save_predictions = predictions_path is not None and epoch == n_epochs - 1
        val_probabilities = []

        with torch.inference_mode():
            for X, y in val_loader:
                logits = model(X)
                loss = loss_fn(logits, y)

                val_loss += loss.item() * len(y)

                predictions = logits.argmax(dim=1)
                correct += (predictions == y).sum().item()
                if save_predictions:
                    val_probabilities.append(logits.softmax(dim=1).cpu())

        if save_predictions:
            probabilities = torch.cat(val_probabilities).numpy()
            prediction_df = pd.DataFrame({
                "true_label": data.val_labels.cpu().numpy(),
                "predicted_label": probabilities.argmax(axis=1),
            }, index=data.val_index)
            for class_id in range(probabilities.shape[1]):
                prediction_df[f"probability_{class_id}"] = probabilities[:, class_id]
            prediction_df.to_csv(predictions_path, index_label="row_index")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss / len(data.train_labels),
            "val_loss": val_loss / len(data.val_labels),
            "val_accuracy": correct / len(data.val_labels),
        })

    return pd.DataFrame(history)

