import argparse
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.datasets import Amazon, Planetoid
from transformers import BertConfig, BertModel


@dataclass
class DatasetBundle:
    name: str
    x: torch.Tensor
    y: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    num_features: int
    num_classes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BERT-style node classification baselines on citation/product datasets."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Optional dataset root. Supports custom datasets such as children/history/photo.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ogbn-arxiv", "cora", "pubmed", "amazon-photo"],
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--feature-types",
        nargs="+",
        default=["raw"],
        help="Feature types to evaluate. For custom datasets, supports raw or plm.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bert"],
        help="Accepted for compatibility with multi-model scripts. Only bert is used here.",
    )
    parser.add_argument("--runs", type=int, default=5, help="Repeated runs per dataset.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=1024, help="Evaluation batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Classifier dropout.")
    parser.add_argument("--hidden-size", type=int, default=128, help="Transformer hidden size.")
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Alias of --hidden-size for compatibility with GNN-style scripts.",
    )
    parser.add_argument(
        "--num-hidden-layers", type=int, default=4, help="Number of transformer layers."
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Accepted for compatibility with GNN-style scripts. Ignored by this script.",
    )
    parser.add_argument(
        "--num-attention-heads", type=int, default=4, help="Number of attention heads."
    )
    parser.add_argument(
        "--intermediate-size", type=int, default=256, help="Feed-forward hidden size."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Top-k feature dimensions converted into tokens for each node.",
    )
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Directory for csv results."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
    args = parser.parse_args()
    if args.hidden_dim is not None:
        args.hidden_size = args.hidden_dim
    args.datasets = [dataset.lower() for dataset in args.datasets]
    args.feature_types = [feature_type.lower() for feature_type in args.feature_types]
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_index(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.bool:
        return mask.nonzero(as_tuple=False).view(-1)
    return mask.view(-1)


def stratified_split(
    y: torch.Tensor,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    y_np = y.cpu().numpy()
    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []

    for label in np.unique(y_np):
        cls_idx = np.where(y_np == label)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        train_parts.append(cls_idx[:n_train])
        val_parts.append(cls_idx[n_train : n_train + n_val])
        test_parts.append(cls_idx[n_train + n_val :])

    train_idx = torch.from_numpy(np.concatenate(train_parts)).long()
    val_idx = torch.from_numpy(np.concatenate(val_parts)).long()
    test_idx = torch.from_numpy(np.concatenate(test_parts)).long()
    return train_idx, val_idx, test_idx


def _find_existing_path(candidates: List[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _title_case_dataset_name(name: str) -> str:
    return {
        "children": "Children",
        "history": "History",
        "photo": "Photo",
    }.get(name, name)


def _resolve_custom_dataset_paths(root: Path, name: str) -> Tuple[Path, Path]:
    title_name = _title_case_dataset_name(name)
    csv_path = _find_existing_path(
        [
            root / title_name / f"{title_name}.csv",
            root / name / f"{name}.csv",
            root / "CSTAG" / title_name / f"{title_name}.csv",
            root / "CSTAG" / name / f"{name}.csv",
        ]
    )
    if csv_path is None:
        raise FileNotFoundError(f"Unable to locate csv for dataset '{name}' under {root}.")
    dataset_dir = csv_path.parent
    return dataset_dir, csv_path


def _resolve_feature_path(root: Path, dataset_dir: Path, name: str, feature_type: str) -> Optional[Path]:
    title_name = _title_case_dataset_name(name)
    feature_type = feature_type.lower()
    candidates: List[Path] = []
    if feature_type == "plm":
        candidates.extend(
            [
                root / "manual_features" / f"{name}_plm.npy",
                root / "manual_features" / f"{title_name.lower()}_plm.npy",
                root / "manual_features" / f"{title_name}_plm.npy",
                dataset_dir / "Feature" / f"{title_name}_roberta_base_512_cls.npy",
                dataset_dir / "Feature" / f"{name}_roberta_base_512_cls.npy",
            ]
        )
    elif feature_type == "raw":
        return None
    else:
        raise ValueError(f"Unsupported feature type '{feature_type}' for dataset '{name}'.")

    feature_path = _find_existing_path(candidates)
    if feature_path is None:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"Unable to locate feature file for dataset '{name}' and feature type '{feature_type}'. "
            f"Searched: {searched}"
        )
    return feature_path


def _load_custom_dataset(name: str, root: Path, split_seed: int, feature_type: str) -> DatasetBundle:
    dataset_dir, csv_path = _resolve_custom_dataset_paths(root, name)
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"Dataset csv is missing required 'label' column: {csv_path}")

    feature_path = _resolve_feature_path(root, dataset_dir, name, feature_type)
    if feature_path is None:
        raise ValueError(
            f"Dataset '{name}' does not contain built-in dense features. "
            f"Please provide '--feature-types plm' with a valid .npy feature file."
        )

    x = torch.from_numpy(np.load(feature_path).astype(np.float32))
    y = torch.from_numpy(df["label"].to_numpy()).long()

    if x.size(0) != y.size(0):
        raise ValueError(
            f"Feature rows ({x.size(0)}) do not match label rows ({y.size(0)}) for dataset '{name}'."
        )

    train_idx, val_idx, test_idx = stratified_split(
        y=y,
        train_ratio=0.2,
        val_ratio=0.2,
        seed=split_seed,
    )
    return DatasetBundle(
        name=f"{name}-{feature_type}",
        x=x,
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        num_features=x.size(1),
        num_classes=int(y.max().item()) + 1,
    )


def load_dataset(name: str, root: Path, split_seed: int, feature_type: str = "raw") -> DatasetBundle:
    if name == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(root / "ogb"))
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        x = data.x.float()
        y = data.y.view(-1).long()
        return DatasetBundle(
            name=name,
            x=x,
            y=y,
            train_idx=split_idx["train"].long(),
            val_idx=split_idx["valid"].long(),
            test_idx=split_idx["test"].long(),
            num_features=x.size(1),
            num_classes=dataset.num_classes,
        )

    if name in {"cora", "pubmed"}:
        dataset = Planetoid(root=str(root / "planetoid"), name="Cora" if name == "cora" else "PubMed")
        data = dataset[0]
        x = data.x.float()
        y = data.y.long()
        return DatasetBundle(
            name=name,
            x=x,
            y=y,
            train_idx=to_index(data.train_mask),
            val_idx=to_index(data.val_mask),
            test_idx=to_index(data.test_mask),
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
        )

    if name == "amazon-photo":
        dataset = Amazon(root=str(root / "amazon"), name="Photo")
        data = dataset[0]
        x = data.x.float()
        y = data.y.long()
        train_idx, val_idx, test_idx = stratified_split(
            y=y,
            train_ratio=0.2,
            val_ratio=0.2,
            seed=split_seed,
        )
        return DatasetBundle(
            name=name,
            x=x,
            y=y,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
        )

    if name in {"children", "history", "photo"}:
        return _load_custom_dataset(name=name, root=root, split_seed=split_seed, feature_type=feature_type)

    raise ValueError(f"Unsupported dataset: {name}")


class FeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, hidden_size: int, max_tokens: int):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.max_tokens = max_tokens
        self.feature_embedding = nn.Embedding(num_features + 1, hidden_size, padding_idx=0)
        self.value_projection = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_features = x.shape
        k = min(self.max_tokens, num_features)
        scores = x.abs()
        top_values, top_indices = torch.topk(scores, k=k, dim=1)
        original_values = torch.gather(x, 1, top_indices)

        feature_ids = top_indices + 1
        active_mask = top_values > 0
        feature_ids = feature_ids.masked_fill(~active_mask, 0)
        original_values = original_values.masked_fill(~active_mask, 0.0)

        token_embeds = self.feature_embedding(feature_ids)
        value_embeds = self.value_projection(original_values.unsqueeze(-1))
        inputs_embeds = self.layer_norm(token_embeds + value_embeds)
        attention_mask = active_mask.long()

        if attention_mask.sum(dim=1).eq(0).any():
            empty_rows = attention_mask.sum(dim=1).eq(0)
            inputs_embeds[empty_rows, 0, :] = 0.0
            attention_mask[empty_rows, 0] = 1

        return inputs_embeds, attention_mask


class BertNodeClassifier(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float,
        max_tokens: int,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(
            num_features=num_features,
            hidden_size=hidden_size,
            max_tokens=max_tokens,
        )
        config = BertConfig(
            vocab_size=num_features + 1,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_tokens + 2,
            type_vocab_size=1,
            pad_token_id=0,
        )
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs_embeds, attention_mask = self.tokenizer(x)
        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        return self.classifier(pooled)


def build_loaders(
    dataset: DatasetBundle,
    batch_size: int,
    eval_batch_size: int,
) -> Dict[str, DataLoader]:
    def make_loader(indices: torch.Tensor, shuffle: bool, current_batch_size: int) -> DataLoader:
        ds = TensorDataset(dataset.x[indices], dataset.y[indices])
        return DataLoader(ds, batch_size=current_batch_size, shuffle=shuffle)

    return {
        "train": make_loader(dataset.train_idx, shuffle=True, current_batch_size=batch_size),
        "val": make_loader(dataset.val_idx, shuffle=False, current_batch_size=eval_batch_size),
        "test": make_loader(dataset.test_idx, shuffle=False, current_batch_size=eval_batch_size),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for batch_x, batch_y in loader:
        logits = model(batch_x.to(device))
        pred = logits.argmax(dim=-1).cpu().numpy()
        preds.append(pred)
        labels.append(batch_y.cpu().numpy())

    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def train_one_run(
    dataset: DatasetBundle,
    args: argparse.Namespace,
    run_seed: int,
) -> Dict[str, float]:
    set_seed(run_seed)
    device = torch.device(args.device)
    loaders = build_loaders(dataset, args.batch_size, args.eval_batch_size)

    model = BertNodeClassifier(
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        dropout=args.dropout,
        max_tokens=args.max_tokens,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_state = None
    best_score = -1.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for batch_x, batch_y in loaders["train"]:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()

            batch_size = batch_y.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        val_metrics = evaluate(model, loaders["val"], device)
        val_score = val_metrics["acc"] + val_metrics["f1_macro"]

        print(
            f"[{dataset.name}] seed={run_seed} epoch={epoch:03d} "
            f"train_loss={total_loss / max(total_examples, 1):.4f} "
            f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1_macro']:.4f}"
        )

        if val_score > best_score:
            best_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    val_metrics = evaluate(model, loaders["val"], device)
    test_metrics = evaluate(model, loaders["test"], device)
    return {
        "val_acc": val_metrics["acc"],
        "val_f1_macro": val_metrics["f1_macro"],
        "test_acc": test_metrics["acc"],
        "test_f1_macro": test_metrics["f1_macro"],
    }


def run_experiments(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.root) if args.root is not None else output_dir / "datasets"

    all_rows: List[Dict[str, float]] = []

    for feature_type in args.feature_types:
        for dataset_name in args.datasets:
            for run in range(args.runs):
                run_seed = args.seed + run
                dataset = load_dataset(dataset_name, data_root, split_seed=run_seed, feature_type=feature_type)
                metrics = train_one_run(dataset, args, run_seed=run_seed)
                row = {
                    "dataset": dataset_name,
                    "feature_type": feature_type,
                    "run": run + 1,
                    "seed": run_seed,
                    **metrics,
                }
                all_rows.append(row)

    run_df = pd.DataFrame(all_rows)
    summary_df = (
        run_df.groupby(["dataset", "feature_type"])
        .agg(
            runs=("run", "count"),
            test_acc_mean=("test_acc", "mean"),
            test_acc_std=("test_acc", "std"),
            test_f1_macro_mean=("test_f1_macro", "mean"),
            test_f1_macro_std=("test_f1_macro", "std"),
            val_acc_mean=("val_acc", "mean"),
            val_f1_macro_mean=("val_f1_macro", "mean"),
        )
        .reset_index()
    )

    run_df.to_csv(output_dir / "per_run_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    return run_df, summary_df


def main() -> None:
    args = parse_args()
    run_df, summary_df = run_experiments(args)
    print("\nPer-run metrics:")
    print(run_df)
    print("\nSummary metrics:")
    print(summary_df)


if __name__ == "__main__":
    main()
