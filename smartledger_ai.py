import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"

CATEGORICAL_INPUT_FIELDS = [
    "user_id",
    "company_id",
    "transaction_type",
    "customer_supplier_id",
    "weekday",
    "month",
    "payment_direction",
]

NUMERIC_INPUT_FIELDS = ["hour", "amount"]

TEXT_INPUT_FIELD = "description"

OUTPUT_FIELDS = [
    "category",
    "account",
    "tax_type",
    "payment_method",
    "description_template",
]


class LabelEncoder:
    """Simple label encoder with reserved unknown token support."""

    def __init__(self, add_pad: bool = False) -> None:
        self.add_pad = add_pad
        self.token_to_index: Dict[str, int] = {}
        self.index_to_token: List[str] = []
        self.fitted = False

    def fit(self, values: List[Any]) -> "LabelEncoder":
        unique_values = [UNK_TOKEN]
        if self.add_pad:
            unique_values.insert(0, PAD_TOKEN)

        seen = set(unique_values)
        for value in values:
            token = self._normalize_token(value)
            if token not in seen:
                unique_values.append(token)
                seen.add(token)

        self.index_to_token = unique_values
        self.token_to_index = {token: idx for idx, token in enumerate(self.index_to_token)}
        self.fitted = True
        return self

    def transform(self, value: Any) -> int:
        self._check_is_fitted()
        token = self._normalize_token(value)
        return self.token_to_index.get(token, self.token_to_index[UNK_TOKEN])

    def inverse_transform(self, index: int) -> str:
        self._check_is_fitted()
        if 0 <= index < len(self.index_to_token):
            return self.index_to_token[index]
        return UNK_TOKEN

    def size(self) -> int:
        self._check_is_fitted()
        return len(self.index_to_token)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "add_pad": self.add_pad,
            "index_to_token": self.index_to_token,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "LabelEncoder":
        encoder = cls(add_pad=state["add_pad"])
        encoder.index_to_token = list(state["index_to_token"])
        encoder.token_to_index = {token: idx for idx, token in enumerate(encoder.index_to_token)}
        encoder.fitted = True
        return encoder

    @staticmethod
    def _normalize_token(value: Any) -> str:
        if value is None:
            return UNK_TOKEN
        token = str(value).strip()
        return token if token else UNK_TOKEN

    def _check_is_fitted(self) -> None:
        if not self.fitted:
            raise ValueError("LabelEncoder must be fitted before use.")


class SimpleTextVectorizer:
    """Lightweight tokenizer + bag-of-words counter vectorizer."""

    def __init__(self, max_vocab_size: int = 1000, min_freq: int = 1) -> None:
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.vocab: Dict[str, int] = {}
        self.index_to_token: List[str] = []
        self.fitted = False

    def fit(self, texts: List[Optional[str]]) -> "SimpleTextVectorizer":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self.tokenize(text))

        kept_tokens = [
            token
            for token, freq in counter.most_common(self.max_vocab_size)
            if freq >= self.min_freq
        ]

        self.index_to_token = [UNK_TOKEN] + kept_tokens
        self.vocab = {token: idx for idx, token in enumerate(self.index_to_token)}
        self.fitted = True
        return self

    def transform(self, text: Optional[str]) -> torch.Tensor:
        self._check_is_fitted()
        vector = torch.zeros(len(self.index_to_token), dtype=torch.float32)
        tokens = self.tokenize(text)

        if not tokens:
            vector[self.vocab[UNK_TOKEN]] = 1.0
            return vector

        for token in tokens:
            index = self.vocab.get(token, self.vocab[UNK_TOKEN])
            vector[index] += 1.0

        # Normalize by token count so long descriptions do not dominate.
        vector /= max(float(len(tokens)), 1.0)
        return vector

    def vocab_size(self) -> int:
        self._check_is_fitted()
        return len(self.index_to_token)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "max_vocab_size": self.max_vocab_size,
            "min_freq": self.min_freq,
            "index_to_token": self.index_to_token,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "SimpleTextVectorizer":
        vectorizer = cls(
            max_vocab_size=state["max_vocab_size"],
            min_freq=state["min_freq"],
        )
        vectorizer.index_to_token = list(state["index_to_token"])
        vectorizer.vocab = {
            token: idx for idx, token in enumerate(vectorizer.index_to_token)
        }
        vectorizer.fitted = True
        return vectorizer

    @staticmethod
    def tokenize(text: Optional[str]) -> List[str]:
        if not text:
            return []
        return re.findall(r"[a-z0-9]+", text.lower())

    def _check_is_fitted(self) -> None:
        if not self.fitted:
            raise ValueError("SimpleTextVectorizer must be fitted before use.")


@dataclass
class NumericNormalizer:
    """Stores mean/std for numeric columns and applies z-score normalization."""

    means: Dict[str, float]
    stds: Dict[str, float]

    @classmethod
    def fit(cls, rows: List[Dict[str, Any]], numeric_fields: List[str]) -> "NumericNormalizer":
        means: Dict[str, float] = {}
        stds: Dict[str, float] = {}

        for field in numeric_fields:
            values = [float(row.get(field, 0.0) or 0.0) for row in rows]
            mean = sum(values) / max(len(values), 1)
            variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
            std = math.sqrt(variance)
            means[field] = mean
            stds[field] = std if std > 1e-8 else 1.0

        return cls(means=means, stds=stds)

    def transform(self, row: Dict[str, Any], numeric_fields: List[str]) -> torch.Tensor:
        values = []
        for field in numeric_fields:
            raw_value = safe_float(row.get(field), default=self.means[field], field_name=field)
            normalized = (raw_value - self.means[field]) / self.stds[field]
            values.append(normalized)
        return torch.tensor(values, dtype=torch.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {"means": self.means, "stds": self.stds}

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "NumericNormalizer":
        return cls(means=dict(state["means"]), stds=dict(state["stds"]))


def embedding_dim_for_size(cardinality: int) -> int:
    """Small embedding heuristic that keeps the model lightweight."""
    return min(32, max(4, int(math.ceil(math.sqrt(cardinality)) + 1)))


def safe_float(value: Any, default: float, field_name: str) -> float:
    """Convert numeric inputs safely and raise a clear error for bad values."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Field '{field_name}' must be numeric, but received: {value!r}"
        ) from exc


class JournalDataset(Dataset):
    """Dataset that encodes mixed journal-entry inputs for training."""

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        input_encoders: Dict[str, LabelEncoder],
        output_encoders: Dict[str, LabelEncoder],
        text_vectorizer: SimpleTextVectorizer,
        numeric_normalizer: NumericNormalizer,
    ) -> None:
        self.rows = rows
        self.input_encoders = input_encoders
        self.output_encoders = output_encoders
        self.text_vectorizer = text_vectorizer
        self.numeric_normalizer = numeric_normalizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]

        categorical_tensor = torch.tensor(
            [self.input_encoders[field].transform(row.get(field)) for field in CATEGORICAL_INPUT_FIELDS],
            dtype=torch.long,
        )

        numeric_tensor = self.numeric_normalizer.transform(row, NUMERIC_INPUT_FIELDS)
        text_tensor = self.text_vectorizer.transform(row.get(TEXT_INPUT_FIELD))

        targets = {
            field: torch.tensor(self.output_encoders[field].transform(row.get(field)), dtype=torch.long)
            for field in OUTPUT_FIELDS
        }

        return {
            "categorical": categorical_tensor,
            "numeric": numeric_tensor,
            "text": text_tensor,
            "targets": targets,
        }


class SmartLedgerModel(nn.Module):
    """
    Neural network with:
    - one embedding per categorical input field
    - normalized numeric inputs
    - a lightweight text projection for bag-of-words input
    - shared dense layers
    - separate classification heads for each predicted field
    """

    def __init__(
        self,
        categorical_cardinalities: Dict[str, int],
        numeric_dim: int,
        text_vocab_size: int,
        output_sizes: Dict[str, int],
        shared_hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.categorical_fields = list(CATEGORICAL_INPUT_FIELDS)
        self.embedding_layers = nn.ModuleDict()
        embedding_output_dim = 0

        for field in self.categorical_fields:
            cardinality = categorical_cardinalities[field]
            emb_dim = embedding_dim_for_size(cardinality)
            self.embedding_layers[field] = nn.Embedding(cardinality, emb_dim)
            embedding_output_dim += emb_dim

        text_projection_dim = 64
        self.text_encoder = nn.Sequential(
            nn.Linear(text_vocab_size, text_projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        combined_dim = embedding_output_dim + numeric_dim + text_projection_dim
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_dim, shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_hidden_dim, shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output_heads = nn.ModuleDict(
            {
                field: nn.Linear(shared_hidden_dim, output_sizes[field])
                for field in OUTPUT_FIELDS
            }
        )

    def forward(
        self,
        categorical_inputs: torch.Tensor,
        numeric_inputs: torch.Tensor,
        text_inputs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        embedded_parts = []

        for index, field in enumerate(self.categorical_fields):
            embedded = self.embedding_layers[field](categorical_inputs[:, index])
            embedded_parts.append(embedded)

        categorical_features = torch.cat(embedded_parts, dim=1)
        text_features = self.text_encoder(text_inputs)
        combined = torch.cat([categorical_features, numeric_inputs, text_features], dim=1)
        shared = self.shared_layers(combined)

        return {field: head(shared) for field, head in self.output_heads.items()}


def validate_row(row: Dict[str, Any], require_outputs: bool = True) -> None:
    required_inputs = CATEGORICAL_INPUT_FIELDS + NUMERIC_INPUT_FIELDS
    required_fields = required_inputs + OUTPUT_FIELDS if require_outputs else required_inputs

    missing = [field for field in required_fields if field not in row]
    if missing:
        raise ValueError(f"Row is missing required fields: {missing}")

    for field in NUMERIC_INPUT_FIELDS:
        safe_float(row.get(field), default=0.0, field_name=field)


def build_encoders(
    rows: List[Dict[str, Any]],
) -> Tuple[
    Dict[str, LabelEncoder],
    Dict[str, LabelEncoder],
    SimpleTextVectorizer,
    NumericNormalizer,
]:
    input_encoders: Dict[str, LabelEncoder] = {}
    output_encoders: Dict[str, LabelEncoder] = {}

    for field in CATEGORICAL_INPUT_FIELDS:
        encoder = LabelEncoder()
        encoder.fit([row.get(field) for row in rows])
        input_encoders[field] = encoder

    for field in OUTPUT_FIELDS:
        encoder = LabelEncoder()
        encoder.fit([row.get(field) for row in rows])
        output_encoders[field] = encoder

    text_vectorizer = SimpleTextVectorizer(max_vocab_size=200, min_freq=1)
    text_vectorizer.fit([row.get(TEXT_INPUT_FIELD) for row in rows])

    numeric_normalizer = NumericNormalizer.fit(rows, NUMERIC_INPUT_FIELDS)
    return input_encoders, output_encoders, text_vectorizer, numeric_normalizer


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    categorical = torch.stack([item["categorical"] for item in batch])
    numeric = torch.stack([item["numeric"] for item in batch])
    text = torch.stack([item["text"] for item in batch])
    targets = {
        field: torch.stack([item["targets"][field] for item in batch])
        for field in OUTPUT_FIELDS
    }
    return {"categorical": categorical, "numeric": numeric, "text": text, "targets": targets}


def train_model(
    data: List[Dict[str, Any]],
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 0.001,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train the SmartLedger model and return a reusable model bundle.

    The returned bundle contains the model, encoders, normalizer, and metadata
    needed to run predictions or save/load the artifact.
    """
    if not data:
        raise ValueError("Training data cannot be empty.")

    for row in data:
        validate_row(row, require_outputs=True)

    random.seed(seed)
    torch.manual_seed(seed)

    input_encoders, output_encoders, text_vectorizer, numeric_normalizer = build_encoders(data)

    dataset = JournalDataset(
        rows=data,
        input_encoders=input_encoders,
        output_encoders=output_encoders,
        text_vectorizer=text_vectorizer,
        numeric_normalizer=numeric_normalizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    categorical_cardinalities = {
        field: input_encoders[field].size() for field in CATEGORICAL_INPUT_FIELDS
    }
    output_sizes = {field: output_encoders[field].size() for field in OUTPUT_FIELDS}

    model = SmartLedgerModel(
        categorical_cardinalities=categorical_cardinalities,
        numeric_dim=len(NUMERIC_INPUT_FIELDS),
        text_vocab_size=text_vectorizer.vocab_size(),
        output_sizes=output_sizes,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()

            outputs = model(
                categorical_inputs=batch["categorical"],
                numeric_inputs=batch["numeric"],
                text_inputs=batch["text"],
            )

            total_loss = 0.0
            for field in OUTPUT_FIELDS:
                total_loss = total_loss + loss_fn(outputs[field], batch["targets"][field])

            total_loss.backward()
            optimizer.step()
            epoch_loss += float(total_loss.item())

        average_loss = epoch_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {average_loss:.4f}")

    return {
        "model": model,
        "input_encoders": input_encoders,
        "output_encoders": output_encoders,
        "text_vectorizer": text_vectorizer,
        "numeric_normalizer": numeric_normalizer,
        "config": {
            "categorical_fields": CATEGORICAL_INPUT_FIELDS,
            "numeric_fields": NUMERIC_INPUT_FIELDS,
            "text_field": TEXT_INPUT_FIELD,
            "output_fields": OUTPUT_FIELDS,
        },
    }


def _prepare_single_input(model_bundle: Dict[str, Any], partial_input: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    prepared_input = dict(partial_input)

    for field in CATEGORICAL_INPUT_FIELDS:
        prepared_input.setdefault(field, UNK_TOKEN)

    for field in NUMERIC_INPUT_FIELDS:
        prepared_input.setdefault(field, model_bundle["numeric_normalizer"].means[field])

    prepared_input.setdefault(TEXT_INPUT_FIELD, "")

    categorical_tensor = torch.tensor(
        [
            model_bundle["input_encoders"][field].transform(prepared_input.get(field))
            for field in CATEGORICAL_INPUT_FIELDS
        ],
        dtype=torch.long,
    ).unsqueeze(0)

    numeric_tensor = model_bundle["numeric_normalizer"].transform(
        prepared_input,
        NUMERIC_INPUT_FIELDS,
    ).unsqueeze(0)

    text_tensor = model_bundle["text_vectorizer"].transform(
        prepared_input.get(TEXT_INPUT_FIELD)
    ).unsqueeze(0)

    return {
        "categorical": categorical_tensor,
        "numeric": numeric_tensor,
        "text": text_tensor,
    }


def predict_suggestions(
    model_bundle: Dict[str, Any],
    partial_input: Dict[str, Any],
    top_k: int = 3,
) -> Dict[str, List[Dict[str, float]]]:
    """
    Predict top-k suggestions with confidence for each output field.

    Unknown categorical values are safely mapped to <UNK>.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    prepared = _prepare_single_input(model_bundle, partial_input)
    model: SmartLedgerModel = model_bundle["model"]
    output_encoders: Dict[str, LabelEncoder] = model_bundle["output_encoders"]

    model.eval()
    with torch.no_grad():
        logits = model(
            categorical_inputs=prepared["categorical"],
            numeric_inputs=prepared["numeric"],
            text_inputs=prepared["text"],
        )

    predictions: Dict[str, List[Dict[str, float]]] = {}
    for field in OUTPUT_FIELDS:
        probabilities = torch.softmax(logits[field], dim=1)[0]
        k = min(top_k, probabilities.shape[0])
        top_probs, top_indices = torch.topk(probabilities, k=k)

        predictions[field] = []
        for prob, index in zip(top_probs.tolist(), top_indices.tolist()):
            predictions[field].append(
                {
                    "value": output_encoders[field].inverse_transform(index),
                    "confidence": round(float(prob), 4),
                }
            )

    return predictions


def save_model(model_bundle: Dict[str, Any], path: str) -> None:
    """Save model weights and preprocessing artifacts to disk."""
    payload = {
        "model_state_dict": model_bundle["model"].state_dict(),
        "input_encoders": {
            field: encoder.state_dict()
            for field, encoder in model_bundle["input_encoders"].items()
        },
        "output_encoders": {
            field: encoder.state_dict()
            for field, encoder in model_bundle["output_encoders"].items()
        },
        "text_vectorizer": model_bundle["text_vectorizer"].state_dict(),
        "numeric_normalizer": model_bundle["numeric_normalizer"].state_dict(),
        "config": model_bundle["config"],
        "model_hyperparams": {
            "categorical_cardinalities": {
                field: model_bundle["input_encoders"][field].size()
                for field in CATEGORICAL_INPUT_FIELDS
            },
            "numeric_dim": len(NUMERIC_INPUT_FIELDS),
            "text_vocab_size": model_bundle["text_vectorizer"].vocab_size(),
            "output_sizes": {
                field: model_bundle["output_encoders"][field].size()
                for field in OUTPUT_FIELDS
            },
        },
    }
    torch.save(payload, path)


def load_model(path: str) -> Dict[str, Any]:
    """Load a previously saved SmartLedger model bundle."""
    payload = torch.load(path, map_location="cpu")

    input_encoders = {
        field: LabelEncoder.from_state_dict(state)
        for field, state in payload["input_encoders"].items()
    }
    output_encoders = {
        field: LabelEncoder.from_state_dict(state)
        for field, state in payload["output_encoders"].items()
    }
    text_vectorizer = SimpleTextVectorizer.from_state_dict(payload["text_vectorizer"])
    numeric_normalizer = NumericNormalizer.from_state_dict(payload["numeric_normalizer"])

    model = SmartLedgerModel(
        categorical_cardinalities=payload["model_hyperparams"]["categorical_cardinalities"],
        numeric_dim=payload["model_hyperparams"]["numeric_dim"],
        text_vocab_size=payload["model_hyperparams"]["text_vocab_size"],
        output_sizes=payload["model_hyperparams"]["output_sizes"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "input_encoders": input_encoders,
        "output_encoders": output_encoders,
        "text_vectorizer": text_vectorizer,
        "numeric_normalizer": numeric_normalizer,
        "config": payload["config"],
    }


def build_fake_journal_data() -> List[Dict[str, Any]]:
    """Small synthetic dataset so the file can run immediately."""
    return [
        {
            "user_id": "user_1",
            "company_id": "company_1",
            "transaction_type": "outgoing",
            "payment_direction": "outgoing",
            "customer_supplier_id": "supplier_abc",
            "weekday": "Monday",
            "month": "April",
            "hour": 10,
            "amount": 500000,
            "description": "bought vegetables from supplier",
            "category": "Inventory Purchase",
            "account": "Cash",
            "tax_type": "VAT Included",
            "payment_method": "Cash",
            "description_template": "Goods purchased from supplier",
        },
        {
            "user_id": "user_1",
            "company_id": "company_1",
            "transaction_type": "outgoing",
            "payment_direction": "outgoing",
            "customer_supplier_id": "supplier_abc",
            "weekday": "Tuesday",
            "month": "April",
            "hour": 11,
            "amount": 620000,
            "description": "purchased fresh produce inventory",
            "category": "Inventory Purchase",
            "account": "Bank",
            "tax_type": "VAT Included",
            "payment_method": "Bank Transfer",
            "description_template": "Goods purchased from supplier",
        },
        {
            "user_id": "user_2",
            "company_id": "company_1",
            "transaction_type": "outgoing",
            "payment_direction": "outgoing",
            "customer_supplier_id": "supplier_office",
            "weekday": "Wednesday",
            "month": "April",
            "hour": 14,
            "amount": 120000,
            "description": "office stationery and printer paper",
            "category": "Office Expense",
            "account": "Cash",
            "tax_type": "VAT Included",
            "payment_method": "Cash",
            "description_template": "Office supplies purchased",
        },
        {
            "user_id": "user_2",
            "company_id": "company_1",
            "transaction_type": "outgoing",
            "payment_direction": "outgoing",
            "customer_supplier_id": "supplier_office",
            "weekday": "Thursday",
            "month": "May",
            "hour": 15,
            "amount": 185000,
            "description": "paid annual software subscription",
            "category": "Software Expense",
            "account": "Bank",
            "tax_type": "VAT Excluded",
            "payment_method": "Card",
            "description_template": "Subscription payment made",
        },
        {
            "user_id": "user_3",
            "company_id": "company_2",
            "transaction_type": "income",
            "payment_direction": "incoming",
            "customer_supplier_id": "customer_xyz",
            "weekday": "Friday",
            "month": "April",
            "hour": 9,
            "amount": 1500000,
            "description": "received payment for catering service",
            "category": "Sales Revenue",
            "account": "Bank",
            "tax_type": "VAT Included",
            "payment_method": "Bank Transfer",
            "description_template": "Customer payment received",
        },
        {
            "user_id": "user_3",
            "company_id": "company_2",
            "transaction_type": "income",
            "payment_direction": "incoming",
            "customer_supplier_id": "customer_xyz",
            "weekday": "Monday",
            "month": "May",
            "hour": 10,
            "amount": 980000,
            "description": "cash sale from retail order",
            "category": "Sales Revenue",
            "account": "Cash",
            "tax_type": "VAT Included",
            "payment_method": "Cash",
            "description_template": "Customer payment received",
        },
        {
            "user_id": "user_4",
            "company_id": "company_2",
            "transaction_type": "income",
            "payment_direction": "incoming",
            "customer_supplier_id": "customer_event",
            "weekday": "Saturday",
            "month": "May",
            "hour": 18,
            "amount": 2500000,
            "description": "advance payment for event booking",
            "category": "Deferred Revenue",
            "account": "Bank",
            "tax_type": "VAT Excluded",
            "payment_method": "Bank Transfer",
            "description_template": "Advance payment received from customer",
        },
        {
            "user_id": "user_4",
            "company_id": "company_2",
            "transaction_type": "outgoing",
            "payment_direction": "outgoing",
            "customer_supplier_id": "supplier_rent",
            "weekday": "Monday",
            "month": "May",
            "hour": 8,
            "amount": 800000,
            "description": "monthly warehouse rent paid",
            "category": "Rent Expense",
            "account": "Bank",
            "tax_type": "VAT Excluded",
            "payment_method": "Bank Transfer",
            "description_template": "Monthly rent payment",
        },
        {
            "user_id": "user_1",
            "company_id": "company_1",
            "transaction_type": "outgoing",
            "payment_direction": "outgoing",
            "customer_supplier_id": "supplier_logistics",
            "weekday": "Tuesday",
            "month": "May",
            "hour": 16,
            "amount": 300000,
            "description": "delivery fee for customer orders",
            "category": "Logistics Expense",
            "account": "Bank",
            "tax_type": "VAT Included",
            "payment_method": "Bank Transfer",
            "description_template": "Delivery service payment",
        },
        {
            "user_id": "user_2",
            "company_id": "company_1",
            "transaction_type": "income",
            "payment_direction": "incoming",
            "customer_supplier_id": "customer_corp",
            "weekday": "Wednesday",
            "month": "May",
            "hour": 13,
            "amount": 4200000,
            "description": "invoice payment received from corporate client",
            "category": "Accounts Receivable Collection",
            "account": "Bank",
            "tax_type": "VAT Included",
            "payment_method": "Bank Transfer",
            "description_template": "Invoice payment received",
        },
        {
            "user_id": "user_1",
            "company_id": "company_1",
            "transaction_type": "outgoing",
            "payment_direction": "outgoing",
            "customer_supplier_id": "supplier_marketing",
            "weekday": "Friday",
            "month": "May",
            "hour": 17,
            "amount": 450000,
            "description": "facebook advertising campaign payment",
            "category": "Marketing Expense",
            "account": "Card",
            "tax_type": "VAT Excluded",
            "payment_method": "Card",
            "description_template": "Marketing campaign payment",
        },
        {
            "user_id": "user_3",
            "company_id": "company_2",
            "transaction_type": "outgoing",
            "payment_direction": "outgoing",
            "customer_supplier_id": "supplier_utilities",
            "weekday": "Sunday",
            "month": "April",
            "hour": 19,
            "amount": 210000,
            "description": "paid electricity and water bill",
            "category": "Utilities Expense",
            "account": "Bank",
            "tax_type": "VAT Excluded",
            "payment_method": "Bank Transfer",
            "description_template": "Utility bill payment",
        },
    ]


if __name__ == "__main__":
    fake_data = build_fake_journal_data()

    print("Training SmartLedger AI demo model...")
    model_bundle = train_model(fake_data, epochs=25, batch_size=4, lr=0.003)

    model_path = "smartledger_demo_model.pt"
    save_model(model_bundle, model_path)
    print(f"Model saved to: {model_path}")

    loaded_bundle = load_model(model_path)
    print("Model reloaded from disk.")

    partial_form_input = {
        "user_id": "user_1",
        "company_id": "company_1",
        "transaction_type": "outgoing",
        "payment_direction": "outgoing",
        "customer_supplier_id": "supplier_abc",
        "weekday": "Monday",
        "month": "May",
        "hour": 10,
        "amount": 540000,
        "description": "vegetable purchase from supplier",
    }

    suggestions = predict_suggestions(loaded_bundle, partial_form_input, top_k=3)
    print("Predicted suggestions:")
    print(json.dumps(suggestions, indent=2))
