"""
ModernBERT Encoder for PhilTrace.

Replaces the old roberta-base encoder with answerdotai/ModernBERT-large (or base fallback).
Produces token-level contextual embeddings for target words in philosophical texts.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.logging_utils import get_logger
from backend.pipeline_config import EncoderConfig, get_pipeline_config

logger = get_logger(__name__)

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except Exception:  # noqa: BLE001
    torch = None
    AutoModel = None
    AutoTokenizer = None


FALLBACK_EMBEDDING_SIZE = 384


class ModernBERTEncoder:
    """
    Token-level contextual encoder using ModernBERT.

    For each target word occurrence in a text, extracts the hidden state(s)
    at the corresponding subword token positions and pools them into a single vector.
    """

    def __init__(self, config: Optional[EncoderConfig] = None):
        self.config = config or get_pipeline_config().encoder
        self.model = None
        self.tokenizer = None
        self.hidden_size = FALLBACK_EMBEDDING_SIZE
        self.model_name_loaded: Optional[str] = None

        if torch is not None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        self._load_model()

    def _load_model(self) -> None:
        """Load ModernBERT model with automatic fallback."""
        if AutoTokenizer is None or AutoModel is None or torch is None:
            logger.warning("Transformers/Torch unavailable, encoder will use hash embeddings fallback")
            return

        # Try primary model first, then fallback
        models_to_try = [self.config.model_name, self.config.fallback_model_name]

        for model_name in models_to_try:
            try:
                logger.info("Attempting to load encoder model: %s", model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                self.hidden_size = int(getattr(self.model.config, "hidden_size", FALLBACK_EMBEDDING_SIZE))
                self.model_name_loaded = model_name
                logger.info(
                    "Loaded encoder '%s' on %s (hidden_size=%d)",
                    model_name, self.device, self.hidden_size
                )
                return
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM loading '%s', trying fallback...", model_name)
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load '%s': %s", model_name, exc)
                continue

        # All models failed
        logger.error("All encoder models failed to load, using hash embedding fallback")
        self.model = None
        self.tokenizer = None
        self.hidden_size = FALLBACK_EMBEDDING_SIZE

    @property
    def is_loaded(self) -> bool:
        """Check if a real model is loaded (not fallback)."""
        return self.model is not None

    def get_model_info(self) -> Dict[str, str]:
        """Return model metadata for logging/display."""
        return {
            "name": self.model_name_loaded or "fallback_hash",
            "hidden_size": str(self.hidden_size),
            "device": str(self.device),
            "fallback_mode": str(not self.is_loaded),
            "subword_pooling": self.config.subword_pooling,
            "hidden_layer_index": str(self.config.hidden_layer_index),
            "max_length": str(self.config.max_length),
        }

    def encode_single(self, text: str, target_word: str) -> np.ndarray:
        """
        Encode a single text and extract the target word's contextual embedding.

        Args:
            text: The full sentence/passage containing the target word.
            target_word: The word to extract embedding for.

        Returns:
            Normalized embedding vector of shape (hidden_size,).
        """
        if not self.is_loaded:
            return self._fallback_embedding(text)

        return self._encode_with_model(text, target_word)

    def encode_batch(self, texts: List[str], target_word: str) -> List[np.ndarray]:
        """
        Encode a batch of texts and extract target word embeddings.

        Args:
            texts: List of sentences containing the target word.
            target_word: The word to extract embeddings for.

        Returns:
            List of normalized embedding vectors.
        """
        if not self.is_loaded:
            return [self._fallback_embedding(t) for t in texts]

        results = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self._encode_batch_with_model(batch, target_word)
            results.extend(batch_results)

        return results

    def _encode_with_model(self, text: str, target_word: str) -> np.ndarray:
        """Encode single text using the loaded model."""
        # Tokenize with offset mapping for target word localization
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            return_offsets_mapping=True,
            padding=False,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        offsets = inputs["offset_mapping"].squeeze(0).tolist()

        # Find target word token indices
        token_indices = self._find_target_token_indices(text, target_word, offsets)
        if not token_indices:
            # Fallback: use [CLS] or first token
            token_indices = [0]

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=(self.config.hidden_layer_index != -1),
            )

        # Select the appropriate hidden layer
        if self.config.hidden_layer_index == -1:
            hidden_states = outputs.last_hidden_state.squeeze(0)
        else:
            # output_hidden_states returns tuple of all layers
            all_hidden = outputs.hidden_states
            layer_idx = self.config.hidden_layer_index
            hidden_states = all_hidden[layer_idx].squeeze(0)

        # Pool target word tokens
        embedding = self._pool_token_embeddings(hidden_states, token_indices)

        # L2 normalize
        norm = np.linalg.norm(embedding) + 1e-8
        return embedding / norm

    def _encode_batch_with_model(self, texts: List[str], target_word: str) -> List[np.ndarray]:
        """Encode a batch of texts using the loaded model."""
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            return_offsets_mapping=True,
            padding=True,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        offsets_batch = inputs["offset_mapping"].tolist()

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=(self.config.hidden_layer_index != -1),
            )

        # Select the appropriate hidden layer
        if self.config.hidden_layer_index == -1:
            hidden_states_batch = outputs.last_hidden_state
        else:
            all_hidden = outputs.hidden_states
            layer_idx = self.config.hidden_layer_index
            hidden_states_batch = all_hidden[layer_idx]

        results = []
        for idx, text in enumerate(texts):
            offsets = offsets_batch[idx]
            token_indices = self._find_target_token_indices(text, target_word, offsets)
            if not token_indices:
                token_indices = [0]

            hidden_states = hidden_states_batch[idx]
            embedding = self._pool_token_embeddings(hidden_states, token_indices)

            # L2 normalize
            norm = np.linalg.norm(embedding) + 1e-8
            results.append(embedding / norm)

        return results

    def _pool_token_embeddings(self, hidden_states, token_indices: List[int]) -> np.ndarray:
        """Pool embeddings from multiple subword tokens."""
        if self.config.subword_pooling == "first":
            embedding = hidden_states[token_indices[0]].cpu().numpy()
        else:
            # Default: mean pooling
            embedding = hidden_states[token_indices].cpu().numpy().mean(axis=0)
        return embedding

    @staticmethod
    def _find_target_token_indices(
        text: str, target_word: str, offset_mapping: List[Tuple[int, int]]
    ) -> List[int]:
        """
        Find token indices corresponding to the target word in the text.

        Uses character-level offset mapping to precisely locate the target word's
        subword tokens, handling cases where the word is split into multiple pieces.
        """
        indices: List[int] = []
        text_lower = text.lower()
        word_lower = target_word.lower()

        # Find all occurrences of target word in text
        word_starts = []
        search_start = 0
        while True:
            pos = text_lower.find(word_lower, search_start)
            if pos == -1:
                break
            # Check word boundaries
            before_ok = (pos == 0) or not text_lower[pos - 1].isalnum()
            after_pos = pos + len(word_lower)
            after_ok = (after_pos >= len(text_lower)) or not text_lower[after_pos].isalnum()
            if before_ok and after_ok:
                word_starts.append((pos, after_pos))
            search_start = pos + 1

        if not word_starts:
            return indices

        # Use the first occurrence
        word_start, word_end = word_starts[0]

        # Find tokens that overlap with the target word span
        for idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start >= tok_end:
                continue  # Skip special tokens
            # Check overlap
            if tok_start >= word_start and tok_end <= word_end:
                indices.append(idx)
            elif tok_start < word_end and tok_end > word_start:
                # Partial overlap - include it
                indices.append(idx)

        return indices

    @staticmethod
    def _fallback_embedding(text: str) -> np.ndarray:
        """Hash-based fallback embedding when no model is available."""
        vec = np.zeros(FALLBACK_EMBEDDING_SIZE, dtype=np.float32)
        words = re.findall(r"\w+", text.lower())
        if not words:
            return vec
        for word in words:
            index = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % FALLBACK_EMBEDDING_SIZE  # noqa: S324
            vec[index] += 1.0
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm

    def compute_embedding_stats(self, vectors: List[np.ndarray]) -> Dict[str, Any]:
        """Compute statistics for a set of embeddings (for verification)."""
        if not vectors:
            return {"count": 0}

        arr = np.vstack(vectors)
        norms = np.linalg.norm(arr, axis=1)
        return {
            "count": len(vectors),
            "shape": arr.shape,
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
            "vector_mean": float(arr.mean()),
            "vector_std": float(arr.std()),
            "vector_min": float(arr.min()),
            "vector_max": float(arr.max()),
        }
