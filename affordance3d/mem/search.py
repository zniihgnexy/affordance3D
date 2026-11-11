"""Keyframe search for the L2 memory: affordance tokens and plain semantics first, a vision-language model after.

A text query is matched in two ways. "Semantic" compares the query with each whole keyframe (image-text
features), which is what a memory without affordances would do. "Affordance" looks only at the regions of the
affordance map: how well the crop around a region matches the query, plus how well the region's action type
matches the action the query asks for. The two are combined to rank keyframes; the best few, with their
regions numbered, are what the vision-language model reads (`mark_prompt`, `parse_mark`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

ACTION_PROMPTS = {
    "rotate": "rotate a knob or a dial",
    "key_press": "press a key or a button on a keyboard or a remote",
    "tip_push": "push a switch or a button with a fingertip",
    "hook_pull": "pull a handle to open a drawer, a door or a cabinet",
    "pinch_pull": "pinch a small knob and pull it",
    "hook_turn": "turn a door handle or a window handle",
    "foot_push": "push a pedal with the foot",
    "plug_in": "plug something into a socket",
    "unplug": "unplug a plug from a socket",
}
AFFINITY_TEMPERATURE = 0.02
NO_TOKEN = -1
EMPTY_KEYFRAME_Z = -3.0  # affordance score of a keyframe without any region, in standard deviations


@dataclass(frozen=True)
class KeyframeScores:
    semantic: np.ndarray  # (K,) query against the whole keyframe
    afford: np.ndarray  # (K,) best affordance token of the keyframe
    combined: np.ndarray  # (K,) what the ranking uses
    token_scores: np.ndarray  # (N,) score of every token
    best_token: np.ndarray  # (K,) token behind `afford`, NO_TOKEN for a keyframe without regions

    def ranking(self, by: str = "combined") -> np.ndarray:
        """Keyframe indices, best first (ties keep time order)."""
        return np.argsort(-getattr(self, by), kind="stable")


def standardise(values: np.ndarray) -> np.ndarray:
    """Zero mean and unit spread; all zeros when the values do not vary. Values must be finite."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    spread = values.std() if len(values) else 0.0
    return (values - values.mean()) / spread if spread > 0 else np.zeros_like(values)


def action_affinity(query: np.ndarray, action_texts: np.ndarray, temperature: float = AFFINITY_TEMPERATURE) -> np.ndarray:
    """How much the query asks for each action type: a softmax over its similarity to the action phrases."""
    logits = (action_texts @ query) / temperature
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def score_keyframes(query: np.ndarray, keyframe_vectors: np.ndarray, token_vectors: np.ndarray,
                    token_keyframe: Sequence[int], token_action: Sequence[int], affinity: np.ndarray | None = None,
                    action_weight: float = 1.0, afford_weight: float = 1.0) -> KeyframeScores:
    """Score every keyframe for a query (all vectors unit length; `affinity` = action_affinity of the query).

    Token score = standardised similarity of the query to the crop around the region, plus `action_weight`
    times how far the affinity of its action type lies above an even split (so a flat affinity adds nothing).
    A keyframe's affordance score is its best token; a keyframe without tokens gets minus infinity.
    """
    semantic = keyframe_vectors @ query
    token_scores = standardise(token_vectors @ query) if len(token_vectors) else np.zeros(0)
    if affinity is not None and len(token_vectors):
        token_scores = token_scores + action_weight * (affinity[np.asarray(token_action)] - 1.0 / len(affinity))
    afford = np.full(len(keyframe_vectors), -np.inf)
    best_token = np.full(len(keyframe_vectors), NO_TOKEN)
    for token, keyframe in enumerate(token_keyframe):
        if token_scores[token] > afford[keyframe]:
            afford[keyframe], best_token[keyframe] = token_scores[token], token
    has_token = best_token != NO_TOKEN
    afford_z = np.full(len(afford), EMPTY_KEYFRAME_Z)
    afford_z[has_token] = standardise(afford[has_token])
    combined = standardise(semantic) + afford_weight * afford_z
    return KeyframeScores(semantic, afford, combined, token_scores, best_token)


def first_hit(ranking: Sequence[int], relevant: set[int]) -> int | None:
    """1-based position of the first relevant item in a ranking, or None."""
    return next((position for position, item in enumerate(ranking, start=1) if item in relevant), None)


def recall_at(hits: Sequence[int | None], ks: Sequence[int]) -> dict[int, float]:
    """Share of queries whose first relevant item is within the top k."""
    return {k: float(np.mean([hit is not None and hit <= k for hit in hits])) if len(hits) else 0.0 for k in ks}


def mark_prompt(task: str, n_marks: int) -> str:
    """Question for the vision-language model about keyframes whose operable regions carry numbers."""
    return (f"These images are keyframes from a walk through one room. Boxes numbered 1 to {n_marks} mark parts "
            f"that a person can operate. Task: \"{task}\". Which numbered box must be operated to do the task? "
            f"Answer with the number only.")


def parse_mark(answer: str, n_marks: int) -> int | None:
    """The first number between 1 and `n_marks` in the model's answer, or None."""
    for text in re.findall(r"\d+", answer):
        if 1 <= int(text) <= n_marks:
            return int(text)
    return None
