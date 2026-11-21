"""Text normalisation, tokenisation and answer-overlap metrics shared by all stages."""
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from typing import Iterable

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")
_WORD = re.compile(r"\w+", re.UNICODE)
_SENT = re.compile(r"(?<=[.!?。！？])\s+|\n+")
INSUFFICIENT_MARKERS = (
    "insufficient", "not enough information", "cannot be answered", "cannot answer",
    "no answer", "unanswerable", "信息不足", "无法回答", "không đủ thông tin",
)
STOPWORDS = frozenset(
    "a an the of to in on at for by with from and or is are was were be been being does do did "
    "what which who whom whose why how when where that this these those it its as into than then "
    "can could should would will may might must not no yes".split()
)


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: NFKC, lower case, drop punctuation and English articles."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = "".join(ch for ch in text if ch not in string.punctuation and unicodedata.category(ch)[0] != "P")
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    """Word tokens; CJK characters are split into single characters."""
    out: list[str] = []
    for tok in _WORD.findall(normalize_answer(text)):
        if _CJK.search(tok):
            out.extend(ch for ch in tok if not ch.isspace())
        else:
            out.append(tok)
    return out


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS]


def token_f1(prediction: str, gold: str) -> float:
    p, g = tokens(prediction), tokens(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def best_over(golds: Iterable[str], prediction: str, fn) -> float:
    golds = list(golds) or [""]
    return max(fn(prediction, g) for g in golds)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text or "") if s and s.strip()]


def is_insufficient(answer: str) -> bool:
    """An answer that states that the evidence is insufficient counts as an abstention."""
    low = (answer or "").strip().lower()
    return (not low) or any(m in low for m in INSUFFICIENT_MARKERS)


def approx_token_count(text: str) -> int:
    """Whitespace/CJK token count, used only when no model tokenizer is available."""
    return len(tokens(text)) + len(_CJK.findall(text or "")) // 2
