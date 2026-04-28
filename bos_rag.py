"""
Local BOS literature retrieval for the LLM agent.

The agent keeps tool-selection prompts short. This module retrieves compact
guideline context only after detection results are available.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
PAPER_DIR = BASE_DIR / "paper"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
GUIDELINE_CARDS_PATH = KNOWLEDGE_DIR / "bos_guideline_cards.json"

PAPER_META = [
    {
        "file": "BOS diagnosis-NIH.pdf",
        "label": "NIH 2014 Chronic GVHD Diagnosis & Staging Consensus",
        "key_sections": ["diagnosis", "staging", "BOS criteria", "lung scoring"],
    },
    {
        "file": "BOS中国指南.pdf",
        "label": "Chinese Expert Consensus on BOS Diagnosis & Treatment (2022)",
        "key_sections": ["diagnostic criteria", "grading", "treatment", "monitoring", "prevention"],
    },
    {
        "file": "Eur Respir J-2024-Bos-Treatment.pdf",
        "label": "ERS/EBMT 2024 Clinical Practice Guidelines on Treatment of Pulmonary cGvHD-BOS",
        "key_sections": ["ICS/LABA", "FAM therapy", "ibrutinib", "ruxolitinib", "lung transplant", "follow-up"],
    },
]

_bos_rag_index_cache: Optional[Dict[str, Any]] = None
_guideline_cards_cache: Optional[List[Dict[str, Any]]] = None


def _load_guideline_cards() -> List[Dict[str, Any]]:
    """Load structured guideline cards from knowledge/."""
    global _guideline_cards_cache
    if _guideline_cards_cache is not None:
        return _guideline_cards_cache

    try:
        with open(GUIDELINE_CARDS_PATH, "r", encoding="utf-8") as file:
            cards = json.load(file)
    except Exception:
        cards = []

    _guideline_cards_cache = [card for card in cards if isinstance(card, dict)]
    return _guideline_cards_cache


def _extract_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extract page-level plain text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(pdf_path))
        return [
            (page_number, page.extract_text() or "")
            for page_number, page in enumerate(reader.pages, start=1)
        ]
    except Exception:
        return []


def _normalize_literature_text(text: str) -> str:
    """Normalize extracted PDF text while preserving English and Chinese content."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _guess_bos_section(text: str) -> str:
    """Assign a coarse section label for retrieval display and routing."""
    lower = text.lower()
    if any(term in lower for term in ["lung transplant", "end-stage", "ecp", "ibrutinib", "ruxolitinib"]):
        return "advanced treatment"
    if any(term in lower for term in ["ics", "laba", "fam", "azithromycin", "montelukast", "budesonide"]):
        return "initial or inhaled treatment"
    if any(term in lower for term in ["spirometry", "follow-up", "pft", "pulmonary rehabilitation", "rehabilitation"]):
        return "follow-up and rehabilitation"
    if any(term in lower for term in ["diagnosis", "diagnostic", "fev1", "lung score"]) or any(
        term in text for term in ["诊断", "肺功能", "分级"]
    ):
        return "diagnosis and staging"
    if any(term in text for term in ["治疗", "推荐", "随访", "康复"]):
        return "treatment and monitoring"
    return "general BOS guidance"


def _chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> List[str]:
    """Split extracted page text into small overlapping chunks."""
    cleaned = _normalize_literature_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def _build_bos_rag_index() -> Dict[str, Any]:
    """Build and cache the local BOS literature retrieval index."""
    global _bos_rag_index_cache
    if _bos_rag_index_cache is not None:
        return _bos_rag_index_cache

    chunks: List[Dict[str, Any]] = []
    for meta in PAPER_META:
        pdf_path = PAPER_DIR / meta["file"]
        if not pdf_path.exists():
            continue
        for page_number, page_text in _extract_pdf_pages(pdf_path):
            for chunk in _chunk_text(page_text):
                chunks.append(
                    {
                        "source": meta["label"],
                        "file": meta["file"],
                        "page": page_number,
                        "section": _guess_bos_section(chunk),
                        "key_sections": meta["key_sections"],
                        "text": chunk,
                    }
                )

    index: Dict[str, Any] = {"chunks": chunks, "vectorizer": None, "matrix": None}
    if chunks:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                lowercase=True,
                max_features=50000,
            )
            matrix = vectorizer.fit_transform(
                [
                    " ".join(
                        [
                            str(chunk["source"]),
                            str(chunk["section"]),
                            " ".join(chunk.get("key_sections", [])),
                            str(chunk["text"]),
                        ]
                    )
                    for chunk in chunks
                ]
            )
            index["vectorizer"] = vectorizer
            index["matrix"] = matrix
        except Exception:
            index["vectorizer"] = None
            index["matrix"] = None

    _bos_rag_index_cache = index
    return index


def flatten_for_retrieval(value: Any, max_chars: int = 6000) -> str:
    """Convert nested detection results into compact retrieval text."""
    try:
        text = json.dumps(value, ensure_ascii=False)
    except Exception:
        text = str(value)
    text = _normalize_literature_text(text)
    return text[:max_chars]


def _score_guideline_card(card: Dict[str, Any], query: str) -> int:
    """Score structured cards with keyword hits and simple clinical routing."""
    query_lower = query.lower()
    score = 0
    for keyword in card.get("keywords", []):
        if str(keyword).lower() in query_lower:
            score += 3

    card_id = card.get("id", "")
    high_risk_terms = ["high-risk", "high risk", "risk_probability", "prediction_label", "positive"]
    if card_id in {"treatment_chinese_2022", "treatment_ers_ebmt_2024", "followup_rehab_ers_ebmt_2024"}:
        if any(term in query_lower for term in high_risk_terms):
            score += 4
    if card_id == "followup_rehab_ers_ebmt_2024":
        if any(term in query_lower for term in ["recheck", "repeat", "trend", "follow-up", "followup"]):
            score += 5
    if card_id == "diagnosis_nih_2014":
        if any(term in query_lower for term in ["diagnosis", "staging", "lung score", "threshold", "criteria"]):
            score += 5
        score += 1
    return score


def retrieve_bos_context(
    query: str,
    max_cards: int = 4,
    max_chunks: int = 6,
    max_chars: int = 7000,
) -> str:
    """
    Retrieve compact BOS literature context for the final LLM response.

    This function intentionally runs after detection, not during tool selection.
    It combines structured guideline cards with page-level PDF chunk retrieval.
    """
    query = _normalize_literature_text(query)
    if not query:
        return ""

    guideline_cards = _load_guideline_cards()
    index = _build_bos_rag_index()
    sections: List[str] = []
    used_chars = 0

    scored_cards = sorted(
        ((card, _score_guideline_card(card, query)) for card in guideline_cards),
        key=lambda item: item[1],
        reverse=True,
    )
    selected_cards = [card for card, score in scored_cards if score > 0][:max_cards]
    if not selected_cards:
        selected_cards = [card for card, _score in scored_cards[:2]]

    card_blocks = []
    for card in selected_cards:
        card_blocks.append(
            f"- [{card['id']}] Source: {card['source']}; pages: {card['pages']}; "
            f"applies to: {', '.join(card['applies_to'])}.\n"
            f"  Summary: {card['content']}"
        )
    if card_blocks:
        card_text = "Structured guideline cards:\n" + "\n".join(card_blocks)
        sections.append(card_text)
        used_chars += len(card_text)

    chunks = index.get("chunks") or []
    vectorizer = index.get("vectorizer")
    matrix = index.get("matrix")
    selected_chunks: List[Dict[str, Any]] = []

    if chunks and vectorizer is not None and matrix is not None:
        try:
            query_vector = vectorizer.transform([query])
            scores = (matrix @ query_vector.T).toarray().ravel()
            ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
            seen = set()
            for idx in ranked_indices:
                if scores[idx] <= 0:
                    break
                chunk = chunks[idx]
                key = (chunk["source"], chunk["page"], chunk["section"])
                if key in seen:
                    continue
                seen.add(key)
                selected_chunks.append(chunk)
                if len(selected_chunks) >= max_chunks:
                    break
        except Exception:
            selected_chunks = []

    if not selected_chunks:
        query_lower = query.lower()
        for chunk in chunks:
            haystack = f"{chunk['source']} {chunk['section']} {chunk['text']}".lower()
            if any(term in haystack for term in query_lower.split()[:20]):
                selected_chunks.append(chunk)
            if len(selected_chunks) >= max_chunks:
                break

    chunk_blocks = []
    for chunk in selected_chunks:
        snippet = chunk["text"][:900]
        block = (
            f"- Source: {chunk['source']}; page: {chunk['page']}; section: {chunk['section']}.\n"
            f"  Excerpt: {snippet}"
        )
        if used_chars + len(block) > max_chars:
            break
        chunk_blocks.append(block)
        used_chars += len(block)

    if chunk_blocks:
        sections.append("Retrieved PDF excerpts:\n" + "\n".join(chunk_blocks))

    if not sections:
        return ""

    return (
        "=== Retrieved BOS Guideline Context ===\n"
        "Use only the relevant points below to support clinical interpretation. "
        "Cite the named guideline or consensus only when the statement is directly supported here. "
        "Do not treat these excerpts as a substitute for formal clinical evaluation.\n\n"
        + "\n\n".join(sections)
    )
