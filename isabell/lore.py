"""World lore: a compact text for every chat turn, a full reference for retrieval,
both hot-reloaded when the files change."""
import logging
import os
import re

from .core import config

LORE_CONTEXT = ""       # full reference
LORE_CHAT_CONTEXT = ""  # compact, sent with every chat message
_lore_mtimes: dict[str, float] = {}


def _lore_paths() -> tuple[str, str]:
    full = config.get("LorePath", "world_lore.txt")
    return full, (config.get("LoreChatPath") or full)


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        logging.exception("Failed to read lore file %s", path)
        return ""


# Retrieval index over the FULL lore: (title, text, matching terms).
# Chat carries only the compact lore; when a message names something from the
# world, the matching section is appended just for that call.
_LORE_CHUNKS: list[tuple[str, str, set[str]]] = []

_LORE_STOP = {
    "the","this","that","they","there","when","what","where","while","their","these","those","some",
    "most","many","each","every","after","before","above","below","during","because","her","his","its",
    "she","and","but","for","not","with","from","into","over","under","been","were","have","has","had",
    "only","more","less","than","then","them","who","how","why","all","any","one","two","new","old",
    "human","humans","women","woman","men","man","people","years","year","time","first","last","other",
    "small","large","black","white","red","blue","green","background","current","player","game",
}


def _split_lore(text: str) -> list[tuple[str, str]]:
    """Split the lore into retrievable chunks: ### subsections where present, else ## sections."""
    chunks: list[tuple[str, str]] = []
    for section in re.split(r"\n(?=## )", text):
        section = section.strip()
        if not section:
            continue
        title = section.split("\n", 1)[0].lstrip("# ").strip()
        subs = re.split(r"\n(?=### )", section)
        if len(subs) > 1:
            if len(subs[0].strip()) > 200:
                chunks.append((title, subs[0].strip()))
            for sub in subs[1:]:
                sub = sub.strip()
                sub_title = sub.split("\n", 1)[0].lstrip("# ").strip()
                chunks.append((f"{title} / {sub_title}", sub))
        else:
            chunks.append((title, section))
    return chunks


def _lore_terms(text: str) -> set[str]:
    """Proper nouns in a chunk — the names a user might reference.

    A single capitalised word only counts when it appears mid-sentence; a capital
    after a period, bullet or line break is just sentence case, not a name.
    """
    terms: set[str] = set()
    # Headings ("### Aeron ...") and bolded labels ("- **Iron Creed:** ...") are
    # where this lore declares its names, so take every capitalised word there.
    for line in re.findall(r"^#{1,4}[ ]*(.+)$", text, re.M) + re.findall(r"\*\*([^*\n]{2,48}?):?\*\*", text):
        terms |= {w.lower() for w in re.findall(r"\b[A-Z][a-z']{3,}\b", line)}
        terms |= {m.lower() for m in re.findall(r"\b[A-Z][a-z']+(?:[ ][A-Z][a-z']+)+\b", line)}
    # Multi-word capitalised phrases anywhere, plus single words capitalised
    # mid-sentence (a capital after a period or line break is just sentence case).
    terms |= {m.lower() for m in re.findall(r"\b[A-Z][a-z']+(?:[ ][A-Z][a-z']+)+\b", text)}
    terms |= {m.lower() for m in re.findall(r"(?<=[a-z,;)] )([A-Z][a-z']{3,})\b", text)}
    return {t for t in terms if t not in _LORE_STOP}


def _build_lore_index():
    """Index the full lore, dropping terms too common to be a useful signal."""
    global _LORE_CHUNKS
    chunks = _split_lore(LORE_CONTEXT) if LORE_CONTEXT else []
    if not chunks:
        _LORE_CHUNKS = []
        return
    # A real proper noun is never written lowercase in the source text. This drops
    # sentence-start capitals ("Female", "Bound", "Horse") that would otherwise
    # match ordinary roleplay and pull in lore nobody asked for.
    lowercase_words = set(re.findall(r"\b[a-z][a-z']{2,}\b", LORE_CONTEXT))
    freq: dict[str, int] = {}
    per_chunk = []
    for title, text in chunks:
        terms = {t for t in _lore_terms(text) if " " in t or t not in lowercase_words}
        per_chunk.append((title, text, terms))
        for t in terms:
            freq[t] = freq.get(t, 0) + 1
    limit = max(2, int(len(chunks) * 0.4))  # a term in >40% of chunks discriminates nothing
    # Weight by rarity: a name unique to one section is a far stronger signal
    # than one sprinkled across several.
    _LORE_CHUNKS = [
        (t, x, {k: 1.0 / freq[k] for k in terms if freq[k] <= limit}) for t, x, terms in per_chunk
    ]
    logging.info(
        "Lore index: %d chunks, %d distinct terms",
        len(_LORE_CHUNKS), len({k for _, _, s in _LORE_CHUNKS for k in s}),  # noqa: E501
    )


def _query_words(q: str) -> set[str]:
    """Words from a query, plus singular forms so 'Vorgath's' and 'drakes' still match."""
    words = set(re.findall(r"[a-z']+", q))
    extra = set()
    for w in words:
        base = re.sub(r"'s$", "", w)
        if base != w:
            extra.add(base)
        for stem in (base, w):
            if stem.endswith("es") and len(stem) > 5:
                extra.add(stem[:-2])
            if stem.endswith("s") and len(stem) > 4:
                extra.add(stem[:-1])
    return words | extra


def retrieve_lore(query: str) -> str:
    """Return full-lore sections matching the query, within the configured budget."""
    if not config.get("LoreRetrievalEnabled", True) or not _LORE_CHUNKS:
        return ""
    q = (query or "").lower()
    if len(q) < 3:
        return ""
    words = _query_words(q)
    scored = []
    for title, text, terms in _LORE_CHUNKS:
        hits = sum(w for t, w in terms.items() if " " not in t and t in words)
        hits += sum(w for t, w in terms.items() if " " in t and t in q)
        if hits:
            scored.append((hits, title, text))
    if not scored:
        return ""
    scored.sort(key=lambda x: (-x[0], len(x[2])))
    budget = float(config.get("LoreRetrievalMaxTokens", 1400))
    picked, titles = [], []
    for hits, title, text in scored[: int(config.get("LoreRetrievalMaxChunks", 2))]:
        cost = len(text) / 3.6
        if cost > budget:
            continue
        picked.append(text)
        titles.append(f"{title}({hits:.2f})")
        budget -= cost
    if picked:
        logging.info("Lore retrieval: %s", ", ".join(titles))
    return "\n\n".join(picked)


def load_lore():
    """(Re)load both lore files. Safe to call repeatedly."""
    global LORE_CONTEXT, LORE_CHAT_CONTEXT, _lore_mtimes
    full_path, chat_path = _lore_paths()
    LORE_CONTEXT = _read_text(full_path) if os.path.exists(full_path) else ""
    LORE_CHAT_CONTEXT = _read_text(chat_path) if os.path.exists(chat_path) else LORE_CONTEXT
    _lore_mtimes = {p: os.path.getmtime(p) for p in {full_path, chat_path} if os.path.exists(p)}
    logging.info(
        "Loaded lore: full=%d chars (%s), chat=%d chars (%s)",
        len(LORE_CONTEXT), full_path, len(LORE_CHAT_CONTEXT), chat_path,
    )
    _build_lore_index()


def lore_changed() -> bool:
    return any(os.path.exists(p) and os.path.getmtime(p) > was for p, was in list(_lore_mtimes.items()))
