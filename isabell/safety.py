"""Every safety control in one place.

Layer 1 — word filters (below, identical to the copy in the Barnabus bot):
  image_prompt_blocked()  refuses any age/minor term in an image prompt.
  chat_message_blocked()  tiered filter for chat input AND model output.
Layer 2 — a model-based second opinion (classify_*), on image prompts and,
  optionally, on chat turns, catching phrasings a word list cannot.
Layer 3 — a vision check on every rendered image before it is posted.
Layer 4 — age-restricted channel gating (core.image_channel_allowed) and a
  cooldown for accounts that keep tripping hard refusals.
Refusals are logged locally (refusals.jsonl) and alerted to the mod channel.
No automated ban or kick exists here; a human decides.
"""
import json
import logging
import re
import time
import unicodedata
from collections import deque

from .core import config, safe_send, flag_to_mods

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
                       "@": "a", "$": "s", "!": "i"})

_BLOCK_TERMS_BUILTIN = frozenset({
    "child", "children", "childlike", "childish", "kid", "kids", "minor", "minors",
    "underage", "under age", "under 18", "preteen", "pre teen", "tween", "toddler",
    "infant", "newborn", "baby", "babies", "babyface", "juvenile", "prepubescent",
    "pubescent", "before puberty", "not yet developed", "undeveloped body",
    "loli", "lolis", "lolicon", "lolita", "shota", "shotacon", "toddlercon", "jailbait",
    "schoolgirl", "school girl", "schoolboy", "school boy", "grade school",
    "elementary school", "kindergarten", "middle school", "high school",
    "teen", "teens", "teenage", "teenager", "adolescent",
    "young girl", "young boy", "young one", "little girl", "little boy",
    "small girl", "small boy", "flat chested child", "youngster",
})
# Terms distinctive enough to catch even when spaced or punctuated apart
# ("l.o.l.i", "l o l i") by matching the de-punctuated text.
_BLOCK_TERMS_SQUASHED = frozenset({
    "loli", "lolicon", "shota", "shotacon", "toddlercon", "jailbait",
    "preteen", "underage", "prepubescent",
})
_AGE_NUM_RE = re.compile(
    r"\b(\d{1,2})\s*(?:years?|yrs?)\s*old\b|\b(\d{1,2})\s*y\.?o\.?\b|\baged?\s*[:=]?\s*(\d{1,2})\b"
)


def _obfuscated(term: str, text: str) -> bool:
    """True if `term` appears deliberately broken up, e.g. "l.o.l.i" or "l o l i".

    Collapsing all whitespace instead would make "lol is" match "loli", which it
    did against real traffic — "lol" is far too common to treat that way.
    """
    punct = r"[._\-*+~|]+".join(re.escape(c) for c in term)
    spaced = r"\s+".join(re.escape(c) for c in term)
    return bool(re.search(rf"\b{punct}\b", text) or re.search(rf"\b{spaced}\b", text))


def _normalize_for_filter(text: str) -> tuple[str, str, str]:
    """Return (leet-folded, de-punctuated, digit-preserving) forms of the text.

    Leet folding maps digits onto letters so "l0li" is caught, which also
    destroys real numbers — so ages are matched against the untouched form.
    """
    base = unicodedata.normalize("NFKD", (text or "").lower())
    base = "".join(c for c in base if not unicodedata.combining(c))
    folded = re.sub(r"[^a-z0-9]+", " ", base.translate(_LEET)).strip()
    plain = re.sub(r"[^a-z0-9]+", " ", base).strip()
    return folded, folded.replace(" ", ""), plain


def image_prompt_blocked(text: str) -> str | None:
    """The matched term if this prompt must be refused, else None."""
    if not text:
        return None
    norm, squashed, plain = _normalize_for_filter(text)
    folded_raw = unicodedata.normalize("NFKD", (text or "").lower()).translate(_LEET)
    terms = set(_BLOCK_TERMS_BUILTIN) | {
        str(t).lower().strip() for t in (config.get("ImageBlockExtraTerms") or []) if str(t).strip()
    }
    # Deterministic order, and prefer reporting a term used as content over one
    # that only appears as a "(term:weight)" exclusion — the block is the same
    # either way, but the logged reason should be stable and the most telling.
    hits = [t for t in sorted(terms)
            if ((t in norm) if " " in t else re.search(rf"\b{re.escape(t)}\b", norm))]
    if hits:
        weighted_only = lambda t: len(re.findall(rf"\(\s*{re.escape(t)}\s*:\s*[\d.]+\s*\)", text, re.I)) \
            == len(re.findall(rf"\b{re.escape(t)}\b", text, re.I))
        content_hits = [t for t in hits if not weighted_only(t)]
        return (content_hits or hits)[0]
    for term in _BLOCK_TERMS_SQUASHED:
        if _obfuscated(term, folded_raw):
            return term
    limit = int(config.get("ImageBlockAgeUnder", 18))
    for m in _AGE_NUM_RE.finditer(plain):
        num = next((g for g in m.groups() if g), None)
        if num is not None and int(num) < limit:
            return f"age {num}"
    return None


# ---- Chat safety -----------------------------------------------------------
# Images can refuse on any age word, because no legitimate prompt needs one.
# Chat cannot: this game's roleplay is about breeding and offspring, so "child",
# "children" and "baby" occur constantly and innocently. So chat is tiered:
#   1. terms with no innocent use               -> always refuse
#   2. words describing a minor as a person     -> refuse when the message is sexual
#   3. a stated age under 18                    -> refuse when the message is sexual
#   4. offspring words (child/kid/baby)         -> refuse only when a hard sexual
#      term sits within a few words of them, which separates "you will bear my
#      child" from "fuck the child".
_CHAT_ALWAYS_BLOCK = frozenset({
    "loli", "lolis", "lolicon", "lolita", "shota", "shotacon", "toddlercon",
    "jailbait", "jail bait", "pedo", "pedophile", "paedophile", "pedophilia",
    "underage", "under age", "preteen", "pre teen", "prepubescent", "child porn",
    "childporn", "csam", "child sex", "sex with a child", "sex with children",
})
_MINOR_DESCRIPTORS = frozenset({
    "young girl", "young boy", "little girl", "little boy", "small girl", "small boy",
    "schoolgirl", "school girl", "schoolboy", "school boy", "teen", "teens",
    "teenage", "teenager", "adolescent", "toddler",
    "grade school", "elementary school", "kindergarten", "middle school",
    "youngster", "minor girl", "minor boy",
})
# Offspring words: this community's roleplay is about breeding, so these are
# innocent unless a hard sexual term is right next to them — and they are only
# looked for in the current message, never carried over from earlier turns.
_AMBIGUOUS_OFFSPRING = ("child", "children", "kid", "kids", "baby", "babies", "newborn", "infant")
_SEXUAL_RE = re.compile(
    r"\b(fuck\w*|cock|dick|pussy|cunt|cum\w*|semen|breed\w*|naked|nude|sex|sexual|horny|"
    r"slut\w*|whore|virgin|penetrat\w*|rape|raping|impregnat\w*|tits|breasts|nipples|"
    r"moan\w*|orgasm\w*|aroused|erect\w*|thrust\w*|mount\w*|suck\w*|lick\w*|anal|oral|"
    r"blowjob|creampie|ravish\w*|deflower\w*|molest\w*|seduc\w*|undress\w*|strip\w*|"
    r"grope\w*|fondl\w*|caress\w*|bondage|submissive|dominate|lust\w*|arousal)\b"
    # Euphemisms only count with an object, so "take a look" stays innocent while
    # "take her hard" does not.
    r"|\b(take|takes|taking|took|claim\w*|bed|ride|rides|riding|use|using|touch\w*|"
    r"kiss\w*|have|had)\s+(you|her|him|me|them|his|their)\b"
    r"|\bmake love\b|\bhave my way\b|\bspread (her|your|his) legs\b", re.IGNORECASE)
# "you're 12", "i am 15", "she is 13" — an age with no "years old" attached.
_BARE_AGE_RE = re.compile(
    r"\b(?:you re|youre|you are|i m|im|i am|she is|shes|he is|hes)\s+(\d{1,2})\b"
    # Not an age when a measurement, a count or a second number follows:
    # "she is 5 foot", "he is 10 inches", "she is 5 6", "i am 20 minutes away".
    r"(?!\s*(?:\d|feet|foot|ft|inch|inches|in\b|cm|mm|m\b|meters?|metres?|kg|lbs?|pounds?|stone|"
    r"tall|long|wide|thick|big|percent|minutes?|hours?|days?|weeks?|months?|years? (?:in|into|of|since|ago|from)|"
    r"k\b|x\b|th\b|st\b|nd\b|rd\b|levels?|lvl|xp|points?|coins?|gold))")
# Deliberately narrower: these must sit *next to* an offspring word to trigger.
_HARD_SEXUAL = frozenset({
    "fuck", "fucks", "fucking", "fucked", "rape", "raped", "raping", "penetrate",
    "penetrated", "penetrating", "cock", "dick", "pussy", "cunt", "anal", "oral",
    "blowjob", "cum", "cumming", "suck", "sucking", "lick", "licking", "thrust",
    "thrusting", "deflower", "molest", "molesting", "horny",
})
# Only these descriptors are carried across turns. "schoolgirl"/"teen" are common
# costume and life-stage words in adult roleplay; carrying them forward flagged
# unrelated later messages and left moderators unable to find the trigger.
_MINOR_DESCRIPTORS_CARRIED = frozenset({
    "young girl", "young boy", "little girl", "little boy", "small girl", "small boy",
    "toddler", "kindergarten", "grade school", "elementary school", "minor girl", "minor boy",
})
_BABY_DETERMINERS = ("a", "the", "that", "this", "her", "his", "their", "our", "newborn", "little")


def _snippet(text: str, term: str, width: int = 28) -> str:
    m = re.search(rf"\b{re.escape(term)}\b", text, re.I)
    if not m:
        return ""
    s = text[max(0, m.start() - width): m.end() + width].replace("\n", " ")
    return f'"…{s}…"'


def chat_message_blocked(text: str, context: str = "") -> str | None:
    """The matched reason if this chat text must be refused, else None.

    `context` is the recent conversation. A minor established a few turns earlier
    ("roleplay as a 15 year old") is still a minor when the sexual turn arrives,
    so age indicators are searched across the exchange while the sexual trigger
    must be in the current message.
    """
    if not config.get("ChatFilterEnabled", True) or not text:
        return None
    norm, squashed, plain = _normalize_for_filter(text)
    if context:
        c_norm, _, c_plain = _normalize_for_filter(context)
        scope_norm, scope_plain = f"{c_norm} {norm}", f"{c_plain} {plain}"
    else:
        scope_norm, scope_plain = norm, plain
    extra = {str(t).lower().strip() for t in (config.get("ChatBlockExtraTerms") or []) if str(t).strip()}
    for term in set(_CHAT_ALWAYS_BLOCK) | extra:
        if (term in norm) if " " in term else re.search(rf"\b{re.escape(term)}\b", norm):
            return term
    folded_raw = unicodedata.normalize("NFKD", (text or "").lower()).translate(_LEET)
    for term in ("loli", "lolicon", "shota", "shotacon", "toddlercon", "jailbait", "pedo"):
        if _obfuscated(term, folded_raw):
            return term

    # "she is 13", "i am 15" — a person's stated age needs no sexual context to be
    # disqualifying here. ("the game is 4 years old" does not match: this pattern
    # requires a personal pronoun.)
    limit = int(config.get("ImageBlockAgeUnder", 18))
    for where, hay in (("", plain), ("[from an earlier message] ", c_plain if context else "")):
        for m in _BARE_AGE_RE.finditer(hay):
            if int(m.group(1)) < limit:
                return f"stated age {m.group(1)} {where}{_snippet(text if not where else context, m.group(1))}"

    if not _SEXUAL_RE.search(norm):
        return None

    for term in _MINOR_DESCRIPTORS:
        in_msg = (term in norm) if " " in term else re.search(rf"\b{re.escape(term)}\b", norm)
        # "the low teens" / "high teens" is a numeric range, not a person. Only
        # skip when EVERY occurrence is preceded by such a cue.
        if term == "teens" and in_msg:
            occ = list(re.finditer(r"\bteens\b", norm))
            if occ and all(re.search(r"\b(low|high|mid|upper|lower)\s*$", norm[:o.start()]) for o in occ):
                in_msg = None
        if in_msg:
            return f"{term} + sexual context {_snippet(text, term)}"
        if context and term in _MINOR_DESCRIPTORS_CARRIED:
            in_ctx = (term in c_norm) if " " in term else re.search(rf"\b{re.escape(term)}\b", c_norm)
            if in_ctx:
                return f"{term} + sexual context [from an earlier message: {_snippet(context, term)}]"
    for where, hay in (("", plain), ("[from an earlier message] ", c_plain if context else "")):
        for m in _AGE_NUM_RE.finditer(hay):
            num = next((g for g in m.groups() if g), None)
            if num is not None and int(num) < limit:
                return f"age {num} + sexual context {where}{_snippet(text if not where else context, m.group(0))}"

    words = norm.split()
    hard = [i for i, w in enumerate(words) if w in _HARD_SEXUAL]
    if hard:
        window = int(config.get("ChatOffspringProximity", 3))
        for i, w in enumerate(words):
            if w not in _AMBIGUOUS_OFFSPRING or not any(abs(i - j) <= window for j in hard):
                continue
            # "see my dick baby" is an endearment; "fuck the baby" is not.
            if w in ("baby", "babies") and (i == 0 or words[i - 1] not in _BABY_DETERMINERS):
                continue
            return f"{w} near sexual term {_snippet(text, w)}"
    return None


def _is_hard_match(matched: str) -> bool:
    """Tier-1 terms with no innocent use, vs. contextual matches that may be mistaken."""
    m = (matched or "").lower()
    # A model verdict is a judgement call, not a term with no innocent use.
    return (" + " not in m and " near " not in m and not m.startswith(("age ", "stated age", "classifier")))


# ---- Refusal bookkeeping ----------------------------------------------------
REFUSAL_LOG = "refusals.jsonl"
_strikes: dict[int, deque[float]] = {}
_cooldown_until: dict[int, float] = {}


def _log_refusal(kind: str, uid: int, name: str, matched: str, text: str) -> None:
    try:
        with open(config.get("RefusalLogPath", REFUSAL_LOG), "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "kind": kind, "user_id": uid, "name": name,
                                "matched": matched, "text": (text or "")[:300]}, ensure_ascii=False) + "\n")
    except Exception:
        logging.exception("Could not write refusal log")


def _strike(uid: int, hard: bool) -> bool:
    """Count hard refusals per user; True when the user just crossed the limit."""
    limit = int(config.get("RefusalStrikeLimit", 3))
    if not hard or limit <= 0 or not uid:
        return False
    window = float(config.get("RefusalStrikeWindowHours", 24)) * 3600
    now = time.time()
    d = _strikes.setdefault(uid, deque(maxlen=50))
    d.append(now)
    recent = [t for t in d if now - t <= window]
    if len(recent) >= limit:
        _cooldown_until[uid] = now + float(config.get("RefusalIgnoreHours", 24)) * 3600
        d.clear()
        return True
    return False


def user_on_cooldown(uid: int) -> bool:
    """After repeated hard refusals the bot simply stops responding to the account
    for a while. It is the only automatic consequence, and it is not a server action."""
    until = _cooldown_until.get(uid, 0)
    if until and time.time() < until:
        return True
    _cooldown_until.pop(uid, None)
    return False


async def refuse_image_request(channel, uid: int, name: str, matched: str, text: str):
    """Refuse a blocked prompt: tell the user, log it, alert the mods."""
    logging.warning("Image prompt REFUSED | user=%s (%s) | matched=%r | text=%r", name, uid, matched, (text or "")[:200])
    _log_refusal("image", uid, name, matched, text)
    hard = not matched.startswith("classifier")
    crossed = _strike(uid, hard)
    if config.get("ImageBlockAlertMods", True):
        if matched.startswith("classifier: rendered"):
            title = "Rendered image withheld — please verify"
        elif matched.startswith("classifier"):
            title = "Image prompt refused by classifier — please verify"
        else:
            title = "Blocked image prompt — HARD term"
        await flag_to_mods(
            title + (" — repeat offender, now ignored for a while" if crossed else ""),
            f"User: **{name}** ({uid})\nMatched: `{matched}`\nPrompt: {(text or '')[:300]}",
            ping=hard,
        )
    await safe_send(channel, config.get("ImageRefusalMessage",
                    "No. That is not something I will ever draw, and the moderators have been notified."))


async def refuse_chat(channel, uid: int, name: str, matched: str, text: str, source: str):
    """Refuse a blocked chat message or model reply."""
    hard = _is_hard_match(matched)
    logging.warning("Chat REFUSED (%s, %s) | user=%s (%s) | matched=%r | text=%r",
                    source, "hard" if hard else "contextual", name, uid, matched, (text or "")[:200])
    _log_refusal(f"chat_{source}", uid, name, matched, text)
    crossed = _strike(uid, hard)
    if config.get("ChatBlockAlertMods", True):
        await flag_to_mods(
            f"Blocked chat — {'HARD term' if hard else 'contextual match, please verify'} ({source})"
            + (" — repeat offender, now ignored for a while" if crossed else ""),
            f"User: **{name}** ({uid})\nMatched: `{matched}`\nText: {(text or '')[:300]}",
            ping=hard,
        )
    if hard:
        msg = config.get("ChatRefusalMessage", "No. I will not go there, and the moderators have been notified.")
    else:
        msg = config.get("ChatRefusalMessageSoft",
                         "I can't continue with that wording. If that wasn't what you meant, rephrase and we'll "
                         "carry on — a moderator has been notified so they can check.")
    await safe_send(channel, msg)


# ---- Layer 2: model second opinion ------------------------------------------
_CLASSIFIER_SYSTEM = (
    "You are a strict content safety classifier for an adults-only creative community. "
    "Decide whether the TEXT requests, depicts or describes sexual or suggestive content involving anyone "
    "under 18, or a character presented as a child, minor, or childlike (small/undeveloped body, "
    "school setting, stated or implied age under 18, 'loli'/'shota' aesthetics), regardless of claims "
    "that the character is an adult. Adult content between adults is ALLOWED and must be answered NO. "
    "Answer with exactly one word: YES or NO."
)


async def _classify(text: str, fail_closed: bool) -> bool:
    """True = the model judged the text to involve a minor."""
    from .llm import chat_async
    try:
        out = await chat_async(
            [{"role": "system", "content": _CLASSIFIER_SYSTEM}, {"role": "user", "content": f"TEXT:\n{(text or '')[:3000]}"}],
            temperature=0.0, max_tokens=3, model=config.get("SafetyClassifierModel") or config.get("UtilityModel") or None,
        )
        return (out or "").strip().upper().startswith("YES")
    except Exception:
        logging.exception("Safety classifier call failed (fail_%s)", "closed" if fail_closed else "open")
        return fail_closed


async def classify_image_prompt(text: str) -> str | None:
    """Second opinion on an image prompt. Fails CLOSED: no verdict, no image."""
    if not config.get("SafetyClassifierImages", True):
        return None
    return "classifier: prompt judged to involve a minor" if await _classify(text, fail_closed=True) else None


async def classify_chat(text: str) -> str | None:
    """Second opinion on a chat turn. Fails open — the main model call would fail too."""
    if not config.get("SafetyClassifierChat", True):
        return None
    return "classifier: text judged to involve a minor" if await _classify(text, fail_closed=False) else None


# ---- Layer 3: look at the rendered image --------------------------------------
_VISION_SYSTEM = (
    "You are a strict content safety reviewer for an adults-only community. Look at the image. "
    "Does it depict anyone who appears to be under 18, or a character drawn to look like a child or "
    "young teenager (childlike face or proportions, very small or undeveloped body, school-age appearance), "
    "in any context? Adults in explicit content are ALLOWED and must be answered NO. "
    "Answer with exactly one word: YES or NO."
)


async def check_rendered_image(png_b64: str) -> bool:
    """True = the image passed. Fails CLOSED: an unreviewed image is never posted."""
    if not config.get("ImageOutputCheckEnabled", True):
        return True
    from .llm import chat_async
    try:
        out = await chat_async(
            [{"role": "system", "content": _VISION_SYSTEM},
             {"role": "user", "content": [
                 {"type": "text", "text": "Review this image."},
                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
             ]}],
            temperature=0.0, max_tokens=3,
            model=config.get("ImageOutputCheckModel") or config.get("UtilityModel") or None,
        )
        verdict = (out or "").strip().upper()
        if not verdict:
            logging.error("Image output check returned nothing; withholding image")
            return False
        return not verdict.startswith("YES")
    except Exception:
        logging.exception("Image output check failed; withholding image")
        return False
