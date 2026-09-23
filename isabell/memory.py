"""Per-channel conversation windows with summaries.

Every turn carries a timestamp. Turns older than RetentionDays are deleted, and a
summary is deleted once the oldest conversation folded into it passes that age,
so no message content is held for longer than the retention period. Stored
encrypted via store.py.
"""
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field

from . import store
from .core import config
from .llm import chat_async, llm_enabled, utility_model


@dataclass
class Utterance:
    author_id: int
    author_name: str
    content: str
    message_id: int
    ts: float


@dataclass
class ConversationWindow:
    channel_id: int
    turns: deque = field(default_factory=lambda: deque(maxlen=40))   # (role, content, ts)
    utterances: deque = field(default_factory=lambda: deque(maxlen=100))
    summary: str = ""
    summary_since: float = 0.0   # ts of the oldest turn folded into the summary
    is_dm: bool = False

    def pairs(self) -> list[tuple[str, str]]:
        return [(r, c) for r, c, _ in self.turns]


class ConversationManager:
    def __init__(self, maxlen_turns: int = 40, channel_history_path: str | None = None,
                 dm_history_dir: str | None = None):
        self._by_channel: dict[int, ConversationWindow] = {}
        self.maxlen_turns = maxlen_turns
        self.channel_history_path = channel_history_path
        self.dm_history_dir = dm_history_dir
        self._dirty = False
        self._compress_at = maxlen_turns - 2
        if self.dm_history_dir:
            os.makedirs(self.dm_history_dir, exist_ok=True)
        self._load()

    # ---------- persistence ----------
    def _window_from(self, d: dict, ch_id: int, is_dm: bool) -> ConversationWindow:
        now = time.time()
        turns = deque(maxlen=self.maxlen_turns)
        for t in d.get("turns", []):
            turns.append((t[0], t[1], float(t[2]) if len(t) > 2 else now))   # legacy turns get "now"
        uttrs = deque(maxlen=100)
        for u in d.get("utterances", []):
            uttrs.append(Utterance(u.get("author_id"), u.get("author_name", ""), u.get("content", ""),
                                   u.get("message_id", 0), u.get("ts", now)))
        return ConversationWindow(channel_id=ch_id, turns=turns, utterances=uttrs,
                                  summary=d.get("summary", ""),
                                  summary_since=float(d.get("summary_since") or (now if d.get("summary") else 0)),
                                  is_dm=is_dm)

    @staticmethod
    def _window_to(cv: ConversationWindow) -> dict:
        return {"channel_id": cv.channel_id, "turns": [list(t) for t in cv.turns],
                "utterances": [u.__dict__ for u in cv.utterances], "summary": cv.summary,
                "summary_since": cv.summary_since, "is_dm": cv.is_dm}

    def _load(self):
        path = self.channel_history_path
        if path and os.path.exists(path):
            try:
                for key, d in store.read_json(path).items():
                    ch = int(d.get("channel_id", key))
                    self._by_channel[ch] = self._window_from(d, ch, False)
                logging.info("Loaded %d channel windows", len(self._by_channel))
            except Exception:
                logging.exception("Failed to load channel history from %s", path)
        if self.dm_history_dir and os.path.isdir(self.dm_history_dir):
            n = 0
            for fname in os.listdir(self.dm_history_dir):
                if not fname.endswith(".json"):
                    continue
                try:
                    d = store.read_json(os.path.join(self.dm_history_dir, fname))
                    ch = int(d.get("channel_id", fname[:-5]))
                    self._by_channel[ch] = self._window_from(d, ch, True)
                    n += 1
                except Exception:
                    logging.exception("Failed to load DM history %s", fname)
            logging.info("Loaded %d DM windows", n)
        self._dirty = True   # re-save once so legacy plain files become encrypted

    def mark_dirty(self):
        self._dirty = True

    def save_if_dirty(self):
        if self._dirty:
            self.force_save()

    def force_save(self):
        try:
            if self.channel_history_path:
                store.write_json(self.channel_history_path, {
                    str(ch): self._window_to(cv) for ch, cv in self._by_channel.items() if not cv.is_dm})
            if self.dm_history_dir:
                live = set()
                for ch, cv in self._by_channel.items():
                    if cv.is_dm:
                        store.write_json(os.path.join(self.dm_history_dir, f"{ch}.json"), self._window_to(cv))
                        live.add(f"{ch}.json")
                for fname in os.listdir(self.dm_history_dir):   # windows deleted by expiry or !forget
                    if fname.endswith(".json") and fname not in live:
                        os.remove(os.path.join(self.dm_history_dir, fname))
            self._dirty = False
        except Exception:
            logging.exception("Failed to save conversation memory")

    # ---------- retention and deletion ----------
    def expire(self) -> int:
        """Delete content older than the retention period. Returns items removed."""
        cutoff, removed = store.retention_cutoff(), 0
        for ch in list(self._by_channel):
            cv = self._by_channel[ch]
            while cv.turns and cv.turns[0][2] < cutoff:
                cv.turns.popleft(); removed += 1
            while cv.utterances and cv.utterances[0].ts < cutoff:
                cv.utterances.popleft(); removed += 1
            if cv.summary and cv.summary_since < cutoff:
                cv.summary, cv.summary_since = "", 0.0; removed += 1
            if not cv.turns and not cv.utterances and not cv.summary:
                del self._by_channel[ch]
        if removed:
            self.mark_dirty()
            logging.info("Retention: removed %d expired memory item(s)", removed)
        return removed

    def forget_user(self, user_id: int) -> int:
        """Deletion request: drop the user's DM memory and every channel window they
        took part in (turns are not attributable to one user, so the window goes)."""
        gone = 0
        for ch in list(self._by_channel):
            cv = self._by_channel[ch]
            if (cv.is_dm and ch == user_id) or any(u.author_id == user_id for u in cv.utterances):
                del self._by_channel[ch]; gone += 1
        if gone:
            self.mark_dirty(); self.save_if_dirty()
        return gone

    # ---------- API ----------
    def get(self, channel_id: int, is_dm: bool | None = None) -> ConversationWindow:
        cv = self._by_channel.get(channel_id)
        if cv is None:
            cv = self._by_channel[channel_id] = ConversationWindow(channel_id=channel_id, is_dm=bool(is_dm))
            self.mark_dirty()
        elif is_dm is not None and cv.is_dm != is_dm:
            cv.is_dm = is_dm
            self.mark_dirty()
        return cv

    def clear_channel(self, channel_id: int) -> bool:
        if channel_id not in self._by_channel:
            return False
        del self._by_channel[channel_id]
        self.mark_dirty(); self.save_if_dirty()
        return True

    def add_user(self, channel_id: int, is_dm: bool, author_id: int, author_name: str, content: str, message_id: int):
        cv, now = self.get(channel_id, is_dm=is_dm), time.time()
        cv.utterances.append(Utterance(author_id, author_name, content, message_id, now))
        cv.turns.append(("user", content if is_dm else f"[{author_name}]: {content}", now))
        self.mark_dirty()

    def add_assistant(self, channel_id: int, content: str):
        self.get(channel_id).turns.append(("assistant", content, time.time()))
        self.mark_dirty()

    async def maybe_compress(self, channel_id: int):
        """If the turn window is nearly full, summarise the oldest half."""
        cv = self.get(channel_id)
        if len(cv.turns) < self._compress_at or not llm_enabled():
            return
        all_turns = list(cv.turns)
        half = len(all_turns) // 2
        old, keep = all_turns[:half], all_turns[half:]
        text_block = "\n".join(f"{r}: {c}" for r, c, _ in old)
        existing = f"Previous summary:\n{cv.summary}\n\n" if cv.summary else ""
        system = (
            "Compress the following conversation excerpt into a concise summary paragraph. "
            "Record only facts, events, decisions, user names, and open questions the assistant "
            "needs to continue naturally. Do NOT describe or quote the assistant's writing style, "
            "tone, or recurring phrasing — summarize what happened, never how it was worded. "
            "Write in third person. Keep it under 200 words."
        )
        try:
            new_summary = await chat_async(
                [{"role": "system", "content": system},
                 {"role": "user", "content": f"{existing}New turns to incorporate:\n{text_block}"}],
                temperature=0.2, max_tokens=300, model=utility_model())
            oldest = min(t[2] for t in old)
            cv.summary_since = min(cv.summary_since, oldest) if cv.summary else oldest
            cv.summary = (new_summary or "").strip()
            cv.turns = deque(keep, maxlen=self.maxlen_turns)
            self.mark_dirty()
            logging.info("Compressed %d old turns for channel %s", half, channel_id)
        except Exception:
            logging.exception("Turn compression failed for channel %s", channel_id)

    def build_messages(self, channel_id: int, system_prefix: str) -> list[dict[str, str]]:
        c = self.get(channel_id)
        msgs = [{"role": "system", "content": system_prefix}]
        if c.summary:
            msgs.append({"role": "system", "content": f"Conversation summary so far:\n{c.summary}"})
        msgs.extend({"role": r, "content": t} for r, t in c.pairs())
        return msgs


cm = ConversationManager(
    channel_history_path=config.get("ChannelHistoryPath", "channel_history.json"),
    dm_history_dir=config.get("DMHistoryDir", "dm_history"),
)
