"""Per-channel conversation windows with summaries, persisted to disk."""
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

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
    turns: deque[tuple[str, str]] = field(default_factory=lambda: deque(maxlen=40))
    utterances: deque[Utterance] = field(default_factory=lambda: deque(maxlen=100))
    summary: str = ""
    is_dm: bool = False


class ConversationManager:
    def __init__(
        self,
        maxlen_turns: int = 40,
        channel_history_path: str | None = None,
        dm_history_dir: str | None = None,
    ):
        self._by_channel: dict[int, ConversationWindow] = {}
        self.maxlen_turns = maxlen_turns
        self.channel_history_path = channel_history_path
        self.dm_history_dir = dm_history_dir
        self._utterance_maxlen = 100
        self._dirty = False
        self._compress_at = maxlen_turns - 2

        if self.dm_history_dir:
            os.makedirs(self.dm_history_dir, exist_ok=True)

        self._load_from_disk()

    # ---------- Persistence ----------

    def _load_from_disk(self):
        self._load_channels()
        self._load_dms()

    def _load_channels(self):
        path = self.channel_history_path
        if not path or not os.path.exists(path):
            logging.info("No channel history file at %s; starting fresh.", path)
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded = 0
            for ch_key, cv_data in data.items():
                try:
                    ch_id = int(cv_data.get("channel_id", ch_key))
                    turns = deque(
                        [tuple(t) for t in cv_data.get("turns", [])],
                        maxlen=self.maxlen_turns,
                    )
                    uttrs: deque[Utterance] = deque(maxlen=self._utterance_maxlen)
                    for u in cv_data.get("utterances", []):
                        uttrs.append(Utterance(
                            author_id=u.get("author_id"),
                            author_name=u.get("author_name", ""),
                            content=u.get("content", ""),
                            message_id=u.get("message_id", 0),
                            ts=u.get("ts", 0.0),
                        ))
                    self._by_channel[ch_id] = ConversationWindow(
                        channel_id=ch_id,
                        turns=turns,
                        utterances=uttrs,
                        summary=cv_data.get("summary", ""),
                        is_dm=bool(cv_data.get("is_dm", False)),
                    )
                    loaded += 1
                except Exception:
                    logging.exception("Failed to load window for channel %r", ch_key)
            logging.info("Loaded %d channel windows from %s", loaded, path)
        except Exception:
            logging.exception("Failed to load channel history from %s", path)

    def _load_dms(self):
        dir_path = self.dm_history_dir
        if not dir_path or not os.path.isdir(dir_path):
            logging.info("No DM history dir at %s; starting fresh.", dir_path)
            return
        loaded = 0
        for fname in os.listdir(dir_path):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(dir_path, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    cv_data = json.load(f)
                ch_id = int(cv_data.get("channel_id", os.path.splitext(fname)[0]))
                turns = deque(
                    [tuple(t) for t in cv_data.get("turns", [])],
                    maxlen=self.maxlen_turns,
                )
                uttrs: deque[Utterance] = deque(maxlen=self._utterance_maxlen)
                for u in cv_data.get("utterances", []):
                    uttrs.append(Utterance(
                        author_id=u.get("author_id"),
                        author_name=u.get("author_name", ""),
                        content=u.get("content", ""),
                        message_id=u.get("message_id", 0),
                        ts=u.get("ts", 0.0),
                    ))
                self._by_channel[ch_id] = ConversationWindow(
                    channel_id=ch_id,
                    turns=turns,
                    utterances=uttrs,
                    summary=cv_data.get("summary", ""),
                    is_dm=True,
                )
                loaded += 1
            except Exception:
                logging.exception("Failed to load DM history from %s", fpath)
        logging.info("Loaded %d DM windows from %s", loaded, dir_path)

    def mark_dirty(self):
        self._dirty = True

    def save_if_dirty(self):
        if not self._dirty:
            return
        self._save_channels()
        self._save_dms()
        self._dirty = False

    def force_save(self):
        self._save_channels()
        self._save_dms()
        self._dirty = False

    def _save_channels(self):
        path = self.channel_history_path
        if not path:
            return
        try:
            serializable: dict[str, Any] = {}
            for ch_id, cv in self._by_channel.items():
                if cv.is_dm:
                    continue
                serializable[str(ch_id)] = {
                    "channel_id": ch_id,
                    "turns": [list(t) for t in cv.turns],
                    "utterances": [u.__dict__ for u in cv.utterances],
                    "summary": cv.summary,
                    "is_dm": False,
                }
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception:
            logging.exception("Failed to save channel history to %s", path)

    def _save_dms(self):
        dir_path = self.dm_history_dir
        if not dir_path:
            return
        try:
            for ch_id, cv in self._by_channel.items():
                if not cv.is_dm:
                    continue
                data = {
                    "channel_id": ch_id,
                    "turns": [list(t) for t in cv.turns],
                    "utterances": [u.__dict__ for u in cv.utterances],
                    "summary": cv.summary,
                    "is_dm": True,
                }
                fpath = os.path.join(dir_path, f"{ch_id}.json")
                tmp = fpath + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, fpath)
        except Exception:
            logging.exception("Failed to save DM histories to %s", dir_path)

    # ---------- API ----------

    def get(self, channel_id: int, is_dm: bool | None = None) -> ConversationWindow:
        cv = self._by_channel.get(channel_id)
        if cv is None:
            cv = ConversationWindow(channel_id=channel_id, is_dm=bool(is_dm))
            self._by_channel[channel_id] = cv
            self.mark_dirty()
        elif is_dm is not None and cv.is_dm != is_dm:
            cv.is_dm = is_dm
            self.mark_dirty()
        return cv

    def clear_channel(self, channel_id: int) -> bool:
        """Wipe a channel's conversation history. Returns True if anything was cleared."""
        cv = self._by_channel.get(channel_id)
        if cv is None:
            return False
        cv.turns.clear()
        cv.utterances.clear()
        cv.summary = ""
        self.mark_dirty()
        self.save_if_dirty()  # persist immediately so a restart can't restore the bad state
        return True

    def add_user(
        self,
        channel_id: int,
        is_dm: bool,
        author_id: int,
        author_name: str,
        content: str,
        message_id: int,
    ):
        cv = self.get(channel_id, is_dm=is_dm)
        cv.utterances.append(Utterance(author_id, author_name, content, message_id, time.time()))
        if is_dm:
            cv.turns.append(("user", content))
        else:
            cv.turns.append(("user", f"[{author_name}]: {content}"))
        self.mark_dirty()

    def add_assistant(self, channel_id: int, content: str):
        cv = self.get(channel_id)
        cv.turns.append(("assistant", content))
        self.mark_dirty()

    async def maybe_compress(self, channel_id: int):
        """If the turn window is nearly full, summarize the oldest half."""
        cv = self.get(channel_id)
        if len(cv.turns) < self._compress_at or not llm_enabled():
            return

        all_turns = list(cv.turns)
        half = len(all_turns) // 2
        old_turns = all_turns[:half]
        keep_turns = all_turns[half:]

        text_block = "\n".join(f"{role}: {content}" for role, content in old_turns)
        existing = f"Previous summary:\n{cv.summary}\n\n" if cv.summary else ""

        system = (
            "Compress the following conversation excerpt into a concise summary paragraph. "
            "Record only facts, events, decisions, user names, and open questions the assistant "
            "needs to continue naturally. Do NOT describe or quote the assistant's writing style, "
            "tone, or recurring phrasing — summarize what happened, never how it was worded. "
            "Write in third person. Keep it under 200 words."
        )
        user_msg = f"{existing}New turns to incorporate:\n{text_block}"

        try:
            new_summary = await chat_async(
                [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
                temperature=0.2,
                max_tokens=300,
                model=utility_model(),
            )
            cv.summary = (new_summary or "").strip()
            cv.turns = deque(
                [tuple(t) for t in keep_turns],
                maxlen=self.maxlen_turns,
            )
            self.mark_dirty()
            logging.info("Compressed %d old turns for channel %s", half, channel_id)
        except Exception:
            logging.exception("Turn compression failed for channel %s", channel_id)

    def build_messages(self, channel_id: int, system_prefix: str) -> list[dict[str, str]]:
        c = self.get(channel_id)
        msgs: list[dict[str, str]] = [{"role": "system", "content": system_prefix}]
        if c.summary:
            msgs.append({"role": "system", "content": f"Conversation summary so far:\n{c.summary}"})
        msgs.extend({"role": r, "content": t} for r, t in c.turns)
        return msgs


cm = ConversationManager(
    channel_history_path=config.get("ChannelHistoryPath", "channel_history.json"),
    dm_history_dir=config.get("DMHistoryDir", "dm_history"),
)
