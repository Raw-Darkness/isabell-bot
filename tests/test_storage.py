import os
import time

from isabell import store, safety
from isabell.memory import ConversationManager
from isabell.images import ImagePromptMemory, ImagePromptRecord


def test_files_are_encrypted_on_disk():
    store.write_json("x.json", {"secret": "roleplay text"})
    assert b"roleplay" not in open("x.json", "rb").read()
    assert store.read_json("x.json") == {"secret": "roleplay text"}
    assert oct(os.stat("x.json").st_mode)[-3:] == "600"


def test_legacy_plain_files_still_load():
    open("plain.json", "w").write('{"a": 1}')
    assert store.read_json("plain.json") == {"a": 1}


def test_memory_expires_old_turns_and_summaries():
    cm = ConversationManager(channel_history_path="ch.json", dm_history_dir="dms")
    cm.add_user(1, False, 7, "u", "old message", 1)
    cm.add_user(1, False, 7, "u", "new message", 2)
    cv = cm.get(1)
    r, c, _ = cv.turns[0]
    cv.turns[0] = (r, c, time.time() - 40 * 86400)
    cv.utterances[0].ts = time.time() - 40 * 86400
    cv.summary, cv.summary_since = "an old summary", time.time() - 31 * 86400
    cm.expire()
    assert [c for _, c in cv.pairs()] == ["[u]: new message"]
    assert cv.summary == ""
    cm.force_save()
    assert b"new message" not in open("ch.json", "rb").read()
    reloaded = ConversationManager(channel_history_path="ch.json", dm_history_dir="dms")
    assert [c for _, c in reloaded.get(1).pairs()] == ["[u]: new message"]


def test_forget_user_removes_dm_and_shared_windows():
    cm = ConversationManager(channel_history_path="f.json", dm_history_dir="fdms")
    cm.add_user(99, True, 99, "u", "dm text", 1)
    cm.add_user(5, False, 99, "u", "channel text", 2)
    cm.add_user(6, False, 42, "other", "unrelated", 3)
    cm.force_save()
    assert os.path.exists("fdms/99.json")
    assert cm.forget_user(99) == 2
    assert not os.path.exists("fdms/99.json")
    assert 6 in cm._by_channel and 5 not in cm._by_channel


def test_image_memory_expiry_and_forget():
    ipm = ImagePromptMemory(path="img.json")
    ipm.add(ImagePromptRecord(1, 1, "old", "old", ts=time.time() - 40 * 86400, meta={"by_id": 3}))
    ipm.add(ImagePromptRecord(1, 2, "new", "new", ts=time.time(), meta={"by_id": 3}))
    assert ipm.expire() == 1
    assert ipm.forget_user(3) == 1
    assert ipm.last_for_channel(1) is None


def test_refusal_log_encrypted_expired_and_forgotten(monkeypatch):
    from isabell import core
    monkeypatch.setitem(core.config, "RefusalLogPath", "refusals-storage-test.jsonl")
    safety._log_refusal("chat_input", 11, "a", "loli", "some text")
    safety._log_refusal("chat_input", 12, "b", "loli", "other text")
    path = "refusals-storage-test.jsonl"
    assert b"some text" not in open(path, "rb").read()
    rows = store.read_lines(path)
    rows[0]["ts"] = time.time() - 40 * 86400
    store.rewrite_lines(path, rows)
    assert safety.expire_refusals() == 1
    assert safety.forget_refusals(12) == 1
    assert store.read_lines(path) == []
