"""Encrypted-at-rest storage for everything Isabell writes that contains message
content: conversation memory, image prompts and the refusal log.

Files are Fernet-encrypted with a key in isabell.key (created on first run, 0600,
never committed). Plain files written by older versions are read transparently
and encrypted on the next save.
"""
import json
import logging
import os
import time
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from .core import config

_MAGIC = b"ISAENC1:"
_fernet: Fernet | None = None


def _key() -> Fernet:
    global _fernet
    if _fernet is None:
        path = config.get("StorageKeyPath", "isabell.key")
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(Fernet.generate_key())
            os.chmod(path, 0o600)
            logging.info("Generated storage key at %s — back it up with the data", path)
        with open(path, "rb") as f:
            _fernet = Fernet(f.read().strip())
    return _fernet


def _enc(raw: bytes) -> bytes:
    return _MAGIC + _key().encrypt(raw)


def _dec(blob: bytes) -> bytes:
    if blob.startswith(_MAGIC):
        return _key().decrypt(blob[len(_MAGIC):])
    return blob  # legacy plain file


def write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(_enc(json.dumps(obj, ensure_ascii=False).encode("utf-8")))
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def read_json(path: str) -> Any:
    with open(path, "rb") as f:
        blob = f.read()
    try:
        return json.loads(_dec(blob).decode("utf-8"))
    except InvalidToken:
        raise ValueError(f"{path}: encrypted with a different key")


def append_line(path: str, obj: dict) -> None:
    """One encrypted record per line (the refusal log)."""
    line = _enc(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "ab") as f:
        f.write(line + b"\n")


def read_lines(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(_dec(raw).decode("utf-8")))
            except Exception:
                logging.warning("Skipping unreadable line in %s", path)
    return out


def rewrite_lines(path: str, rows: list[dict]) -> None:
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        for r in rows:
            f.write(_enc(json.dumps(r, ensure_ascii=False).encode("utf-8")) + b"\n")
    os.replace(tmp, path)


def retention_cutoff() -> float:
    """Anything older than this is deleted. RetentionDays defaults to 30."""
    return time.time() - float(config.get("RetentionDays", 30)) * 86400
