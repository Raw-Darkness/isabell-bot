"""Shared state: logging, hot-reloadable config, the Discord client, small helpers,
channel gating and the mod-channel alert."""
import difflib
import json
import logging
import os
import time
from collections import deque
from logging.handlers import TimedRotatingFileHandler
from typing import Any

import aiohttp
import discord
from discord import app_commands

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_fh = TimedRotatingFileHandler("app.log", when="midnight", interval=1, backupCount=7)
_fh.setLevel(logging.INFO)
_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
if not any(isinstance(h, TimedRotatingFileHandler) for h in logging.getLogger().handlers):
    logging.getLogger().addHandler(_fh)

# ---- Config (updated in place, so `from .core import config` never goes stale) ----
_SECRET_KEYS = {"DiscordToken", "OpenAPIKey", "Personality"}


def _default_config_path() -> str:
    for cand in ("Config.json", "Isabell.json"):
        if os.path.exists(cand):
            return cand
    return "Config.json"


CONFIG_PATH = os.environ.get("BOT_CONFIG") or _default_config_path()
config: dict[str, Any] = {}
config_mtime: float = 0.0


def load_config() -> None:
    global config_mtime
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    config.clear()
    config.update(data)
    config_mtime = os.path.getmtime(CONFIG_PATH)
    logging.info("Loaded config %s: %s", CONFIG_PATH, {k: v for k, v in config.items() if k not in _SECRET_KEYS})


load_config()


def cfg_int(key: str, default: int = 0) -> int:
    try:
        return int(config.get(key, default) or 0)
    except (TypeError, ValueError):
        return default


def cfg_ids(key: str) -> set[int]:
    out: set[int] = set()
    for v in config.get(key) or []:
        try:
            out.add(int(v))
        except (TypeError, ValueError):
            pass
    return out


# ---- Discord client ---------------------------------------------------------
intents = discord.Intents.default()
intents.guilds = True
intents.message_content = bool(config.get("EnableMessageContentIntent", True))
bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)


# ---- Helpers ----------------------------------------------------------------
def clamp_2000(text: str) -> str:
    return (text or "")[:2000]


async def safe_send(channel: discord.abc.Messageable, text: str | None = None, **kwargs):
    try:
        if text is not None:
            return await channel.send(clamp_2000(text), **kwargs)
        return await channel.send(**kwargs)
    except Exception:
        logging.exception("safe_send failed")


async def get_channel(channel_id: int):
    if not channel_id:
        return None
    try:
        return bot.get_channel(channel_id) or await bot.fetch_channel(channel_id)
    except Exception:
        logging.exception("Cannot resolve channel %s", channel_id)
        return None


def channel_key(message: discord.Message) -> int:
    """Threads share memory with their parent; DMs use the author ID."""
    if isinstance(message.channel, discord.DMChannel):
        return message.author.id
    parent = getattr(message.channel, "parent", None)
    return parent.id if parent is not None else message.channel.id


def _is_age_restricted(channel) -> bool:
    if channel is None:
        return False
    ch = getattr(channel, "parent", None) or channel
    fn = getattr(ch, "is_nsfw", None)
    return bool(fn()) if callable(fn) else False


def channel_allowed(channel, *, purpose: str = "chat") -> bool:
    """Where the bot may operate at all.

    - DMs only when AllowInDMs is true (default false: a DM is never age-restricted).
    - Otherwise the channel (or its parent) must be in AllowedChannels, and — when
      RequireAgeRestrictedChannel is true (default) — flagged age-restricted in Discord.
    """
    if channel is None:
        return False
    if isinstance(channel, discord.DMChannel):
        return bool(config.get("AllowInDMs", False))
    allowed = cfg_ids("AllowedChannels")
    parent = getattr(channel, "parent", None)
    in_list = channel.id in allowed or (parent is not None and parent.id in allowed)
    if not in_list:
        return False
    if config.get("RequireAgeRestrictedChannel", True) and not _is_age_restricted(channel):
        return False
    return True


def image_channel_allowed(channel) -> bool:
    return channel_allowed(channel, purpose="image")


def is_allowed(message: discord.Message) -> bool:
    return channel_allowed(message.channel)


def is_ignored(message: discord.Message) -> bool:
    if message.author.id in cfg_ids("IgnoredUsers"):
        return True
    low = (message.content or "").lower()
    return any(str(w).lower() in low for w in config.get("IgnoredWords") or [])


async def ensure_can_send(message: discord.Message) -> bool:
    if isinstance(message.channel, discord.DMChannel):
        return True
    try:
        me = message.guild.me or await message.guild.fetch_member(bot.user.id)
        if not message.channel.permissions_for(me).send_messages:
            logging.warning("No send_messages perm in #%s (%s)", message.channel, message.channel.id)
            return False
        return True
    except Exception:
        logging.exception("Permission check failed; assuming False")
        return False


def too_similar(a: str, b: str, threshold: float = 0.9) -> bool:
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity, self.tokens, self.last, self.refill_rate = capacity, float(capacity), time.time(), refill_rate

    def consume(self, n: int = 1) -> bool:
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.refill_rate)
        self.last = now
        if n <= self.tokens:
            self.tokens -= n
            return True
        return False


_user_buckets: dict[int, TokenBucket] = {}


def get_user_bucket(user_id: int) -> TokenBucket:
    """5 images per minute per user."""
    b = _user_buckets.get(user_id)
    if b is None:
        b = _user_buckets[user_id] = TokenBucket(capacity=5, refill_rate=5.0 / 60.0)
    return b


def images_enabled() -> bool:
    return bool(config.get("ImageGenerationEnabled", True))


async def image_unavailable(channel) -> None:
    await safe_send(channel, config.get("ImageDisabledNotice", "Image generation is currently disabled."))


# ---- Mod alerts -------------------------------------------------------------
flag_history: deque[tuple[float, str, str]] = deque(maxlen=200)


async def flag_to_mods(title: str, details: str, ping: bool = True) -> None:
    """Post to the mod channel. ping=False posts silently for contextual matches
    that may be false positives."""
    flag_history.append((time.time(), title, details))
    prefix = "@here " if ping and config.get("ModAlertPing", True) else ""
    text = clamp_2000(f"{prefix}⚠️ **{title}**\n{details}")
    # Preferred: a webhook in the mod channel, so the bot needs no access to it.
    hook = str(config.get("ModAlertWebhook") or "")
    if hook:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
                async with s.post(hook, json={
                    "content": text, "username": f"{config.get('Name', 'Bot')} safety",
                    "allowed_mentions": {"parse": ["everyone"] if prefix else []},
                }) as r:
                    if r.status < 300:
                        return
                    logging.error("Mod alert webhook returned HTTP %s", r.status)
        except Exception:
            logging.exception("Mod alert webhook failed; trying the channel")
    ch = await get_channel(cfg_int("ModChannelID"))
    if ch is None:
        logging.error("Mod alert could not be delivered: %s", title)
        return
    await safe_send(ch, text, allowed_mentions=discord.AllowedMentions(everyone=True, users=False, roles=False))
