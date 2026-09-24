"""Event wiring, /draw, owner commands and process lifecycle."""
import asyncio
import functools
import logging
import os
import re
import signal
import sys
import time
from typing import Any

import discord
from discord import app_commands

from . import core, lore
from .core import config, bot, tree, safe_send, channel_key, get_user_bucket, images_enabled, image_channel_allowed
from .memory import cm
from .images import recent_conversation, ipm, ImageActionsView, run_image_job, compile_sd_prompt
from .chat import (handle_text_message, handle_image_message, looks_like_image_request,
                   should_route_to_image_followup, _looks_like_tag_prompt)
from .safety import strip_pasted_negative, image_prompt_blocked, refuse_image_request, user_on_cooldown, expire_refusals, forget_refusals
from .llm import llm_enabled

_started = False


@bot.event
async def on_ready():
    global _started
    logging.info("READY as %s (id=%s) pid=%s", bot.user, getattr(bot.user, "id", "?"), os.getpid())
    if not _started:
        _started = True
        _signals()
        bot.add_view(ImageActionsView())  # persistent buttons survive restarts
        for coro in (_periodic_save(), _config_watch(), _sync_commands(), _retention_loop()):
            bot.loop.create_task(coro)
        logging.info("Gating: age-restricted channels only=%s, DMs=%s, allowed channels=%s",
                     config.get("RequireAgeRestrictedChannel", True), config.get("AllowInDMs", False),
                     sorted(core.cfg_ids("AllowedChannels")))


@bot.event
async def on_message(message: discord.Message):
    if bot.user is None or message.author.id == bot.user.id or message.author.bot:
        return
    try:
        if core.is_ignored(message):
            return
        if await _owner_command(message):
            return
        if not core.is_allowed(message):
            if isinstance(message.channel, discord.DMChannel):
                await _dm_redirect(message)
            return
        if user_on_cooldown(message.author.id):
            return
        raw = message.content or ""
        text = raw
        if config.get("OnlyWhenCalled") and not isinstance(message.channel, discord.DMChannel):
            name = config.get("Name", "")
            if not ((name.lower() in raw.lower()) or (bot.user in message.mentions)):
                return
            text = re.sub(re.escape(name), "", raw, flags=re.IGNORECASE).strip()
        if not llm_enabled():
            if isinstance(message.channel, discord.DMChannel) or bot.user in message.mentions:
                await safe_send(message.channel, config.get("LLMDisabledNotice", "Chat is currently disabled."))
            return
        logging.info("ROUTER in=%r ch=%s author=%s", raw[:120], channel_key(message), message.author)
        if should_route_to_image_followup(message) or looks_like_image_request(text):
            asyncio.create_task(handle_image_message(message, text_override=text))
        else:
            asyncio.create_task(handle_text_message(message, text_override=text))
    except Exception:
        logging.exception("on_message router failure")


# ---- DMs ---------------------------------------------------------------------
_dm_redirected: dict[int, float] = {}


async def _dm_redirect(message: discord.Message) -> None:
    """DMs cannot be age-restricted, so she does not play there. Say so once a
    day per person instead of going silent, and point to her channel."""
    now = time.time()
    if now - _dm_redirected.get(message.author.id, 0) < 86400:
        return
    _dm_redirected[message.author.id] = now
    channel = next(iter(sorted(core.cfg_ids("AllowedChannels"))), 0)
    template = config.get("DMRedirectMessage") or (
        "Not here, darling. Private messages can't be age-restricted, so I only play in {channel}. Come find me there.")
    try:
        await safe_send(message.channel, template.format(channel=f"<#{channel}>" if channel else "my channel"))
    except Exception:
        logging.exception("DM redirect failed")


# ---- Owner commands (DM only) -----------------------------------------------
async def _owner_command(message: discord.Message) -> bool:
    if not isinstance(message.channel, discord.DMChannel) or message.author.id != core.cfg_int("OwnerID"):
        return False
    text = (message.content or "").strip()
    if not text.startswith("!"):
        return False
    parts = text.split(None, 1)
    cmd, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
    if cmd == "!reload":
        try:
            core.load_config()
            lore.load_lore()
            await message.channel.send("✅ Config and lore reloaded.")
        except Exception as e:
            await message.channel.send(f"❌ Reload failed: {e}")
        return True
    if cmd == "!clearhistory":
        try:
            target = int(arg.strip().lstrip("#<").rstrip(">"))
        except ValueError:
            await message.channel.send("Usage: `!clearhistory <channel_id>`")
            return True
        ok = cm.clear_channel(target)
        await message.channel.send("✅ Cleared." if ok else "No stored history for that channel.")
        return True
    if cmd == "!forget":
        try:
            uid = int(arg.strip().lstrip("<@!").rstrip(">"))
        except ValueError:
            await message.channel.send("Usage: `!forget <user_id>` — deletes everything held about that user.")
            return True
        w, i, r = cm.forget_user(uid), ipm.forget_user(uid), forget_refusals(uid)
        logging.info("Data deletion for %s: %d windows, %d image records, %d refusal entries", uid, w, i, r)
        await message.channel.send(f"🧹 Deleted {w} conversation window(s), {i} image record(s), {r} refusal entr(ies) for `{uid}`.")
        return True
    if cmd == "!flags":
        recent = list(core.flag_history)[-20:]
        if not recent:
            await message.channel.send("No refusals recorded since start.")
            return True
        body = "\n".join(f"**{t}** — {d.replace(chr(10), ' | ')[:180]}" for _, t, d in reversed(recent))
        await message.channel.send(body[:1900], allowed_mentions=discord.AllowedMentions.none())
        return True
    if cmd == "!help":
        await message.channel.send("Owner commands: `!reload`, `!clearhistory <channel_id>`, `!forget <user_id>`, `!flags`")
        return True
    return False


# ---- /draw ------------------------------------------------------------------
@tree.command(name="draw", description="Ask for an image")
@app_commands.describe(prompt="What to draw", style="Style preset (optional)", aspect="Image shape",
                       count="How many images (1-4)", exact="Send your prompt to Stable Diffusion unchanged")
@app_commands.choices(aspect=[app_commands.Choice(name="square", value="square"),
                              app_commands.Choice(name="portrait", value="portrait"),
                              app_commands.Choice(name="landscape", value="landscape")])
async def draw_command(interaction: discord.Interaction, prompt: str, style: str | None = None,
                       aspect: app_commands.Choice[str] | None = None,
                       count: app_commands.Range[int, 1, 4] = 1, exact: bool = False):
    try:
        channel = interaction.channel
        if not images_enabled():
            await interaction.response.send_message(config.get("ImageDisabledNotice", "Image generation is currently disabled."), ephemeral=True)
            return
        if not image_channel_allowed(channel):
            await interaction.response.send_message(config.get(
                "ChannelNotAllowedNotice", "I only paint in the age-restricted channels set aside for it."), ephemeral=True)
            return
        if user_on_cooldown(interaction.user.id):
            await interaction.response.send_message("Not right now.", ephemeral=True)
            return
        prompt = strip_pasted_negative(prompt)
        blocked = image_prompt_blocked(prompt)
        if blocked:
            await interaction.response.send_message(config.get(
                "ImageRefusalMessage", "No. That is not something I will ever draw, and the moderators have been notified."), ephemeral=True)
            await refuse_image_request(channel, interaction.user.id, interaction.user.display_name, blocked, prompt)
            return
        if not get_user_bucket(interaction.user.id).consume():
            await interaction.response.send_message("You're requesting images too fast — slow down a bit.", ephemeral=True)
            return
        width = height = None
        if aspect and aspect.value in ("portrait", "landscape"):
            size = config.get("SDPortraitSize" if aspect.value == "portrait" else "SDLandscapeSize") or []
            if len(size) == 2:
                width, height = int(size[0]), int(size[1])
        positive_prefix, neg = None, config.get("SDNegativePrompt", "(lowres, blurry, deformed)")
        if style:
            preset = next((v for k, v in (config.get("SDStylePresets") or {}).items() if k.lower() == style.lower()), None)
            if preset:
                positive_prefix, neg = preset.get("positive", ""), preset.get("negative") or neg
        await interaction.response.send_message("Hang on while I sketch that for you…")
        status_msg = await interaction.original_response()
        raw = prompt.strip()[:1500]
        parent = getattr(channel, "parent", None)
        ch_key = parent.id if parent is not None else channel.id
        sd_prompt = raw if (exact or _looks_like_tag_prompt(raw)) else await compile_sd_prompt(raw, recent_conversation(ch_key))
        asyncio.create_task(run_image_job(
            channel, ch_id=parent.id if parent is not None else channel.id, user_prompt=raw, sd_prompt=sd_prompt,
            neg=neg, batch=int(count), width=width, height=height, positive_prefix=positive_prefix,
            requested_by=interaction.user.display_name, requester_id=interaction.user.id, status_msg=status_msg))
    except Exception:
        logging.exception("/draw failed")


@draw_command.autocomplete("style")
async def _draw_style_autocomplete(interaction: discord.Interaction, current: str):
    presets = config.get("SDStylePresets") or {}
    return [app_commands.Choice(name=k, value=k) for k in presets if current.lower() in k.lower()][:25]


# ---- Background -------------------------------------------------------------
async def _sync_commands():
    await bot.wait_until_ready()
    try:
        gid = core.cfg_int("AppCommandGuildID")
        if gid:
            guild = discord.Object(id=gid)
            tree.copy_global_to(guild=guild)
            synced = await tree.sync(guild=guild)
        else:
            synced = await tree.sync()
        logging.info("Synced %d app command(s): %s", len(synced), [c.name for c in synced])
    except Exception:
        logging.exception("App command sync failed")


async def _retention_loop():
    """Hourly: delete stored content older than RetentionDays (default 30)."""
    await bot.wait_until_ready()
    while not bot.is_closed():
        try:
            cm.expire(); ipm.expire(); expire_refusals()
            cm.save_if_dirty(); ipm.save_if_dirty()
        except Exception:
            logging.exception("Retention sweep failed")
        await asyncio.sleep(3600)


async def _periodic_save():
    while True:
        await asyncio.sleep(30)
        try:
            cm.save_if_dirty()
            ipm.save_if_dirty()
        except Exception:
            logging.exception("Periodic save failed")


async def _config_watch():
    while True:
        await asyncio.sleep(10)
        try:
            if os.path.getmtime(core.CONFIG_PATH) > core.config_mtime:
                core.load_config()
                lore.load_lore()
                logging.info("Config hot-reloaded")
            elif lore.lore_changed():
                lore.load_lore()
                logging.info("Lore hot-reloaded")
        except FileNotFoundError:
            pass
        except Exception:
            logging.exception("Config watch error")


def _signals():
    # Called from on_ready, inside the running loop. Python 3.14 no longer
    # creates a loop on demand, so asking for one before bot.run() fails.
    loop = asyncio.get_running_loop()

    def _shutdown(sig):
        logging.info("Received %s — flushing state", sig.name)
        cm.force_save()
        ipm.force_save()
        loop.create_task(bot.close())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, functools.partial(_shutdown, sig))
        except NotImplementedError:
            pass


def main() -> None:
    lore.load_lore()
    token = config.get("DiscordToken") or ""
    if not token:
        logging.error("No DiscordToken in %s.", core.CONFIG_PATH)
        sys.exit(78)
    try:
        bot.run(token)
    except discord.LoginFailure:
        logging.error("Discord rejected the token. Not retrying.")
        sys.exit(78)
    except discord.PrivilegedIntentsRequired:
        logging.error("Message Content intent not approved for this application. Not retrying.")
        sys.exit(78)
