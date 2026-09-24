"""Roleplay chat and the natural-language image path."""
import asyncio
import json
import logging
import re
import time

import discord

from . import lore
from .core import config, safe_send, channel_key, ensure_can_send, get_user_bucket, too_similar as _too_similar, images_enabled, image_unavailable
from .llm import chat_async
from .memory import cm
from .images import (recent_conversation, ipm, _last_image_b64, run_image_job, compile_sd_prompt, refine_image_prompt,
                     image_tools_for, run_tool_image)
from .safety import strip_pasted_negative, chat_message_blocked, refuse_chat, image_prompt_blocked, refuse_image_request, classify_chat


_IMAGE_TRIGGER_RE = re.compile(
    r"\b(draw|paint|sketch|illustrate|render|generate an image|make a picture)\b",
    re.IGNORECASE,
)
EXACT_TRIGGERS = ("draw exact", "image exact", "img exact", "exact:")

FOLLOWUP_STARTS = (
    "same", "again", "keep", "also", "but now", "make it",
    "change", "adjust", "brighter", "darker", "add",
)


def is_exact_trigger(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in EXACT_TRIGGERS)


# Users often send bare SD prompt fragments as follow-ups — "(Translucent skin:1.6)",
# "Labia spreading:1.3" — with no trigger word at all. Measured against real traffic,
# these were the bulk of the requests keyword matching missed.
_SD_WEIGHT_RE = re.compile(r"\([^()\n]{2,60}:\s*\d(?:\.\d+)?\)|\b[a-z][a-z ]{2,40}:\s*\d\.\d\b", re.IGNORECASE)


def looks_like_sd_syntax(text: str) -> bool:
    t = (text or "").strip()
    if not t or "http://" in t or "https://" in t:
        return False
    return bool(_SD_WEIGHT_RE.search(t))


def looks_like_image_request(text: str) -> bool:
    t = (text or "").strip()
    if bool(_IMAGE_TRIGGER_RE.search(t)) or t.lower().startswith(("img:", "image:", "art:")):
        return True
    return looks_like_sd_syntax(t)


def _looks_like_tag_prompt(text: str) -> bool:
    """Comma-heavy tag lists are already SD-ready — skip the LLM rewrite."""
    if not config.get("SkipRewriteForTagPrompts", True):
        return False
    return (text or "").count(",") >= 5


def looks_like_followup(text: str) -> bool:
    t = (text or "").lower().strip()
    return any(t.startswith(s) for s in FOLLOWUP_STARTS)


def should_route_to_image_followup(message: discord.Message) -> bool:
    # Replying to ANY of the bot's images routes to refinement of that image.
    if message.reference and ipm.find_by_message(message.reference.message_id):
        return True
    last = ipm.last_for_channel(channel_key(message))
    if not last:
        return False
    if not looks_like_followup(message.content or ""):
        return False
    window = int(config.get("ImageFollowupWindowSec", 600))
    if last.ts <= 0 or (time.time() - last.ts > window):
        return False
    return True


def build_system_prefix(query: str = "") -> str:
    name = config.get("Name", "Assistant")
    persona = (config.get("Personality") or "").strip()

    parts = [f"You are {name}."]
    if persona:
        parts.append(f"\nStay in character as {name}:\n{persona}")
    if lore.LORE_CHAT_CONTEXT:
        parts.append(f"\n\nWorld knowledge (use this to answer questions about the world):\n{lore.LORE_CHAT_CONTEXT}")
    detail = lore.retrieve_lore(query)
    if detail:
        parts.append(f"\n\nRelevant lore detail for this message:\n{detail}")
    # Last, so it outweighs style cues in the persona and in her own earlier replies.
    style = (config.get("ReplyStyle") or "").strip()
    if style:
        parts.append(f"\n\n{style}")
    return "\n".join(parts)


async def handle_text_message(message: discord.Message, text_override: str | None = None):
    try:
        if not await ensure_can_send(message):
            return
        ch_id = channel_key(message)
        is_dm = isinstance(message.channel, discord.DMChannel)

        incoming = text_override if text_override is not None else (message.content or "")
        recent_ctx = " ".join(t for _, t in cm.get(ch_id).pairs()[-6:])[-1500:]
        blocked = chat_message_blocked(incoming, recent_ctx) or await classify_chat(incoming, recent_ctx)
        if blocked:
            # Never reaches the model and never enters conversation memory, so it
            # cannot steer later replies.
            await refuse_chat(message.channel, message.author.id,
                              message.author.display_name, blocked, incoming, "input")
            return

        cm.add_user(ch_id, is_dm, message.author.id, message.author.display_name, message.content, message.id)
        await cm.maybe_compress(ch_id)

        # Build the retrieval query from the recent exchange, not just this line —
        # otherwise "tell me more about him" retrieves nothing.
        this_text = text_override if text_override is not None else (message.content or "")
        recent = [t for r, t in cm.get(ch_id).pairs()[-4:] if r == "user"]
        retrieval_query = " ".join(recent[-2:] + [this_text])[-1200:]
        system_prefix = build_system_prefix(retrieval_query)
        msgs = cm.build_messages(ch_id, system_prefix=system_prefix)

        if text_override is not None and msgs and msgs[-1]["role"] == "user":
            if not is_dm:
                msgs[-1] = {"role": "user", "content": f"[{message.author.display_name}]: {text_override}"}
            else:
                msgs[-1] = {"role": "user", "content": text_override}

        logging.info("TEXT -> LLM | ch=%s | msg_id=%s", ch_id, message.id)
        freq_pen = float(config.get("FrequencyPenalty", 0.3))
        pres_pen = float(config.get("PresencePenalty", 0.3))
        tools = image_tools_for(message)
        async with message.channel.typing():
            result = await chat_async(
                msgs, temperature=0.6, max_tokens=600,
                frequency_penalty=freq_pen, presence_penalty=pres_pen,
                **({"tools": tools, "return_message": True} if tools else {}),
            )

        reply = result
        if tools:
            calls = getattr(result, "tool_calls", None) or []
            for call in calls:
                if getattr(call.function, "name", "") != "generate_image":
                    continue
                try:
                    args = json.loads(call.function.arguments or "{}")
                except Exception:
                    logging.exception("Image tool: bad arguments %r", call.function.arguments)
                    break
                said = (getattr(result, "content", "") or "").strip()
                if said:
                    cm.add_assistant(ch_id, said)
                    await safe_send(message.channel, said)
                if await run_tool_image(message, args):
                    return
            reply = getattr(result, "content", None)

        if not (reply or "").strip():
            # Model returned nothing (e.g. provider refusal). Don't store or
            # send an empty turn — it would 400 on Discord and pollute memory.
            logging.warning("Empty LLM reply in ch %s; sending fallback", ch_id)
            await safe_send(message.channel, config.get("EmptyReplyFallback", "…I have nothing to say to that."))
            return

        # Loop breaker: a reply that near-duplicates a recent one gets one
        # retry with an explicit nudge. A still-duplicated reply is sent but
        # NOT stored, so the repetition cannot reinforce itself in memory.
        recent = [t for r, t in cm.get(ch_id).pairs()[-8:] if r == "assistant"]
        if any(_too_similar(reply, prev) for prev in recent):
            logging.warning("Repetition detected in ch %s; retrying with nudge", ch_id)
            retry_msgs = msgs + [
                {"role": "assistant", "content": reply},
                {
                    "role": "system",
                    "content": (
                        "Your last reply repeats an earlier one almost verbatim. Write a completely "
                        "different reply: new sentence structure, new imagery, no reused phrases."
                    ),
                },
            ]
            fresh = await chat_async(
                retry_msgs, temperature=0.9, max_tokens=600,
                frequency_penalty=max(freq_pen, 0.5), presence_penalty=max(pres_pen, 0.5),
            )
            if (fresh or "").strip() and not any(_too_similar(fresh, prev) for prev in recent):
                reply = fresh
            else:
                logging.warning("Repetition persists in ch %s; reply withheld from memory", ch_id)
                await safe_send(message.channel, (fresh or "").strip() or reply)
                return

        # Include the user's current turn: they may have set the scene in the very
        # message that prompted this reply.
        out_blocked = chat_message_blocked(reply, f"{recent_ctx} {incoming}") or await classify_chat(reply, f"{recent_ctx} {incoming}")
        if out_blocked:
            # Drop the exchange entirely: storing it would let the reply seed
            # later turns through the conversation window and summaries.
            cv = cm.get(ch_id)
            if cv.turns:
                cv.turns.pop()
            cm.mark_dirty()
            await refuse_chat(message.channel, message.author.id,
                              message.author.display_name, out_blocked, reply, "model output")
            return

        cm.add_assistant(ch_id, reply)
        await safe_send(message.channel, reply)
    except asyncio.CancelledError:
        return
    except Exception:
        logging.exception("Error in handle_text_message")
        await safe_send(message.channel, "Oops — something went wrong with that one.")


async def handle_image_message(message: discord.Message, text_override: str | None = None):
    try:
        if not await ensure_can_send(message):
            return
        ch_id = channel_key(message)

        if not get_user_bucket(message.author.id).consume():
            await safe_send(message.channel, "You're requesting images too fast — slow down a bit.")
            return

        text_in = strip_pasted_negative(text_override if text_override is not None else (message.content or ""))
        if not images_enabled():
            await image_unavailable(message.channel)
            return
        blocked = image_prompt_blocked(text_in)
        if blocked:
            await refuse_image_request(message.channel, message.author.id,
                                       message.author.display_name, blocked, text_in)
            return
        exact_mode = is_exact_trigger(text_in)

        raw = text_in
        for phrase in EXACT_TRIGGERS:
            raw = re.sub(re.escape(phrase), "", raw, count=1, flags=re.IGNORECASE)
        # Strip only a LEADING trigger so words inside the prompt survive
        # ("art nouveau", "a dragon drawing a sword").
        raw = re.sub(r"^\s*(img|image|art)\s*:\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"^\s*(please\s+)?(draw|paint|sketch|illustrate|render)\b\s*(me\s+)?", "", raw, flags=re.IGNORECASE)
        if exact_mode:
            raw = re.sub(r"^\s*exact\b:?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s+", " ", raw).strip(" -:;,. \n\t")

        # --- lightweight parameters parsed from the request ---
        width = height = None
        if re.search(r"\b(portrait|tall)\b", raw, re.IGNORECASE):
            size = config.get("SDPortraitSize") or []
            if len(size) == 2:
                width, height = int(size[0]), int(size[1])
        elif re.search(r"\b(landscape|wide)\b", raw, re.IGNORECASE):
            size = config.get("SDLandscapeSize") or []
            if len(size) == 2:
                width, height = int(size[0]), int(size[1])

        batch = 1
        m = re.search(r"\b([2-9])x\b|\bx([2-9])\b", raw)
        if m:
            batch = int(m.group(1) or m.group(2))
            raw = (raw[:m.start()] + raw[m.end():]).strip()

        positive_prefix = None
        preset_neg = None
        presets = config.get("SDStylePresets") or {}
        m = re.search(r"\bstyle:\s*(\w+)\b", raw, re.IGNORECASE)
        if m:
            preset = next((v for k, v in presets.items() if k.lower() == m.group(1).lower()), None)
            if preset:
                positive_prefix = preset.get("positive", "")
                preset_neg = preset.get("negative")
                raw = (raw[:m.start()] + raw[m.end():]).strip()

        neg = preset_neg or config.get("SDNegativePrompt", "(lowres, blurry, deformed)")

        ref_rec = ipm.find_by_message(message.reference.message_id if message.reference else None)
        base = ref_rec or ipm.last_for_channel(ch_id)
        t_norm = re.sub(r"[\s!.…]+$", "", (text_in or "").strip().lower())
        is_reroll = t_norm in {"again", "same", "same again", "again please", "reroll", "another", "another one", "one more"}

        seed = -1
        init_b64 = None
        if base and is_reroll:
            # Bare "again": same prompt, fresh random seed — a new take.
            status_msg = await safe_send(message.channel, "Rolling a fresh take on that…")
            sd_prompt, neg = base.final_sd_prompt, base.negative_prompt
            positive_prefix = base.positive_prefix or None
            width, height = base.width or None, base.height or None
        elif base and (looks_like_followup(text_in) or ref_rec):
            # A change request: refine the prompt, keep the seed, and — when we
            # still hold the source image — run img2img so the composition stays put.
            status_msg = await safe_send(message.channel, config.get("ImageRefinementNotice", "Refining the previous image…"))
            refined = await refine_image_prompt(base, text_in)
            sd_prompt = refined.get("prompt", base.final_sd_prompt)
            neg = refined.get("negative", base.negative_prompt)
            seed = base.seed
            positive_prefix = base.positive_prefix or None
            width, height = base.width or None, base.height or None
            if base is ipm.last_for_channel(ch_id):
                init_b64 = _last_image_b64.get(ch_id)
        else:
            status_msg = await safe_send(message.channel, "Hang on while I sketch that for you…")
            sd_prompt = raw if (exact_mode or _looks_like_tag_prompt(raw)) else await compile_sd_prompt(raw, recent_conversation(ch_id))

        await run_image_job(
            message.channel,
            ch_id=ch_id,
            user_prompt=text_in,
            sd_prompt=sd_prompt,
            neg=neg,
            seed=seed,
            batch=batch,
            width=width,
            height=height,
            positive_prefix=positive_prefix,
            init_image_b64=init_b64,
            requested_by=message.author.display_name,
            requester_id=message.author.id,
            trigger_message_id=message.id,
            status_msg=status_msg,
        )
    except asyncio.CancelledError:
        return
    except Exception:
        logging.exception("Error in handle_image_message")
        await safe_send(message.channel, "Oops — image generation hit a snag.")
