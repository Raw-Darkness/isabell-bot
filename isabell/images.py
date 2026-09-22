"""Stable Diffusion: prompt memory, the Forge API client, the generation job with
its safety gates, persistent buttons, prompt compose/refine, and the model tool."""
import asyncio
import base64
import io
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import discord
from PIL import Image

from .core import config, bot, safe_send, channel_key, get_user_bucket, image_channel_allowed, images_enabled, image_unavailable
from .llm import chat_async, utility_model
from .safety import image_prompt_blocked, refuse_image_request, classify_image_prompt, check_rendered_image, user_on_cooldown


@dataclass
class ImagePromptRecord:
    channel_id: int
    message_id: int
    user_prompt: str
    final_sd_prompt: str
    negative_prompt: str = ""
    seed: int = -1
    width: int = 0
    height: int = 0
    positive_prefix: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    bot_message_id: int | None = None
    ts: float = 0.0


class ImagePromptMemory:
    """Per-channel image history, persisted so follow-ups and buttons survive restarts."""

    def __init__(self, path: str | None = None, max_per_channel: int = 20):
        self.by_channel: dict[int, list[ImagePromptRecord]] = {}
        self.path = path
        self.max_per_channel = max_per_channel
        self._dirty = False
        self._load()

    def add(self, rec: ImagePromptRecord):
        arr = self.by_channel.setdefault(rec.channel_id, [])
        arr.append(rec)
        del arr[:-self.max_per_channel]
        self._dirty = True

    def last_for_channel(self, channel_id: int) -> ImagePromptRecord | None:
        arr = self.by_channel.get(channel_id, [])
        return arr[-1] if arr else None

    def find_by_message(self, message_id: int | None) -> ImagePromptRecord | None:
        if not message_id:
            return None
        for arr in self.by_channel.values():
            for rec in reversed(arr):
                if rec.bot_message_id == message_id:
                    return rec
        return None

    def save_if_dirty(self):
        if not self._dirty or not self.path:
            return
        try:
            data = {str(ch): [rec.__dict__ for rec in arr] for ch, arr in self.by_channel.items() if arr}
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self.path)
            self._dirty = False
        except Exception:
            logging.exception("Failed to save image memory to %s", self.path)

    def force_save(self):
        self._dirty = True
        self.save_if_dirty()

    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        known = set(ImagePromptRecord.__dataclass_fields__)
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for ch, arr in data.items():
                self.by_channel[int(ch)] = [
                    ImagePromptRecord(**{k: v for k, v in d.items() if k in known}) for d in arr
                ]
            logging.info("Loaded image memory for %d channels", len(self.by_channel))
        except Exception:
            logging.exception("Failed to load image memory from %s", self.path)


ipm = ImagePromptMemory(path=config.get("ImageMemoryPath", "image_memory.json"))

# Latest generated image per channel (base64 PNG) for img2img refinements.
# In-memory only — after a restart, refinements fall back to seed reuse.
_last_image_b64: dict[int, str] = {}

def image_ok(img: Image.Image | None) -> bool:
    if img is None:
        return False
    try:
        w, h = img.size
        return w > 0 and h > 0
    except Exception:
        return False


class SDOfflineError(Exception):
    """The Stable Diffusion backend cannot be reached at all."""


def _sd_base_url() -> str:
    return config["SDURL"].split("/sdapi/")[0]


async def _sd_post(path: str, payload: dict) -> dict:
    timeout = aiohttp.ClientTimeout(total=int(config.get("SDTimeout", 180)))
    try:
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(_sd_base_url() + path, json=payload) as r:
                r.raise_for_status()
                return await r.json()
    except aiohttp.ClientConnectorError as e:
        logging.warning("SD backend unreachable: %s", e)
        raise SDOfflineError(str(e)) from e


async def _sd_get(path: str) -> dict | None:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
            async with s.get(_sd_base_url() + path) as r:
                r.raise_for_status()
                return await r.json()
    except Exception:
        return None


async def sd_generate(
    *,
    prompt: str,
    negative: str,
    seed: int = -1,
    subseed_strength: float = 0.0,
    hires: bool = False,
    batch: int = 1,
    width: int | None = None,
    height: int | None = None,
    positive_prefix: str | None = None,
    init_image_b64: str | None = None,
) -> tuple[list[Image.Image], int, str | None]:
    """txt2img, or img2img when an init image is given.

    Returns (images, seed_used, first_image_b64)."""
    prefix = positive_prefix if positive_prefix is not None else config["SDPositivePrompt"]
    batch = max(1, min(int(config.get("SDMaxBatch", 4)), batch))
    payload = {
        "prompt": prefix + prompt,
        "negative_prompt": negative,
        "steps": config["SDSteps"],
        "width": width or config["SDWidth"],
        "height": height or config["SDHeight"],
        "cfg_scale": config["cfg_scale"],
        "sampler_index": config["SDSampler"],
        "seed": seed,
        "batch_size": batch,
    }
    if config.get("scheduler"):
        payload["scheduler"] = config["scheduler"]
    if subseed_strength > 0:
        payload["subseed"] = -1
        payload["subseed_strength"] = subseed_strength

    endpoint = "/sdapi/v1/txt2img"
    if init_image_b64:
        endpoint = "/sdapi/v1/img2img"
        payload["init_images"] = [init_image_b64]
        payload["denoising_strength"] = float(config.get("SDImg2ImgDenoise", 0.5))
    elif hires:
        payload["enable_hr"] = True
        payload["hr_scale"] = float(config.get("SDUpscaleFactor", 2.0))
        payload["hr_upscaler"] = config.get("SDHiresUpscaler", "Latent")
        payload["denoising_strength"] = 0.4
        # Forge defaults this to None and then tests membership on it, so an API
        # hires request without it fails with "argument of type 'NoneType' is not
        # iterable". "Use same choices" means "keep the base model's modules".
        payload["hr_additional_modules"] = ["Use same choices"]

    try:
        j = await _sd_post(endpoint, payload)
    except SDOfflineError:
        raise
    except Exception as e:
        logging.exception("SD generation failed: %s", e)
        return [], seed, None

    raw_images = (j.get("images") or [])[:batch]
    images: list[Image.Image] = []
    for b in raw_images:
        try:
            images.append(Image.open(io.BytesIO(base64.b64decode(b))))
        except Exception:
            logging.exception("Failed to decode SD image")
    used_seed = seed
    try:
        used_seed = int(json.loads(j.get("info") or "{}").get("seed", seed))
    except Exception:
        pass
    return images, used_seed, (raw_images[0] if raw_images else None)


# One GPU — serialize generations ourselves so a queued request waits with
# feedback instead of burning its HTTP timeout inside the SD backend.
_sd_semaphore: asyncio.Semaphore | None = None
_sd_waiting = 0


def _get_sd_semaphore() -> asyncio.Semaphore:
    global _sd_semaphore
    if _sd_semaphore is None:
        _sd_semaphore = asyncio.Semaphore(int(config.get("SDMaxConcurrent", 1)))
    return _sd_semaphore


async def _progress_updates(status_msg, gen_task: asyncio.Task):
    """Edit the status message with live progress while a generation runs."""
    if status_msg is None:
        return
    try:
        while not gen_task.done():
            await asyncio.sleep(4)
            if gen_task.done():
                return
            j = await _sd_get("/sdapi/v1/progress?skip_current_image=true")
            if not j:
                continue
            pct = int(float(j.get("progress") or 0) * 100)
            if pct <= 0:
                continue
            eta = int(float(j.get("eta_relative") or 0))
            text = f"🎨 {pct}%" + (f" · ~{eta}s left" if eta > 0 else "")
            try:
                await status_msg.edit(content=text)
            except Exception:
                return
    except asyncio.CancelledError:
        return


class ImageActionsView(discord.ui.View):
    """Persistent buttons attached to every generated image."""

    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="Redo", emoji="🔁", style=discord.ButtonStyle.secondary, custom_id="imggen:redo")
    async def redo(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _image_button(interaction, "redo")

    @discord.ui.button(label="Variation", emoji="✨", style=discord.ButtonStyle.secondary, custom_id="imggen:vary")
    async def vary(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _image_button(interaction, "vary")

    @discord.ui.button(label="Upscale", emoji="⬆️", style=discord.ButtonStyle.secondary, custom_id="imggen:upscale")
    async def upscale(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _image_button(interaction, "upscale")

    @discord.ui.button(label="Prompt", emoji="📋", style=discord.ButtonStyle.secondary, custom_id="imggen:prompt")
    async def prompt(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _image_button(interaction, "prompt")


async def _image_button(interaction: discord.Interaction, action: str):
    try:
        rec = ipm.find_by_message(interaction.message.id if interaction.message else None)
        if rec is None:
            await interaction.response.send_message("I no longer remember this image, sorry.", ephemeral=True)
            return

        if action == "prompt":
            text = (
                f"**Prompt:**\n```{(rec.final_sd_prompt or '')[:1700]}```\n"
                f"**Negative:** {(rec.negative_prompt or '')[:300]}\n"
                f"**Seed:** `{rec.seed}`"
            )
            await interaction.response.send_message(text, ephemeral=True)
            return

        if not images_enabled():
            await interaction.response.send_message(
                config.get("ImageDisabledNotice", "Image generation is currently disabled."),
                ephemeral=True)
            return
        blocked = image_prompt_blocked(rec.final_sd_prompt) or image_prompt_blocked(rec.user_prompt)
        if blocked:
            await interaction.response.send_message(config.get(
                "ImageRefusalMessage",
                "No. That is not something I will ever draw, and the moderators have been notified."), ephemeral=True)
            await refuse_image_request(interaction.channel, interaction.user.id,
                                       interaction.user.display_name, blocked, rec.final_sd_prompt)
            return
        if not get_user_bucket(interaction.user.id).consume():
            await interaction.response.send_message("You're requesting images too fast — slow down a bit.", ephemeral=True)
            return

        labels = {"redo": "Rolling a fresh take…", "vary": "Painting a variation…", "upscale": "Upscaling…"}
        await interaction.response.send_message(f"🎨 {labels.get(action, 'Working…')}")
        status_msg = await interaction.original_response()

        kwargs: dict[str, Any] = dict(
            ch_id=rec.channel_id,
            user_prompt=rec.user_prompt,
            sd_prompt=rec.final_sd_prompt,
            neg=rec.negative_prompt,
            positive_prefix=rec.positive_prefix or None,
            width=rec.width or None,
            height=rec.height or None,
            requested_by=interaction.user.display_name,
            requester_id=interaction.user.id,
            status_msg=status_msg,
        )
        if action == "vary":
            kwargs.update(seed=rec.seed, subseed_strength=float(config.get("SDVariationStrength", 0.35)))
        elif action == "upscale":
            kwargs.update(seed=rec.seed, hires=True)
        asyncio.create_task(run_image_job(interaction.channel, **kwargs))
    except Exception:
        logging.exception("Image button %s failed", action)


async def run_image_job(
    channel,
    *,
    ch_id: int,
    user_prompt: str,
    sd_prompt: str,
    neg: str,
    seed: int = -1,
    subseed_strength: float = 0.0,
    hires: bool = False,
    batch: int = 1,
    width: int | None = None,
    height: int | None = None,
    positive_prefix: str | None = None,
    init_image_b64: str | None = None,
    requested_by: str = "",
    requester_id: int = 0,
    trigger_message_id: int = 0,
    status_msg=None,
):
    """Queue a generation, show progress, deliver the result with action buttons."""
    global _sd_waiting
    try:
        # Last line of defence: every image path funnels through here, including
        # buttons, refinements and LLM-rewritten prompts.
        if not images_enabled():
            await image_unavailable(channel)
            return
        if requester_id and user_on_cooldown(requester_id):
            return
        blocked = image_prompt_blocked(sd_prompt) or image_prompt_blocked(user_prompt)
        if blocked:
            await refuse_image_request(channel, requester_id, requested_by or "unknown",
                                       blocked, sd_prompt)
            return
        # Layer 2: a model reads the final prompt too. Word lists miss paraphrase;
        # this fails closed, so an unreviewed prompt is never rendered.
        verdict = await classify_image_prompt(f"{user_prompt}\n---\n{sd_prompt}")
        if verdict:
            await refuse_image_request(channel, requester_id, requested_by or "unknown", verdict, sd_prompt)
            return
        sem = _get_sd_semaphore()
        queued = sem.locked()
        if queued:
            _sd_waiting += 1
            await safe_send(channel, f"🎨 The easel is busy — you're #{_sd_waiting} in line.")
        try:
            async with sem:
                if queued:
                    _sd_waiting = max(0, _sd_waiting - 1)
                async with channel.typing():
                    gen_task = asyncio.create_task(sd_generate(
                        prompt=sd_prompt, negative=neg, seed=seed,
                        subseed_strength=subseed_strength, hires=hires, batch=batch,
                        width=width, height=height, positive_prefix=positive_prefix,
                        init_image_b64=init_image_b64,
                    ))
                    progress_task = asyncio.create_task(_progress_updates(status_msg, gen_task))
                    try:
                        images, seed_used, first_b64 = await gen_task
                    finally:
                        progress_task.cancel()
        except SDOfflineError:
            await safe_send(channel, config.get("SDOfflineNotice", "The image engine is offline right now — try again later."))
            return

        images = [im for im in images if image_ok(im)]
        if not images:
            await safe_send(channel, "I couldn't render that image — try tweaking the description.")
            return

        # Layer 3: look at what was actually rendered before anyone else does.
        # Every image in the batch is reviewed; one failure withholds the whole post.
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            if not await check_rendered_image(base64.b64encode(buf.getvalue()).decode("ascii")):
                logging.warning("Image WITHHELD by output check | user=%s (%s) | prompt=%r", requested_by, requester_id, sd_prompt[:200])
                await refuse_image_request(channel, requester_id, requested_by or "unknown",
                                           "classifier: rendered image judged to depict a minor", sd_prompt)
                return

        files = []
        for i, img in enumerate(images):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            files.append(discord.File(buf, filename=f"output_{i + 1}.png"))

        content = f"🎨 for **{requested_by}** — use the buttons to iterate." if requested_by else None
        sent = await safe_send(channel, content, files=files, view=ImageActionsView())

        if status_msg is not None:
            try:
                await status_msg.delete()
            except Exception:
                pass

        if first_b64:
            _last_image_b64[ch_id] = first_b64
        ipm.add(ImagePromptRecord(
            channel_id=ch_id,
            message_id=trigger_message_id,
            user_prompt=user_prompt,
            final_sd_prompt=sd_prompt,
            negative_prompt=neg,
            seed=seed_used,
            width=width or 0,
            height=height or 0,
            positive_prefix=positive_prefix or "",
            meta={"by": requested_by},
            bot_message_id=getattr(sent, "id", None),
            ts=time.time(),
        ))
    except asyncio.CancelledError:
        return
    except Exception:
        logging.exception("Image job failed")
        await safe_send(channel, "Oops — image generation hit a snag.")


async def compile_sd_prompt(user_text: str) -> str:
    max_chars = int(config.get("ImagePromptMaxChars", 1600))
    name = (config.get("Name") or "the assistant").strip()

    system = (
        "You are an expert prompt engineer for Pony Realism SDXL models.\n\n"
        "Rewrite the USER PROMPT into ONE comma-separated line of tags optimized for Pony Realistic SDXL.\n\n"
        "RULES:\n"
        "- Output exactly one line of pure tags. No quotes, no explanations.\n"
        f"- Stay under {max_chars} characters.\n"
        "- Amplify and clarify every visual and aesthetic element from the user's description.\n"
        "- Use ( ) with weights for emphasis, e.g. (detailed eyes:1.3)\n"
        "- Add relevant body/lighting/camera tags when implied by the scene.\n"
        "- NEVER add characters, locations, or elements not implied by the user prompt.\n"
        f"- NEVER mention {name} or any persona metadata unless the USER PROMPT explicitly references it.\n"
    )

    user = f"USER PROMPT:\n{user_text}"
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    try:
        tok_budget = max(256, min(2000, max_chars // 3))
        raw = await chat_async(msgs, temperature=0.0, max_tokens=tok_budget, model=utility_model())
        raw = (raw or "").strip().strip("`")
        raw = re.sub(r"\((\d(?:\.\d+)?)\)\s*([^,()\n]+)", lambda m: f"({m.group(2).strip()}:{m.group(1)})", raw)
        return raw[:max_chars]
    except Exception:
        logging.exception("LLM prompt compose failed; returning user text")
        return (user_text or "")[:max_chars]


async def refine_image_prompt(last: ImagePromptRecord, followup_text: str) -> dict[str, str]:
    max_chars = int(config.get("ImagePromptMaxChars", 1600))
    name = (config.get("Name") or "the assistant").strip()

    system = (
        "Refine a Stable Diffusion prompt based on a follow-up instruction.\n"
        "- Preserve subject, style, and descriptors from the previous prompt.\n"
        "- Merge ONLY new instructions from the follow-up.\n"
        f"- Do NOT introduce {name} or any persona unless explicitly mentioned.\n"
        f"- Keep under {max_chars} chars.\n"
        '- Output JSON: {"prompt":"...","negative":"..."}.'
    )
    user = (
        f"Previous: {last.final_sd_prompt}\n"
        f"Negative: {last.negative_prompt}\n"
        f"Follow-up: {followup_text}"
    )
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        raw = await chat_async(msgs, temperature=0.0, max_tokens=max(256, min(2000, max_chars // 3)), model=utility_model())
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("bad json")
        data["prompt"] = (data.get("prompt", last.final_sd_prompt) or "")[:max_chars]
        return data
    except Exception:
        logging.warning("Refine parse failed; fallback append")
        return {
            "prompt": f"{last.final_sd_prompt}, {followup_text}"[:max_chars],
            "negative": last.negative_prompt,
        }


IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": (
            "Draw and post a picture. Call this ONLY when the user is asking to be shown "
            "or drawn something. Never call it for ordinary conversation, roleplay narration, "
            "or when the user is merely describing or commenting on something visual."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "A detailed comma-separated Stable Diffusion prompt for the image the user "
                        "wants, using the conversation for any context they left implicit."
                    ),
                },
                "aspect": {"type": "string", "enum": ["square", "portrait", "landscape"]},
            },
            "required": ["prompt"],
        },
    },
}


def image_tools_for(message: discord.Message):
    """The tool list for this message, or None when model-decided drawing is off."""
    if not config.get("ImageToolEnabled", False) or not images_enabled():
        return None
    if not image_channel_allowed(message.channel):
        return None
    return [IMAGE_TOOL]


async def run_tool_image(message: discord.Message, args: dict) -> bool:
    """Act on a generate_image tool call. Returns True if a job was started."""
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return False
    if not images_enabled():
        await image_unavailable(message.channel)
        return True
    blocked = image_prompt_blocked(prompt)
    if blocked:
        await refuse_image_request(message.channel, message.author.id,
                                   message.author.display_name, blocked, prompt)
        return True
    if not get_user_bucket(message.author.id).consume():
        await safe_send(message.channel, "You're requesting images too fast — slow down a bit.")
        return True
    width = height = None
    aspect = (args.get("aspect") or "").lower()
    if aspect in ("portrait", "landscape"):
        size = config.get("SDPortraitSize" if aspect == "portrait" else "SDLandscapeSize") or []
        if len(size) == 2:
            width, height = int(size[0]), int(size[1])
    status = await safe_send(message.channel, "Hang on while I sketch that for you…")
    logging.info("Image tool fired | ch=%s | prompt=%r", channel_key(message), prompt[:90])
    asyncio.create_task(run_image_job(
        message.channel,
        ch_id=channel_key(message),
        user_prompt=message.content or "",
        sd_prompt=prompt[: int(config.get("ImagePromptMaxChars", 1600))],
        neg=config.get("SDNegativePrompt", "(lowres, blurry, deformed)"),
        width=width,
        height=height,
        requested_by=message.author.display_name,
        requester_id=message.author.id,
        trigger_message_id=message.id,
        status_msg=status,
    ))
    return True
