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
from . import store
from .llm import chat_async, utility_model
from .safety import strip_pasted_negative, image_prompt_blocked, refuse_image_request, classify_image_prompt, check_rendered_image, user_on_cooldown


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
            store.write_json(self.path, {str(ch): [rec.__dict__ for rec in arr] for ch, arr in self.by_channel.items() if arr})
            self._dirty = False
        except Exception:
            logging.exception("Failed to save image memory to %s", self.path)

    def force_save(self):
        self._dirty = True
        self.save_if_dirty()

    def expire(self) -> int:
        cutoff, removed = store.retention_cutoff(), 0
        for ch in list(self.by_channel):
            keep = [r for r in self.by_channel[ch] if r.ts >= cutoff]
            removed += len(self.by_channel[ch]) - len(keep)
            if keep:
                self.by_channel[ch] = keep
            else:
                del self.by_channel[ch]
        if removed:
            self._dirty = True
        return removed

    def forget_user(self, user_id: int) -> int:
        removed = 0
        for ch in list(self.by_channel):
            keep = [r for r in self.by_channel[ch] if r.meta.get("by_id") != user_id]
            removed += len(self.by_channel[ch]) - len(keep)
            self.by_channel[ch] = keep
        if removed:
            self.force_save()
        return removed

    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        known = set(ImagePromptRecord.__dataclass_fields__)
        try:
            data = store.read_json(self.path)
            for ch, arr in data.items():
                self.by_channel[int(ch)] = [ImagePromptRecord(**{k: v for k, v in d.items() if k in known}) for d in arr]
            self._dirty = True   # re-save encrypted
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
    # Explicit-leaning extras (e.g. "uncensored" and an NSFW LoRA) are left out when the
    # prompt is rated safe; otherwise they push nudity into images nobody asked to be nude.
    extras = str(config.get("SDExplicitExtras") or "")
    if extras and positive_prefix is None and "rating_safe" not in (prompt or ""):
        prefix = prefix.rstrip().rstrip(",") + ", " + extras.strip().rstrip(",") + ", "
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
            # The negative prompt is not shown: members copy this text back as a new
            # request, and ours lists the very terms the filter refuses.
            text = (
                f"**Prompt:**\n```{(rec.final_sd_prompt or '')[:1700]}```\n"
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
            meta={"by": requested_by, "by_id": requester_id},
            bot_message_id=getattr(sent, "id", None),
            ts=time.time(),
        ))
    except asyncio.CancelledError:
        return
    except Exception:
        logging.exception("Image job failed")
        await safe_send(channel, "Oops — image generation hit a snag.")


# ---- Prompt writing for Pony Realism ------------------------------------------
# One guide for every path that writes a Stable Diffusion prompt: the rewrite of a
# member's request, follow-up refinements, and the chat model's generate_image tool.
# Pony-based models respond to booru-style tags in a rough subject-first order and
# are steered by rating tags; the text encoder reads 75-token chunks and weighs
# early tags most, so short and ordered beats long and "amplified".
def pony_guide(max_chars: int) -> str:
    name = (config.get("Name") or "the assistant").strip()
    return (
        "Write image prompts for a photorealistic SDXL model trained on booru-style tags. "
        "It understands tags, not sentences.\n"
        "FORMAT: one line of comma-separated tags, nothing else. Write tags with spaces, not underscores "
        "(the only exceptions are rating_safe, rating_questionable, rating_explicit).\n"
        "ORDER:\n"
        "1. rating tag: rating_explicit only for sexual content, rating_questionable for suggestive or "
        "nude-but-not-sexual, otherwise rating_safe\n"
        "2. who: the count and gender the request states (1girl, 1boy, 2girls, 1girl 1boy); if it states no "
        "gender, write solo or the count only; for scenery with nobody in it write no humans, never solo\n"
        "3. the subject: species or race, then appearance the request describes\n"
        "4. clothing as requested or as the scene implies\n"
        "5. pose and action\n"
        "6. setting and background\n"
        "7. lighting, and framing that fits: full body or wide shot for a scene, cowboy shot or upper body for a character\n"
        "RULES:\n"
        f"- Under {max_chars} characters and at most 30 tags. The most important tags go first. Stop when the scene is described.\n"
        "- Include only what the request states or clearly implies. Never add characters, places, nudity, "
        "sexual content or body-size details the request does not ask for.\n"
        "- Do not add quality tags (score_9, masterpiece, best quality, 8k, highly detailed): they are added automatically.\n"
        "- Weights only for what the request stresses: at most three, between 1.1 and 1.4, written (tag:1.2).\n"
        "- Everyone depicted is an adult. Never write tags that make anyone look young: child, young, teen, "
        "loli, shota, school uniform, flat chest, childlike or small proportions.\n"
        "- No negative-prompt terms, no repeated tags, no explanations.\n"
        f"- Do not mention {name} or any persona unless the request does."
    )


_KEEP_UNDERSCORE = ("rating_", "score_", "source_")
# Added automatically by the prefix, or meaningless to Pony: stripped if the model writes them.
_QUALITY_TAGS = {
    "masterpiece", "best quality", "high quality", "highest quality", "top quality", "ultra detailed",
    "ultra-detailed", "highly detailed", "extremely detailed", "detailed", "intricate", "intricate details",
    "4k", "8k", "uhd", "hdr", "hd", "high resolution", "absurdres", "photorealistic", "realistic",
    "score 9", "score 8 up", "score 7 up",
}
MAX_TAGS = 30
_RATINGS = ("rating_safe", "rating_questionable", "rating_explicit")
_EXPLICIT_RE = re.compile(
    r"\b(explicit|sex|fuck\w*|cock|dick|penis|pussy|vagina|cum\w*|penetrat\w*|anal|oral|blowjob|"
    r"breed\w*|mating|orgasm\w*|rape|tentacle sex|nsfw|hardcore)\b", re.I)
_NUDE_RE = re.compile(r"\b(nude|naked|topless|bottomless|nipples?|breasts? out|undress\w*|lingerie|bikini)\b", re.I)


def ensure_rating(prompt: str, request: str) -> str:
    """Pony is steered by its rating tag and the explicit extras depend on it, so every
    prompt gets exactly one, first. The model's choice wins; when it forgot, infer it."""
    tags = [t.strip() for t in prompt.split(",") if t.strip()]
    given = [t for t in tags if t.lower() in _RATINGS]
    rest = [t for t in tags if t.lower() not in _RATINGS]
    if given:
        rating = given[0].lower()
    else:
        text = f"{request} {prompt}"
        rating = ("rating_explicit" if _EXPLICIT_RE.search(text)
                  else "rating_questionable" if _NUDE_RE.search(text) else "rating_safe")
    return ", ".join([rating] + rest)


def tidy_tags(prompt: str, request: str = "") -> str:
    """Deterministic clean-up of a model-written prompt: spaces instead of underscores
    (except rating/score/source tags and LoRAs), no quality tags, no duplicates, no
    ponies nobody asked for (the base model's name leaks in), at most MAX_TAGS tags."""
    asked_pony = "pon" in (request or "").lower()
    seen, out = set(), []
    for tag in (prompt or "").split(","):
        tag = tag.strip()
        if not tag:
            continue
        if not tag.lower().startswith(_KEEP_UNDERSCORE) and "<lora:" not in tag:
            tag = tag.replace("_", " ")
        tag = re.sub(r"^[a-z ]{2,20}:\s*(?=[a-z])", "", tag, flags=re.I) if "<lora:" not in tag else tag
        key = re.sub(r"[()]|:\s*\d(\.\d+)?", "", tag).strip().lower()
        if key.startswith("score_") or key in _QUALITY_TAGS or key in seen:
            continue
        if not asked_pony and key in ("pony", "ponies", "my little pony"):
            continue
        seen.add(key)
        out.append(tag)
        if len(out) >= MAX_TAGS:
            break
    return ", ".join(out)


def clip_tags(prompt: str, max_chars: int) -> str:
    """Trim an over-long prompt at a tag boundary instead of mid-word."""
    prompt = (prompt or "").strip().strip(",")
    if len(prompt) <= max_chars:
        return prompt
    cut = prompt[:max_chars]
    return cut[:cut.rfind(",")].strip() if "," in cut else cut


def max_prompt_chars() -> int:
    return int(config.get("ImagePromptMaxChars", 600))


async def compile_sd_prompt(user_text: str) -> str:
    max_chars = max_prompt_chars()
    msgs = [{"role": "system", "content": pony_guide(max_chars)},
            {"role": "user", "content": f"REQUEST:\n{user_text}"}]
    try:
        raw = await chat_async(msgs, temperature=0.0, max_tokens=max(160, max_chars // 3), model=utility_model())
        raw = (raw or "").strip().strip("`").splitlines()[0] if (raw or "").strip() else ""
        raw = re.sub(r"\((\d(?:\.\d+)?)\)\s*([^,()\n]+)", lambda m: f"({m.group(2).strip()}:{m.group(1)})", raw)
        return clip_tags(ensure_rating(tidy_tags(raw, user_text), user_text), max_chars) or clip_tags(user_text, max_chars)
    except Exception:
        logging.exception("LLM prompt compose failed; returning user text")
        return clip_tags(user_text, max_chars)


async def refine_image_prompt(last: ImagePromptRecord, followup_text: str) -> dict[str, str]:
    max_chars = max_prompt_chars()
    system = (
        pony_guide(max_chars)
        + "\n\nTASK: you are given the previous prompt and a follow-up instruction. Keep the previous "
        "subject, style and details, apply ONLY the change the follow-up asks for, and keep the rating tag "
        "consistent with the result. Output JSON only: {\"prompt\": \"...\", \"negative\": \"...\"} where "
        "negative is the previous negative prompt, changed only if the follow-up asks to remove something."
    )
    user = f"Previous: {last.final_sd_prompt}\nNegative: {last.negative_prompt}\nFollow-up: {followup_text}"
    try:
        raw = await chat_async([{"role": "system", "content": system}, {"role": "user", "content": user}],
                               temperature=0.0, max_tokens=max(200, max_chars // 2), model=utility_model())
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("bad json")
        req = f"{last.user_prompt} {followup_text}"
        data["prompt"] = clip_tags(ensure_rating(tidy_tags(data.get("prompt") or last.final_sd_prompt, req), req), max_chars)
        data["negative"] = data.get("negative") or last.negative_prompt
        return data
    except Exception:
        logging.warning("Refine parse failed; appending the follow-up")
        return {"prompt": clip_tags(f"{last.final_sd_prompt}, {followup_text}", max_chars),
                "negative": last.negative_prompt}


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
                        "Booru-style comma-separated tags for Pony Realism, under 600 characters, in this "
                        "order: rating tag (rating_safe / rating_questionable / rating_explicit), who "
                        "(1girl, 1boy, solo...), subject, clothing, pose, setting, lighting and framing. "
                        "Use the conversation for context the user left implicit, but add nothing they did "
                        "not ask for. Spaces not underscores. No quality tags, no sentences. Everyone "
                        "depicted is an adult."
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
    prompt = strip_pasted_negative(args.get("prompt") or "")
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
        sd_prompt=clip_tags(ensure_rating(tidy_tags(prompt, message.content or ""), message.content or ""), max_prompt_chars()),
        neg=config.get("SDNegativePrompt", "(lowres, blurry, deformed)"),
        width=width,
        height=height,
        requested_by=message.author.display_name,
        requester_id=message.author.id,
        trigger_message_id=message.id,
        status_msg=status,
    ))
    return True
