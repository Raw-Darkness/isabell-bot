# Isabell

Roleplay chat and image generation bot for an adults-only game community. Isabell has **no moderation features**; those live in the separate Barnabus bot. What she does have is every control needed to make sure the wrong kind of content is never produced, and a design where none of those controls can be turned off by accident.

## What she does

* **Roleplay chat** in character, with per-channel memory, summaries when the window fills, lore retrieval, and a loop breaker for repetitive replies.
* **Image generation** through Stable Diffusion WebUI Forge: `/draw`, natural-language requests, follow-up refinements (reply to an image or say "make it darker"), Redo / Variation / Upscale / Prompt buttons, style presets, and a `generate_image` tool the chat model can call itself.

## Safety, in layers

Every image and every chat turn passes through all of these. A refusal is logged to `refusals.jsonl` and posted to the mod channel; terms with no innocent use page `@here`, judgement calls are posted for a human to verify. The bot itself never bans or kicks anyone.

| Layer | What | Where |
|---|---|---|
| 1 | **Word filters.** Image prompts refuse any age or minor term, including leetspeak, spaced-out spellings and stated ages under 18; the built-in list cannot be shortened from config. Chat uses a tiered filter tuned for a breeding-roleplay community that carries age context across turns, on both the user's message and the model's reply. | `safety.py` |
| 2 | **Model second opinion.** A classifier reads the final image prompt (fails closed: no verdict, no image) and, optionally, each chat turn (fails open). Catches paraphrase that word lists miss. | `safety.classify_*` |
| 3 | **Vision check on every render.** Each generated image is reviewed by a vision model before it is posted. Anything judged to depict a minor is withheld and reported. Fails closed. | `safety.check_rendered_image` |
| 4 | **Channel gating.** The bot only operates in `AllowedChannels` that Discord flags as age-restricted, and not in DMs unless `AllowInDMs` is set. | `core.channel_allowed` |
| 5 | **Cooldown.** Three hard refusals in 24 hours and the bot stops responding to that account for 24 hours. Contextual matches never count. | `safety.user_on_cooldown` |

A blocked exchange never enters conversation memory, so it cannot steer later replies. The word filter is the same code as the monitor in Barnabus; keep the two in sync when tuning.

## Setup

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
cp Config.example.json Isabell.json   # token, OpenRouter key, channels, SD settings, persona
venv/bin/python -m isabell
```

Needs a reachable Forge API (`SDURL`) and an OpenRouter key. Config, lore and persona hot-reload. `LLMEnabled: false` silences chat entirely; `ImageGenerationEnabled: false` disables every drawing path.

### Discord application

Message Content intent is required. Invite with scopes `bot` + `applications.commands` and only these permissions: View Channels, Send Messages, Send Messages in Threads, Embed Links, Attach Files, Read Message History, Add Reactions. Mark every channel in `AllowedChannels` as age-restricted in Discord, or the bot will refuse to operate there.

## Files

| File | Purpose | Committed? |
|---|---|---|
| `Isabell.json` | real config, token, key, persona | no |
| `channel_history.json`, `dm_history/` | conversation memory | no |
| `image_memory.json` | recent prompts per channel, for buttons and follow-ups | no |
| `refusals.jsonl` | every refusal, for review | no |
| `world_lore.txt`, `world_lore_short.txt` | full and compact lore | no |

Owner commands by DM: `!reload`, `!clearhistory <channel_id>`, `!flags`.

## Privacy

See [Privacy.md](Privacy.md).
