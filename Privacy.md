# Isabell Privacy Policy

*Last updated: 2026-09-23*

Isabell is a chat and image generation bot operated by the staff of the Wicked Island Discord server. This document explains what data the bot processes, why, where it goes and for how long.

## 1. What the bot collects

**Conversation memory.** To hold a conversation the bot keeps recent messages from the channels it is allowed in (up to 40 turns per channel, with a summary of older turns), including display names. It is stored on the operator's machine so the conversation survives restarts.

**Image request memory.** The last 20 image prompts per channel, with the settings used, so that "make it darker" and the Redo/Variation/Upscale buttons work.

**Refusal log.** When a request is refused by a safety control, the bot records the user ID, display name, the reason and a short excerpt, for the moderation team to review.

**Logs.** A local log file with errors and events, rotated daily and deleted after 7 days.

## 2. Third-party processing

Text is sent to OpenRouter, an AI model gateway, to generate replies, rewrite image prompts, and run the safety classifier. Generated images are also sent to a vision model there for a safety review before they are posted. The operator configures requests so that data is not used to train models. Image generation itself runs on hardware the operator controls.

## 3. Content controls and automated decisions

The bot refuses any request for, and withholds any output of, sexual content involving minors, using word filters, a model classifier and a review of every generated image. Refusals are reported to human moderators. The only automatic consequence is that an account which repeatedly triggers hard refusals is ignored by the bot for 24 hours. The bot never removes, bans or restricts anyone in the server.

## 4. Retention and storage

All stored message content is **encrypted at rest** with a key kept separately from the data, and **deleted automatically after 30 days**. A conversation summary is deleted once the oldest conversation it was built from passes 30 days.

| Data | Kept for |
|---|---|
| Conversation memory and summaries | at most 30 days |
| Image request memory | at most 30 days, and only the last 20 per channel |
| Refusal log | 30 days; the alert posted to moderators stays in Discord |
| Log files | 7 days |

## 5. Your rights

Ask any moderator in the server to delete what the bot holds about you. The operator removes it with a single command, which deletes your direct-message history, every channel conversation you took part in, your image requests and any refusal entries. Data is not shared with anyone outside the server's moderation team, other than the processing described in section 2.
