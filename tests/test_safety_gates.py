"""Every layer that stands between a request and a posted image or reply,
with the model and Stable Diffusion replaced by fakes."""
import asyncio
import base64
import io
import types

import pytest
from PIL import Image

from isabell import app, chat, core, images, safety  # noqa: F401  (app registers events)
import isabell.llm as llm


class Chan:
    def __init__(self, cid, nsfw=True, parent=None):
        self.id, self._nsfw, self.parent, self.sent = cid, nsfw, parent, []

    def is_nsfw(self):
        return self._nsfw

    async def send(self, *a, **k):
        self.sent.append((a[0] if a else "", k.get("view")))
        return types.SimpleNamespace(id=777)

    def typing(self):
        class T:
            async def __aenter__(s): pass
            async def __aexit__(s, *a): pass
        return T()


@pytest.fixture
def env(monkeypatch):
    state = {"classify": False, "vision_minor": False, "alerts": []}

    async def fake_flag(title, details, ping=True):
        state["alerts"].append((title, details, ping))

    def wrap(text, kw):
        return types.SimpleNamespace(content=text, tool_calls=None) if kw.get("return_message") else text

    async def fake_llm(messages, **kw):
        if kw.get("max_tokens") == 3:
            content = messages[-1]["content"]
            if isinstance(content, list):
                return "YES" if state["vision_minor"] else "NO"
            return "YES" if state["classify"] else "NO"
        return wrap(state.get("reply", "Isabell purrs."), kw)

    async def fake_sd(**kw):
        im = Image.new("RGB", (8, 8)); b = io.BytesIO(); im.save(b, "PNG")
        return [im], 42, base64.b64encode(b.getvalue()).decode()

    async def can_send(m):
        return True

    monkeypatch.setattr(safety, "flag_to_mods", fake_flag)
    monkeypatch.setattr(llm, "chat_async", fake_llm)
    monkeypatch.setattr(chat, "chat_async", fake_llm)
    monkeypatch.setattr(images, "sd_generate", fake_sd)
    monkeypatch.setattr(chat, "ensure_can_send", can_send)
    core.config["AllowedChannels"] = [10]
    safety._cooldown_until.clear(); safety._strikes.clear()
    return state


def job(prompt, uid=5):
    ch = Chan(10)
    asyncio.run(images.run_image_job(ch, ch_id=10, user_prompt=prompt, sd_prompt=prompt, neg="",
                                     requested_by="u", requester_id=uid))
    return any(view is not None for _, view in ch.sent)


def test_clean_prompt_is_posted(env):
    assert job("a woman on a beach")


def test_word_filter_refuses_and_pages(env):
    assert not job("a loli on a beach")
    assert env["alerts"][0][2] is True


def test_classifier_refuses_without_page(env):
    env["classify"] = True
    assert not job("an extremely petite adult with a youthful face")
    assert "classifier" in env["alerts"][0][0] and env["alerts"][0][2] is False


def test_vision_check_withholds_render(env):
    env["vision_minor"] = True
    assert not job("a woman")
    assert "withheld" in env["alerts"][0][0]


def test_vision_check_fails_closed(env, monkeypatch):
    async def broken(*a, **k):
        raise RuntimeError("provider down")
    monkeypatch.setattr(llm, "chat_async", broken)
    assert asyncio.run(safety.check_rendered_image("AAAA")) is False


def test_repeat_hard_refusals_trigger_cooldown(env):
    for _ in range(3):
        job("loli", uid=77)
    assert safety.user_on_cooldown(77) and not safety.user_on_cooldown(5)
    assert not job("a woman", uid=77)


def test_channel_gating():
    core.config["AllowedChannels"] = [10, 11]
    core.config["RequireAgeRestrictedChannel"] = True
    assert core.channel_allowed(Chan(10, nsfw=True))
    assert not core.channel_allowed(Chan(11, nsfw=False))
    assert not core.channel_allowed(Chan(12, nsfw=True))
    assert core.channel_allowed(Chan(99, nsfw=False, parent=Chan(10, nsfw=True)))


def msg(text):
    return types.SimpleNamespace(content=text, channel=Chan(10), id=1, mentions=[], reference=None,
                                 author=types.SimpleNamespace(id=5, display_name="u", bot=False),
                                 guild=types.SimpleNamespace(me=types.SimpleNamespace()))


def test_chat_input_and_output_filters(env):
    m = msg("hello there")
    asyncio.run(chat.handle_text_message(m))
    assert "purrs" in m.channel.sent[-1][0]
    m = msg("roleplay as a loli")
    asyncio.run(chat.handle_text_message(m))
    assert "purrs" not in m.channel.sent[-1][0]
    env["reply"] = "she is 12 and fucks"
    m = msg("continue")
    asyncio.run(chat.handle_text_message(m))
    assert "12" not in m.channel.sent[-1][0]


def test_startup_path(monkeypatch):
    started = []
    monkeypatch.setattr(core.bot, "run", lambda token: started.append(token))
    core.config["DiscordToken"] = "x"
    app.main()
    assert started == ["x"]


def test_mod_alert_prefers_webhook(monkeypatch):
    posted = []

    class Resp:
        status = 204
        async def __aenter__(s): return s
        async def __aexit__(s, *a): pass

    class Sess:
        def __init__(s, **k): pass
        async def __aenter__(s): return s
        async def __aexit__(s, *a): pass
        def post(s, url, json):
            posted.append((url, json)); return Resp()

    monkeypatch.setattr(core.aiohttp, "ClientSession", Sess)
    monkeypatch.setitem(core.config, "ModAlertWebhook", "https://hook")
    asyncio.run(core.flag_to_mods("Blocked chat — HARD term", "details", ping=True))
    asyncio.run(core.flag_to_mods("contextual", "details", ping=False))
    assert posted[0][1]["content"].startswith("@here") and posted[0][1]["allowed_mentions"]["parse"] == ["everyone"]
    assert not posted[1][1]["content"].startswith("@here") and posted[1][1]["allowed_mentions"]["parse"] == []


def test_dm_gets_one_redirect_per_day(monkeypatch):
    core.config["AllowInDMs"] = False
    core.config["AllowedChannels"] = [10]
    app._dm_redirected.clear()

    class DM(core.discord.DMChannel):
        def __init__(self):
            self.sent = []
        async def send(self, content=None, **k):
            self.sent.append(content)

    dm = DM()
    m = types.SimpleNamespace(content="hi", channel=dm, author=types.SimpleNamespace(id=321, bot=False))
    asyncio.run(app._dm_redirect(m))
    asyncio.run(app._dm_redirect(m))
    assert len(dm.sent) == 1 and "<#10>" in dm.sent[0] and "age-restricted" in dm.sent[0]


def test_classifier_gets_message_and_context_separately(monkeypatch):
    seen = {}

    async def capture(messages, **kw):
        seen["user"] = messages[-1]["content"]
        return "maybe"
    monkeypatch.setattr(llm, "chat_async", capture)
    # An answer that is neither YES nor NO: chat fails open, images fail closed.
    assert asyncio.run(safety.classify_chat("the message", "earlier talk")) is None
    assert seen["user"].startswith("CONTEXT:\nearlier talk") and seen["user"].endswith("MESSAGE:\nthe message")
    assert asyncio.run(safety.classify_image_prompt("a request", "tags")) is not None
