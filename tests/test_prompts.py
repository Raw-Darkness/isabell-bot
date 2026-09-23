"""Deterministic clean-up applied to every model-written image prompt."""
from isabell import images, safety


def test_tidy_strips_quality_tags_underscores_duplicates_and_stray_ponies():
    out = images.tidy_tags("rating_safe, 1girl, pointed_ears, masterpiece, 8k, score_9, pointed ears, ponies, lighting: soft", "an elf")
    assert out == "rating_safe, 1girl, pointed ears, soft"


def test_tidy_keeps_ponies_when_asked_and_caps_length():
    assert "pony" in images.tidy_tags("rating_safe, pony, meadow", "a pony in a meadow")
    many = ", ".join(f"tag{i}" for i in range(50))
    assert images.tidy_tags(many).count(",") + 1 == images.MAX_TAGS


def test_rating_is_kept_moved_first_or_inferred():
    assert images.ensure_rating("1girl, beach, rating_questionable", "").startswith("rating_questionable, 1girl")
    assert images.ensure_rating("1girl, beach", "a woman on a beach").startswith("rating_safe")
    assert images.ensure_rating("1girl, beach", "a nude woman on a beach").startswith("rating_questionable")
    assert images.ensure_rating("1girl 1boy, bed", "explicit sex on a bed").startswith("rating_explicit")


def test_clip_tags_cuts_on_a_tag_boundary():
    assert images.clip_tags("alpha, beta, gamma", 14) == "alpha, beta"


def test_explicit_extras_only_when_not_rated_safe(monkeypatch):
    sent = {}

    async def fake_post(path, payload):
        sent["prompt"] = payload["prompt"]
        return {"images": [], "info": "{}"}
    monkeypatch.setattr(images, "_sd_post", fake_post)
    monkeypatch.setitem(images.config, "SDPositivePrompt", "score_9, ")
    monkeypatch.setitem(images.config, "SDExplicitExtras", "uncensored, <lora:X:1>")
    import asyncio
    asyncio.run(images.sd_generate(prompt="rating_safe, 1girl", negative=""))
    assert "uncensored" not in sent["prompt"]
    asyncio.run(images.sd_generate(prompt="rating_explicit, 1girl", negative=""))
    assert sent["prompt"].startswith("score_9, uncensored, <lora:X:1>, rating_explicit")


def test_e621_youth_tag_is_blocked():
    assert safety.image_prompt_blocked("a wolf cub, rating_explicit") == "cub"
    assert safety.image_prompt_blocked("anthro, cubs") is not None


PASTED = ("**Prompt:**\n```giant orc, muscular, green skin, (tusks:1.2), (from behind:1.3)```\n"
          "**Negative:** (child:2), (loli:2), score_4, score_5, bad anatomy\n**Seed:** `12345`")


def test_pasted_negative_is_cut_before_checking():
    cleaned = safety.strip_pasted_negative(PASTED)
    assert cleaned == "giant orc, muscular, green skin, (tusks:1.2), (from behind:1.3)"
    assert safety.image_prompt_blocked(cleaned) is None


def test_a1111_metadata_format_and_plain_text():
    assert safety.strip_pasted_negative("1girl, beach\nNegative prompt: child, blurry\nSteps: 20") == "1girl, beach"
    assert safety.strip_pasted_negative("a woman in negative space art style") == "a woman in negative space art style"


def test_blocked_term_in_the_positive_part_still_blocks():
    assert safety.image_prompt_blocked(safety.strip_pasted_negative("**Prompt:** a loli\n**Negative:** blurry")) == "loli"
