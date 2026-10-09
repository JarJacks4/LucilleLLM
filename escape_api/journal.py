"""Journal logic: prompts, themes, reflections, weekly letters. LLM use is opt-in, cached and capped."""

from __future__ import annotations

import logging
import random
import re
from collections import Counter
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from escape_api import core, llm
from escape_api.repo import Repo, user_path

logger = logging.getLogger(__name__)

MODES = ("free", "guided", "gratitude", "ritual")

# Cheap theme tagging (no LLM). Keys are theme names shown in Insights.
THEMES = {
    "work": ["work", "job", "boss", "meeting", "deadline", "office", "shift", "coworker", "project", "career"],
    "school": ["school", "class", "exam", "homework", "study", "teacher", "college", "grade"],
    "sleep": ["sleep", "slept", "insomnia", "tired", "nap", "bed", "awake", "dream"],
    "family": ["mom", "dad", "mother", "father", "sister", "brother", "family", "kids", "son", "daughter", "parents"],
    "friends": ["friend", "friends", "hang out", "party", "texted"],
    "love": ["partner", "boyfriend", "girlfriend", "husband", "wife", "date", "relationship", "love"],
    "health": ["sick", "doctor", "pain", "headache", "health", "medication", "therapy"],
    "body": ["workout", "gym", "run", "yoga", "walk", "stretch", "exercise", "ate", "food"],
    "money": ["money", "rent", "bills", "debt", "paycheck", "budget", "expensive"],
    "gratitude": ["grateful", "thankful", "appreciate", "blessed", "gratitude"],
    "creativity": ["music", "art", "write", "writing", "paint", "create", "design", "song"],
    "self-worth": ["not enough", "failure", "proud", "confidence", "worth", "compare"],
}

ENERGY_CENTER_FOR_THEME = {
    "work": "power", "school": "power", "money": "grounding", "sleep": "grounding", "health": "grounding",
    "body": "grounding", "family": "connection", "friends": "connection", "love": "connection",
    "gratitude": "connection", "creativity": "creativity", "self-worth": "power",
}


def extract_themes(text: str, limit: int = 4) -> List[str]:
    t = f" {text.lower()} "
    counts = {k: sum(len(re.findall(r"\b" + re.escape(w) + r"\b", t)) for w in words) for k, words in THEMES.items()}
    return [k for k, v in sorted(counts.items(), key=lambda kv: -kv[1]) if v > 0][:limit]


def entry_text(e: Dict[str, Any]) -> str:
    parts = [e.get("title") or "", e.get("body") or "", " ".join(e.get("gratitude") or []),
             e.get("intention") or "", (e.get("voice") or {}).get("transcript") or ""]
    return "\n".join(p for p in parts if p).strip()


def dataset_prompt(mode: str, word: Optional[str], seed: Optional[int] = None) -> Dict[str, Any]:
    mode = mode if mode in MODES else "free"
    d = core.dataset("journal_prompts")["modes"][mode]
    w = core.find_word(word) if word else None
    tone = core.tone_for(w["valence"], w["energy"], w["family"] == "tender") if w else "mixed"
    options = d["prompts"].get(tone) or d["prompts"]["mixed"]
    rnd = random.Random(seed)
    text = rnd.choice(options).replace("{word}", word.lower() if word else "a little mixed")
    return {"mode": mode, "label": d["label"], "prompt": text, "placeholder": d["placeholder"],
            "tone": tone, "source": "dataset", "aiGenerated": False}


def modes_list() -> List[Dict[str, Any]]:
    d = core.dataset("journal_prompts")["modes"]
    return [{"key": k, "label": v["label"], "sub": v["sub"], "icon": v["icon"], "color": v["color"]} for k, v in d.items()]


def _fallback_reflection(e: Dict[str, Any], word: str, family: str) -> Dict[str, Any]:
    mode = e.get("mode")
    if mode == "gratitude":
        refl = "Noticing good things on purpose is a skill, and you just practised it. Those three moments are yours to keep."
    elif mode == "ritual":
        refl = "One clear intention is enough. Let it be the last thing you think about tonight."
    elif family in ("restless", "heavy"):
        refl = (f"It sounds like a lot is sitting with you, and you still made space to write it down. "
                f"Naming that you feel {word.lower()} is often where it starts to loosen.")
    else:
        refl = f"Thank you for writing this. There's something {word.lower()} in it worth remembering."
    reframe = None
    if family in ("restless", "heavy") and mode in ("free", "guided"):
        reframe = "Instead of 'I have to handle all of it', try 'a few things matter tonight, and the rest can wait for morning.' Which one is tonight's?"
    return {"reflection": refl, "reframe": reframe}


NEGATIVE_SELF_TALK = re.compile(
    r"\b(i'?m (so )?(stupid|useless|a failure|worthless|not good enough|behind)|i always (mess|screw)|"
    r"i never (do|get) anything right|nothing ever works|i can'?t do anything|everyone (hates|is better))\b", re.I)

MODE_GUIDE = {
    "free": "Mirror the main thing they wrote about and how it seems to sit with them.",
    "guided": "Respond to how they answered the prompt; notice one specific detail in their answer.",
    "gratitude": "Savour what they listed: name what these good things say about what matters to them. No reframe.",
    "ritual": "Affirm the intention and make it feel doable tonight (one concrete, tiny next step). No reframe.",
}

BANNED = re.compile(
    r"(diagnos|disorder|\bdepression\b|\bptsd\b|\badhd\b|bipolar|medicat|prescri|\bcure\b|treatment|"
    r"as an ai language model|i am (a )?human|i'?m (a )?human|always be here for you|only need me|"
    r"you should|everything happens for a reason|i understand exactly)", re.I)


def wants_reframe(e: Dict[str, Any], family: str) -> bool:
    """Code decides, not the model: unpleasant mood or negative self-talk, and only in free/guided."""
    if e.get("mode") not in ("free", "guided"):
        return False
    return family in ("restless", "heavy") or bool(NEGATIVE_SELF_TALK.search(entry_text(e)))


def validate_reflection(out: Dict[str, Any], need_reframe: bool) -> Optional[str]:
    """Return a reason string if the model output must be replaced by the fallback."""
    text = str(out.get("reflection") or "")
    words = len(text.split())
    if words < 8 or words > 95:
        return "length"
    if BANNED.search(text) or (out.get("reframe") and BANNED.search(str(out["reframe"]))):
        return "banned_phrase"
    if text.count("?") > 1:
        return "too_many_questions"
    if need_reframe and out.get("reframe") and len(str(out["reframe"]).split()) > 60:
        return "reframe_length"
    return None


async def reflect(e: Dict[str, Any], word: str, family: str, allow_llm: bool) -> Dict[str, Any]:
    base = _fallback_reflection(e, word, family)
    need = wants_reframe(e, family)
    if not need:
        base["reframe"] = None
    themes = e.get("themes") or extract_themes(entry_text(e))
    base.update({"themes": themes, "energyCenter": ENERGY_CENTER_FOR_THEME.get(themes[0]) if themes else None,
                 "soundscapeCategory": _category_for(family)})
    if not allow_llm:
        return dict(base, _source="fallback")
    mode = e.get("mode") if e.get("mode") in MODE_GUIDE else "free"
    task = (
        "Write Lucille's reflection on one journal entry. "
        f"{MODE_GUIDE[mode]} "
        "'reflection': 2-3 sentences, 25-70 words, second person, ending somewhere steadier than where the entry ended. "
        + ("'reframe': the entry holds a harsh or stuck thought. Offer ONE gentler, still-true way to hold it, built "
           "from their own words (e.g. all-or-nothing -> one part of it; a prediction -> what is known right now; "
           "self-criticism -> how they'd speak to a friend). Max 45 words, no jargon or labels, end with one short "
           "question. "
           if need else "'reframe': null. ")
        + "'themes': up to 3 lowercase single words. "
        "'soundscapeCategory': one of music_meditations, vaporwave, jazz, nature, binaural_beats, brainwave_music, "
        "raw_frequencies, sleep_ambient, depression_anxiety, chosen to suit how they feel now. "
        "Return exactly these keys: reflection, reframe, themes, soundscapeCategory.\n"
        'Example (guided, mood Restless): {"reflection": "You named a lot pulling at you at once: the deadline, the '
        'inbox, the call you keep putting off. Writing it down already turned a blur into a short list.", '
        '"reframe": "Instead of \'I have to handle all of it tonight\', maybe \'one of these matters tonight, the rest '
        'has a place tomorrow\'. Which one is tonight\'s?", "themes": ["work"], "soundscapeCategory": "binaural_beats"}'
    )
    content = (f"Mode: {mode}\nMood word: {word}\nPrompt they answered: {e.get('prompt') or '-'}\n"
               f"<entry>\n{entry_text(e)[:2500]}\n</entry>")
    out = await llm.complete_json(task, content, base)
    if out.get("_source") == "llm":
        reason = validate_reflection(out, need)
        if reason:
            logger.info(f"reflection rejected ({reason}); using fallback")
            out = dict(base, _source="fallback", _rejected=reason)
    if not need:
        out["reframe"] = None
    if out.get("soundscapeCategory") not in {c["id"] for c in core.dataset("soundscape_catalog")["categories"]}:
        out["soundscapeCategory"] = base["soundscapeCategory"]
    if not isinstance(out.get("themes"), list):
        out["themes"] = themes
    out["themes"] = [str(t).lower()[:24] for t in out["themes"]][:3]
    return out


def _category_for(family: str) -> str:
    return {"restless": "binaural_beats", "heavy": "depression_anxiety", "calm": "nature",
            "tender": "music_meditations", "energized": "jazz", "mixed": "sleep_ambient"}.get(family, "nature")


def pick_track(category: str, word: str) -> Optional[Dict[str, Any]]:
    tracks = [t for t in core.dataset("soundscape_catalog")["tracks"] if t["category"] == category]
    if not tracks:
        return None
    w = word.lower()
    tracks.sort(key=lambda t: (w not in t["targetWords"], t["premium"]))
    t = tracks[0]
    return {"id": t["id"], "title": t["title"], "category": t["category"], "mode": t["mode"],
            "deeplink": core.resolve_route("SoundscapesNowPlaying", {"trackId": t["id"]})}


def iso_week(tz: str) -> str:
    y, w, _ = date.fromisoformat(core.local_day(tz)).isocalendar()
    return f"{y}-W{w:02d}"


async def weekly_reflection(repo: Repo, uid: str, tz: str, allow_llm: bool, generate: bool = True) -> Optional[Dict[str, Any]]:
    """One per ISO week, stored. Generated at most once per week per user."""
    wk = iso_week(tz)
    path = user_path(uid, "weekly", wk)
    doc = repo.get(path)
    if doc or not generate:
        return doc
    since = (date.fromisoformat(core.local_day(tz)) - timedelta(days=6)).isoformat()
    rows = [d for _, d in repo.query(user_path(uid, "journal_entries"), where=[("localDay", ">=", since)])
            if d.get("status") == "saved"]
    if len(rows) < 2:
        return None
    themes = Counter(t for r in rows for t in (r.get("themes") or []))
    words = Counter((r.get("mood") or {}).get("word") for r in rows if r.get("mood"))
    evenings = sum(1 for r in rows if r.get("localHour", 12) >= 18)
    top_theme = themes.most_common(1)[0][0] if themes else None
    top_word = words.most_common(1)[0][0] if words else "mixed"
    fallback = {
        "text": (f"You wrote {len(rows)} times this week"
                 + (f", mostly about {top_theme}" if top_theme else "")
                 + (", often in the evenings" if evenings > len(rows) / 2 else "")
                 + f". The feeling that came up most was {top_word.lower()}. Want to look back together?"),
        "themes": [t for t, _ in themes.most_common(4)],
    }
    out = fallback
    if allow_llm:
        snippets = "\n".join(f"<entry mode=\"{r.get('mode')}\" mood=\"{(r.get('mood') or {}).get('word', '?')}\">"
                             f"{entry_text(r)[:200]}</entry>" for r in rows[:7])
        task = ("Write Lucille's short weekly letter to the person, 40-70 words: the main themes, one gentle "
                "pattern you notice (time of day, what helped), one kind observation, and end with one inviting "
                "question. Do not quote anything sensitive. Keys: text, themes (up to 4 lowercase words).")
        out = await llm.complete_json(task, f"This week's entries:\n{snippets}", fallback)
        if out.get("_source") == "llm" and validate_reflection({"reflection": out.get("text")}, False):
            out = fallback
    doc = {"week": wk, "text": out["text"], "themes": out.get("themes", fallback["themes"]),
           "entries": len(rows), "topWord": top_word, "aiGenerated": out.get("_source") == "llm",
           "createdAt": core.iso(core.now_utc())}
    repo.set(path, doc)
    return doc
