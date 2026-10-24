# Datasets behind the v1 endpoints

The v1 API answers most questions from small curated files instead of LLM calls. That keeps
cost near zero and makes behaviour predictable and testable.

## Shipped in `escape_api/data/` (Escape-authored, no third-party license)

| File | Used by | Contents |
|---|---|---|
| `mood_words.json` | mood fusion, orb tone, stats | 36 mood words with valence/energy coordinates (circumplex model of affect), families, tone → orb mapping, keywords |
| `journal_prompts.json` | `/v1/journal/prompt` | Prompts per mode (Free, Guided, Gratitude, Ritual Spark) × orb tone; letter prompts |
| `selfcare_activities.json` | suggestions, plans, score next step | 29 activities, each mapped to a real FlutterFlow route |
| `deeplinks.json` | every suggestion | Route allowlist (live / planned + fallback) |
| `soundscape_catalog.json` | `/v1/soundscapes/*` | 9 categories, 39 tracks: mode, Mood Field position, bpm, key, brainwave, Hz, target moods, segment paths |
| `mood_keywords_derived.json` | voice/text mood read (fallback pass) | 323 keywords for 18 moods, derived from GoEmotions by `scripts/datasets/build_mood_keywords.py` |

## External datasets we looked at

| Dataset | What it adds | License note | Status |
|---|---|---|---|
| **GoEmotions** (Google Research, 58k Reddit comments, 27 emotions) | Keyword lexicon for free-text mood reads | Published in google-research/goemotions; check the dataset card before redistributing raw data. We ship only derived word lists. | Used (derived keywords) |
| **NRC Valence-Arousal-Dominance Lexicon** (Saif Mohammad, ~55k words) | Finer valence/energy scores per word | Free for research; commercial use needs a license from NRC Canada | Not shipped. Worth licensing if text mood reads need to get sharper |
| **Meta "PatternReframe"** (Maddela et al., ACL 2023: unhelpful thoughts + reframes) | Few-shot examples for Lucille's reframes | Research release; confirm terms before production use | Reference only; good material for the ML team's prompt tests |
| **Freesound** (CC0 / CC-BY / CC-BY-NC per sound) | Nature beds (rain, ocean, fire) for the Nature category | Per-sound license; use CC0 for anything shipped | Candidate source for Lucille's ambience layers |
| **FMA: Free Music Archive** (106k tracks, genre tags incl. jazz/ambient) | Reference tags and training audio for Jazz / Ambient / Vaporwave styles | Per-track Creative Commons; metadata CC BY 4.0 | Candidate for the ML team's style references, not for shipping |

Rebuild the derived keywords:

```bash
# download train.tsv + emotions.txt from github.com/google-research/google-research/tree/master/goemotions/data
python scripts/datasets/build_mood_keywords.py --data /path/to/goemotions/train.tsv
```
