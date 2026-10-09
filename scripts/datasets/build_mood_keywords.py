"""
Build escape_api/data/mood_keywords_derived.json from the GoEmotions dataset.

Why: the v1 API reads mood from voice transcripts / typed words with a keyword
lexicon (zero LLM cost). The hand-written keywords in mood_words.json cover the
obvious words; this script adds a few hundred more, learned from 43k labelled
comments, so 'Lucille, today was rough and I'm just over it' still lands on a mood.

Only derived word lists are shipped (no comment text). GoEmotions is published by
Google Research in the google-research repo; check its dataset card license before
any commercial redistribution of the raw data.

    python scripts/datasets/build_mood_keywords.py --data path/to/goemotions/train.tsv
    (download train.tsv + emotions.txt from
     https://github.com/google-research/google-research/tree/master/goemotions/data)
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

# GoEmotions label -> Escape mood word (escape_api/data/mood_words.json)
LABEL_TO_WORD = {
    "admiration": "Inspired", "amusement": "Joyful", "anger": "Angry", "annoyance": "Frustrated",
    "approval": "Content", "caring": "Tender", "confusion": "Uncertain", "curiosity": "Reflective",
    "desire": "Hopeful", "disappointment": "Disappointed", "disapproval": "Frustrated", "disgust": "Angry",
    "embarrassment": "Tense", "excitement": "Excited", "fear": "Anxious", "gratitude": "Grateful",
    "grief": "Sad", "joy": "Joyful", "love": "Loved", "nervousness": "Anxious", "optimism": "Hopeful",
    "pride": "Proud", "realization": "Reflective", "relief": "Relaxed", "remorse": "Disappointed",
    "sadness": "Sad", "surprise": "Excited",
}
STOP = set("""a an the and or but if then so to of in on at by for with about from as is am are was were be been being
i me my mine we us our you your he she it they them their this that these those there here what which who whom
do does did doing have has had having not no yes just very really too also can could would should will shall may
might must get got im i'm it's its dont don't cant can't wont won't ive i've id i'd youre you're thats that's
one all any some more most much many lot lots like than up down out over again only own same other such into
how why when where now time thing things way day name people person guy guys""".split())
BAD = re.compile(r"(fuck|shit|bitch|dick|cunt|ass\b|damn|crap|piss|bastard|slut|whore|retard|fag|nigg|rape)")
TOKEN = re.compile(r"[a-z][a-z']{2,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="GoEmotions train.tsv")
    ap.add_argument("--emotions", default=None, help="emotions.txt (defaults next to train.tsv)")
    ap.add_argument("--top", type=int, default=18)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "escape_api", "data",
                                                   "mood_keywords_derived.json"))
    a = ap.parse_args()
    labels = open(a.emotions or os.path.join(os.path.dirname(a.data), "emotions.txt")).read().split()
    per = defaultdict(Counter)
    total = Counter()
    docs = 0
    for line in open(a.data, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        text, ids = parts[0].lower(), [int(i) for i in parts[1].split(",")]
        toks = {t for t in TOKEN.findall(text) if t not in STOP and not BAD.search(t) and "[" not in t}
        docs += 1
        total.update(toks)
        for i in ids:
            if labels[i] in LABEL_TO_WORD:
                per[labels[i]].update(toks)
    out = defaultdict(set)
    for lab, cnt in per.items():
        n_lab = sum(cnt.values())
        n_all = sum(total.values())
        scored = []
        for tok, c in cnt.items():
            if c < 8:
                continue
            rest = total[tok] - c
            lo = math.log((c + 1) / (n_lab - c + 1)) - math.log((rest + 1) / (n_all - n_lab - rest + 1))
            scored.append((lo, tok))
        for lo, tok in sorted(scored, reverse=True)[: a.top]:
            if lo > 1.0:
                out[LABEL_TO_WORD[lab]].add(tok)
    # a token that lands on two moods is ambiguous; keep it on neither
    seen = Counter(t for ws in out.values() for t in ws)
    result = {w: sorted(t for t in ws if seen[t] == 1) for w, ws in sorted(out.items())}
    meta = {"_about": "Derived keyword lists (log-odds, GoEmotions train split). Used by core.interpret_text at "
                      "half weight behind the hand-written keywords. Rebuild with scripts/datasets/build_mood_keywords.py.",
            "source": "GoEmotions (Demszky et al., 2020), google-research/goemotions", "documents": docs}
    json.dump({**meta, "words": result}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {sum(len(v) for v in result.values())} keywords for {len(result)} moods from {docs} comments")


if __name__ == "__main__":
    main()
