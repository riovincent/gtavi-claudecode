"""Moment cues learned from real GTA 6 streamer clips (moment_signals.json, mirrored from
AutoClip Studio/library/moment-signals.json). Turns a transcript into per-second scores that analyze.py
adds to its loudness/hype/event/chat score.

Three signals:
  cue    - category phrases (EN/ID) in what was said, weighted per category
  karma  - a brag/prediction line followed by a loudness spike within 20 s ("instant karma")
  loop   - the same line repeated 3+ times within 40 s (NPC stuck on one sentence)
"""
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
# how much each category is worth when its cue is heard (story reactions are often spoilers -> lower)
WEIGHTS = {"instant_karma": 1.2, "npc_ai_chaos": 1.0, "physics_glitch": 1.0, "cop_wanted_escalation": 0.8,
           "heist_clutch_or_fail": 1.0, "life_sim_irony": 0.7, "story_reaction": 0.5, "chat_vs_streamer": 0.6,
           "irl_reaction": 1.2, "first_time_discovery": 0.8, "skill_flex": 0.9}
BRAG = ("instant_karma", "skill_flex")


def load(path=None):
    return json.loads(Path(path or HERE / "moment_signals.json").read_text(encoding="utf-8"))


def _rx(phrases):
    ps = [re.escape(p.strip().lower()) for p in phrases if p.strip()]
    return re.compile(r"(?<!\w)(" + "|".join(ps) + r")(?!\w)") if ps else None


def compile_cues(sig):
    out = {}
    for cat, c in sig["categories"].items():
        r = _rx(c.get("transcript_cues_en", []) + c.get("transcript_cues_id", []))
        if r:
            out[cat] = r
    return out


def _norm(text):
    return re.sub(r"[^\w ]", "", text.lower()).strip()


def per_second(segs, n, loud=None, sig=None):
    """segs: [{s, e, text}] -> (cue, karma, loop, hits): arrays of length n and {second: [category labels]}."""
    cues = compile_cues(sig or load())
    cue, karma, loop = np.zeros(n), np.zeros(n), np.zeros(n)
    hits = defaultdict(list)
    brag_at = []
    found = [(int(s["s"]), cat, m.lower()) for s in segs if int(s["s"]) < n
             for cat, r in cues.items() for m in r.findall(s["text"].lower())]
    # phrases a streamer says all the time ("chat", "bro", "hours") carry little signal: weight by rarity
    df = defaultdict(int)
    for _, _, m in found:
        df[m] += 1
    for i, cat, m in found:
        cue[i] += WEIGHTS.get(cat, 0.8) / (1 + np.log1p(df[m] / 5))
        hits[i].append(cat)
        if cat in BRAG:
            brag_at.append(i)
    # instant karma: brag/prediction, then something loud happens within 20 s
    if loud is not None:
        for i in brag_at:
            after = loud[i + 2:min(n, i + 21)]
            if after.size and after.max() > 1.5:
                j = i + 2 + int(after.argmax())
                karma[j] += 2.0
                hits[j].append("instant_karma*")
    # NPC loop: same normalised line 3+ times within 40 s
    seen = defaultdict(list)
    for s in segs:
        key = _norm(s["text"])
        if len(key.split()) >= 4 and len(set(key.split())) >= 3:  # a real sentence, not "go go go go"
            seen[key].append(int(s["s"]))
    for ts in seen.values():
        ts.sort()
        for a in range(len(ts) - 2):
            if ts[a + 2] - ts[a] <= 40 and ts[a + 2] < n:
                loop[ts[a + 2]] += 1.5
                hits[ts[a + 2]].append("npc_loop")
    return cue, karma, loop, hits
