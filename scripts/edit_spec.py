"""Stage C: render approved picks as edited vertical clips (short + long cut per pick).

Edit grammar: [teaser: payoff first + hook text] -> white flash + whoosh -> setup ... payoff (one slow-mo
or freeze) -> reaction + punchline -> hard cut. Subtitles only for labelled speakers, in their colour.
Only the pick's section is downloaded (1080p), never the whole VOD.

usage: python edit_spec.py picks.json --out DIR [--profile profiles/x.json]
picks.json: [{"id", "url", "section": [s, e], "layout": "facecam|fullcam", "cam": "x:y:w:h"|null,
              "speakers": {"0": "#FFE500"}, "words": [{w,s,e,p,spk}], "hook", "punchline", "caption",
              "cuts": {"short": {"teaser": [a, b], "keep": [[a, b], ...], "effect": {...}, "punch_at": t}, "long": {...}}}]
All times are VOD seconds. effect: {"type": "slowmo", "start": a, "end": b, "factor": 0.4} | {"type": "freeze", "at": t, "dur": 0.7}
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import ffprobe, load_json, run, save_json, tool, ytdlp_base
from cut import CAM_H, CAP_Y, H, W, build_ass, esc, ts, video_filter

STYLES = (
    "Style: Hook,Arial Black,74,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,8,2,5,50,50,0,1\n"
    "Style: Punch,Arial Black,80,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,8,2,5,50,50,0,1\n"
    "Style: Credit,Arial,36,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,3,1,5,40,40,0,1\n")
SEG_V = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30"]
SEG_A = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]


def text_y(layout):
    # all burned text sits in the gameplay area, never on the facecam panel
    if layout == "facecam":
        return {"hook": CAM_H + 110, "punch": 1560, "credit": H - 60}
    return {"hook": 1700, "punch": 1250, "credit": H - 60}


def wrap(s, n=20):
    out, cur = [], ""
    for w in s.split():
        if cur and len(cur) + 1 + len(w) > n:
            out.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    return "\\N".join(out + ([cur] if cur else []))


def seg_ass(path, words, a, b, layout, speakers, extra):
    body = build_ass(words, a, b, "", CAP_Y[layout], speakers=speakers, min_p=0.5)
    body = body.replace("\n[Events]", STYLES + "\n[Events]", 1)
    Path(path).write_text(body + "\n".join(extra) + "\n", encoding="utf-8")


def whoosh(d):
    f = d / "whoosh.wav"
    if not f.exists():
        run([tool("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i",
             "anoisesrc=d=0.5:c=pink:a=0.5,highpass=f=500,lowpass=f=7000,afade=t=in:d=0.3,afade=t=out:st=0.3:d=0.2",
             "-ar", "48000", "-ac", "2", str(f)])
    return f


def render_seg(src, off, a, b, out, layout, cam, ass, kind="plain", factor=1.0, flash=False, wh=None):
    """One output segment from source seconds [a, b] (VOD time; off = VOD time of the section start)."""
    vf = video_filter(layout, cam, ass)
    v_out, inputs = "[vo]", ["-ss", f"{a - off:.3f}", "-t", f"{b - a:.3f}", "-i", str(src)]
    if kind == "slowmo":
        vf += f";[vo]setpts=PTS/{factor}[vs]"; v_out = "[vs]"
    if kind == "freeze":
        vf += f";[vo]trim=end_frame=1,tpad=stop_mode=clone:stop_duration={b - a:.3f}[vs]"; v_out = "[vs]"
    if flash:
        vf += f";{v_out}fade=t=in:st=0:d=0.3:color=white[vf]"; v_out = "[vf]"
    tempo = ""
    if kind == "slowmo":
        tempo = ",atempo=0.5" + (f",atempo={factor / 0.5:.3f}" if abs(factor - 0.5) > 1e-3 else "")
    af = f"[0:a]aresample=48000,aformat=channel_layouts=stereo{tempo}[a0]"
    a_out = "[a0]"
    if flash and wh:
        inputs += ["-i", str(wh)]
        af += ";[1:a]volume=0.9[w];[a0][w]amix=inputs=2:duration=first:normalize=0[am]"; a_out = "[am]"
    run([tool("ffmpeg"), "-y", "-v", "error", *inputs, "-filter_complex", vf + ";" + af,
         "-map", v_out, "-map", a_out, *SEG_V, *SEG_A, str(out)])


def plan_segments(cut):
    """-> list of (kind, a, b, factor): teaser, then keep ranges with the effect spliced in."""
    segs = []
    if cut.get("teaser"):
        segs.append(("teaser", *cut["teaser"], 1.0))
    eff = cut.get("effect") or {}
    for a, b in cut["keep"]:
        if eff.get("type") == "slowmo" and a <= eff["start"] < eff["end"] <= b:
            segs += [("plain", a, eff["start"], 1.0), ("slowmo", eff["start"], eff["end"], eff.get("factor", 0.4)),
                     ("plain", eff["end"], b, 1.0)]
        elif eff.get("type") == "freeze" and a <= eff["at"] <= b:
            segs += [("plain", a, eff["at"], 1.0), ("freeze", eff["at"], eff["at"] + eff.get("dur", 0.7), 1.0),
                     ("plain", eff["at"], b, 1.0)]
        else:
            segs.append(("plain", a, b, 1.0))
    return [s for s in segs if s[2] - s[1] > 0.05]


def render_cut(p, cut, src, off, layout, cam, d, name, credit):
    y = text_y(layout)
    wh = whoosh(d)
    parts, first_main = [], True
    for i, (kind, a, b, f) in enumerate(plan_segments(cut)):
        dur = (b - a) / (f if kind == "slowmo" else 1)
        extra = [f"Dialogue: 1,{ts(0)},{ts(dur)},Credit,,0,0,0,,{{\\pos({W // 2},{y['credit']})}}{esc(credit)}"]
        if kind == "teaser" or (first_main and not cut.get("teaser") and kind != "teaser"):
            extra.append(f"Dialogue: 2,{ts(0)},{ts(min(dur, 3.5))},Hook,,0,0,0,,{{\\pos({W // 2},{y['hook']})}}{wrap(esc(p['hook'].upper()))}")
        pa = cut.get("punch_at")
        if p.get("punchline") and pa is not None and a <= pa < b and kind != "teaser":
            st = (pa - a) / (f if kind == "slowmo" else 1)
            extra.append(f"Dialogue: 2,{ts(st)},{ts(dur)},Punch,,0,0,0,,{{\\pos({W // 2},{y['punch']})\\fad(80,0)}}{wrap(esc(p['punchline'].upper()))}")
        ass = d / f"{name}_{i}.ass"
        words = [] if kind in ("slowmo", "freeze") else p.get("words", [])
        seg_ass(ass, words, a, b, layout, p.get("speakers") or {}, extra)
        out = d / f"{name}_{i}.mp4"
        flash = kind != "teaser" and first_main and bool(cut.get("teaser"))
        render_seg(src, off, a, b, out, layout, cam, ass, kind=kind, factor=f, flash=flash, wh=wh)
        if kind != "teaser":
            first_main = False
        parts.append(out)
    lst = d / f"{name}_list.txt"
    lst.write_text("".join(f"file '{x.resolve().as_posix()}'\n" for x in parts), encoding="utf-8")
    cat, final = d / f"{name}_cat.mp4", d / f"{name}.mp4"
    run([tool("ffmpeg"), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(cat)])
    run([tool("ffmpeg"), "-y", "-v", "error", "-i", str(cat), "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
         *SEG_A, "-movflags", "+faststart", str(final)])
    for x in parts + [cat, lst] + list(d.glob(f"{name}_*.ass")):
        x.unlink(missing_ok=True)
    return final


def contact_sheet(mp4, teaser_len, out):
    dur = float(ffprobe(mp4)["format"]["duration"])
    pts = [0.5, teaser_len + 0.15 if teaser_len else 1.5, dur * 0.75, max(0, dur - 0.3)]
    tiles = []
    for k, t in enumerate(pts):
        f = out.parent / f"_s{k}.jpg"
        run([tool("ffmpeg"), "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(mp4), "-frames:v", "1", "-vf", "scale=270:-2", str(f)])
        tiles.append(f)
    run([tool("ffmpeg"), "-y", "-v", "error", *sum([["-i", str(f)] for f in tiles], []),
         "-filter_complex", "".join(f"[{i}]" for i in range(len(tiles))) + f"hstack=inputs={len(tiles)}", str(out)])
    for f in tiles:
        f.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks"); ap.add_argument("--out", required=True); ap.add_argument("--profile")
    ap.add_argument("--inputs", help="folder with pre-cut sections (pick['section_file'])")
    a = ap.parse_args()
    prof = load_json(a.profile) if a.profile else {}
    picks = load_json(a.picks)
    root = Path(a.out); root.mkdir(parents=True, exist_ok=True)
    for p in picks:
        d = root / p["id"]; d.mkdir(exist_ok=True)
        s, e = p["section"]
        src = d / "section.mp4"
        if p.get("section_file") and a.inputs:  # section already cut (laptop download): starts at s
            import shutil
            shutil.copy2(Path(a.inputs) / p["section_file"], src)
        elif p.get("src"):  # local file (tests): cut the section out of it
            run([tool("ffmpeg"), "-y", "-v", "error", "-ss", str(s), "-to", str(e), "-i", p["src"], "-c:v", "libx264",
                 "-preset", "veryfast", "-crf", "16", "-c:a", "aac", str(src)])
        if not src.exists():
            run(ytdlp_base() + ["-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b", "--merge-output-format", "mp4",
                                "--download-sections", f"*{s}-{e}", "--force-keyframes-at-cuts", "-o", str(src), p["url"]])
        layout = p.get("layout") or prof.get("layout", "facecam")
        cam = p.get("cam") or prof.get("cam")
        if cam and layout == "facecam" and not p.get("layout"):
            # pinned webcam box for game scenes; a full-camera scene (no game, big face) gets the fullcam layout
            from facecam import find_cam
            r = find_cam(str(src))
            if r.get("mode") != "cam" and r.get("face") and r["face"][2] > 0.15:
                layout, cam = "fullcam", ":".join(f"{v:.4f}" for v in r["face"])
        if not cam:
            from facecam import find_cam
            r = find_cam(str(src))
            box = r.get("box") or r.get("face")
            if r.get("mode") == "cam":
                cam = ":".join(f"{v:.4f}" for v in box)
            elif box:
                layout, cam = "fullcam", ":".join(f"{v:.4f}" for v in box)
            else:
                layout, cam = "blur", None
        if layout == "facecam" and not cam:
            layout = "blur"
        credit = p.get("credit") or prof.get("credit", "")
        payoff = (p["cuts"].get("short") or {}).get("punch_at") or (s + e) / 2
        run([tool("ffmpeg"), "-y", "-v", "error", "-ss", f"{payoff - s:.2f}", "-i", str(src), "-frames:v", "1", str(d / "proof.jpg")])
        for name, cut in p["cuts"].items():
            mp4 = render_cut(p, cut, src, s, layout, cam, d, name, credit)
            tl = (cut["teaser"][1] - cut["teaser"][0]) if cut.get("teaser") else 0
            contact_sheet(mp4, tl, d / f"{name}_sheet.jpg")
            print(f"{p['id']}/{name}: {float(ffprobe(mp4)['format']['duration']):.1f}s ({layout}, cam {cam})", flush=True)
        tags = " ".join(prof.get("hashtags", [])[:5])
        (d / "caption.txt").write_text(f"{p.get('caption', '')}\n\nCredit: {credit}\n\n{tags}\n", encoding="utf-8")
        src.unlink(missing_ok=True)
    save_json(root / "picks_rendered.json", [{"id": p["id"], "hook": p.get("hook"), "punchline": p.get("punchline")} for p in picks])


if __name__ == "__main__":
    main()
