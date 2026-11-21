"""Stage A: analyse a whole VOD on a GPU box without downloading the full-res video.

Audio-only download -> Whisper (auto language per chunk, word timestamps + confidence)
-> ECAPA voice embeddings per speech segment -> speaker clusters. In parallel: live-chat replay
(stops once it passes the analysed window or times out) and a low-res video for scene/webcam detection.
Scores every second (loudness, hype words, game-event cues, chat bursts) and writes ~40 candidate windows
with their transcript (per speaker) for a human/agent to pick from.

usage: python analyze.py <url> --out DIR [--max-minutes 0] [--profile profiles/x.json] [--chat-timeout 100]
writes DIR/: info.json transcript.json segments.json speakers.json scenes.json windows.json
"""
import argparse, json, os, re, subprocess, sys, threading, time, wave
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import load_json, run, save_json, tool, ytdlp_base, ytdlp_json
from moments import chat_series, smooth, z

HERE = Path(__file__).parent
SR = 16000
CHUNK = 600  # seconds per Whisper chunk (language is detected per chunk)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def lang_words():
    return load_json(HERE / "lang_words.json")


def rx(words):
    return re.compile(r"\b(" + "|".join(words) + r")\b", re.I) if words else None


# ---------- downloads ----------
def section_args(max_min):
    return ["--download-sections", f"*0-{max_min * 60}"] if max_min else []


def download_audio(url, d, max_min):
    wav = d / "audio.wav"
    if wav.exists():
        return wav
    run(ytdlp_base() + ["-f", "ba", *section_args(max_min), "-o", str(d / "audio_src.%(ext)s"), url])
    src = next(p for p in d.glob("audio_src.*") if p.suffix != ".part")
    run([tool("ffmpeg"), "-y", "-v", "error", "-i", str(src), "-ac", "1", "-ar", str(SR), str(wav)])
    src.unlink()
    return wav


def download_lowres(url, d, max_min):
    out = d / "lowres.mp4"
    if not out.exists():
        run(ytdlp_base() + ["-f", "bv*[height<=360][ext=mp4]/bv*[height<=360]/wv", *section_args(max_min),
                            "--merge-output-format", "mp4", "-o", str(out), url], check=False)
    return out if out.exists() else None


def chat_last_offset(path):
    try:
        with open(path, "rb") as f:
            f.seek(max(0, path.stat().st_size - 400_000))
            tail = f.read().decode("utf-8", "ignore")
        m = re.findall(r'"videoOffsetTimeMsec":\s*"(\d+)"', tail)
        return int(m[-1]) / 1000 if m else 0
    except OSError:
        return 0


def download_chat(url, d, window_s, timeout_min):
    """Runs in a thread. Stops yt-dlp once chat passes the window or the timeout hits; keeps what arrived."""
    p = subprocess.Popen(ytdlp_base() + ["--skip-download", "--write-subs", "--sub-langs", "live_chat",
                                         "-o", str(d / "chat"), url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0, part, done = time.time(), d / "chat.live_chat.json.part", d / "chat.live_chat.json"
    while p.poll() is None:
        time.sleep(20)
        f = part if part.exists() else done
        off = chat_last_offset(f) if f.exists() else 0
        if (window_s and off > window_s + 60) or time.time() - t0 > timeout_min * 60:
            p.terminate()
            break
    time.sleep(2)
    if not done.exists() and part.exists():
        part.replace(d / "chat-partial.live_chat.json")
    f = next(iter(d.glob("chat*.live_chat.json")), None)
    log(f"chat: {'none' if not f else f'{f.stat().st_size / 2**20:.0f} MB up to {chat_last_offset(f) / 3600:.1f} h'}")


# ---------- audio ----------
def read_pcm(wav, a, b):
    with wave.open(str(wav)) as w:
        w.setpos(int(a * SR))
        return np.frombuffer(w.readframes(int((b - a) * SR)), dtype=np.int16).astype(np.float32) / 32768


def wav_seconds(wav):
    with wave.open(str(wav)) as w:
        return w.getnframes() / SR


def loudness(wav, n):
    db = np.full(n, -90.0)
    for a in range(0, n, 3600):
        x = read_pcm(wav, a, min(n, a + 3600))
        k = len(x) // SR
        if k:
            db[a:a + k] = 20 * np.log10(np.sqrt((x[:k * SR].reshape(k, SR) ** 2).mean(1) + 1e-9))
    return db


class Whisper:
    def __init__(self, fixed_lang):
        from faster_whisper import WhisperModel
        import ctranslate2
        self.dev = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        self.ct = "float16" if self.dev == "cuda" else "int8"
        self.WM, self.models, self.fixed = WhisperModel, {}, None if fixed_lang in (None, "", "auto") else fixed_lang

    def model(self, name):
        if name not in self.models:
            log(f"loading whisper {name} on {self.dev}")
            self.models[name] = self.WM(name, device=self.dev, compute_type=self.ct)
        return self.models[name]

    def lang(self, pcm):
        if self.fixed:
            return self.fixed
        try:
            lang, prob, _ = self.model("small").detect_language(pcm[: SR * 30 * 3], language_detection_segments=3)
            return lang
        except Exception:
            _, info = self.model("small").transcribe(pcm[: SR * 30], beam_size=1)
            return info.language

    def chunk(self, pcm, offset):
        try:
            return self._chunk(pcm, offset)
        except RuntimeError as ex:  # CUDA libs missing on the box: fall back to CPU once, keep going
            if self.dev == "cpu":
                raise
            log(f"GPU whisper failed ({str(ex)[:80]}); switching to CPU")
            self.dev, self.ct, self.models = "cpu", "int8", {}
            return self._chunk(pcm, offset)

    def _chunk(self, pcm, offset):
        lang = self.lang(pcm)
        m = self.model("small" if lang == "en" else "large-v3-turbo")
        segs, _ = m.transcribe(pcm, language=lang, word_timestamps=True, vad_filter=True)
        segs = list(segs)  # run the generator here so GPU errors surface inside the try
        out = []
        for s in segs:
            ws = [{"w": w.word.strip(), "s": round(offset + w.start, 2), "e": round(offset + w.end, 2),
                   "p": round(w.probability, 2), "lang": lang} for w in (s.words or []) if w.word.strip()]
            if ws:
                out.append({"s": round(offset + s.start, 2), "e": round(offset + s.end, 2), "text": s.text.strip(),
                            "lang": lang, "words": ws})
        return out, lang


# ---------- speakers ----------
def speaker_clusters(wav, segs):
    """ECAPA embedding per segment (>=1 s), cluster a sample, assign all to centroids. Sets seg['spk'] (-1 = unknown)."""
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", savedir="/kaggle/temp/ecapa" if Path("/kaggle").exists() else None,
                                             run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"})
    except Exception as ex:
        log(f"speaker model unavailable ({str(ex)[:120]}); all speakers unknown")
        for s in segs:
            s["spk"] = -1
        return {}
    idx = [i for i, s in enumerate(segs) if s["e"] - s["s"] >= 1.0]
    embs = np.zeros((len(idx), 192), np.float32)
    batch, rows = [], []
    def flush():
        L = max(len(x) for x in batch)
        t = torch.zeros(len(batch), L)
        lens = torch.tensor([len(x) / L for x in batch])
        for k, x in enumerate(batch):
            t[k, :len(x)] = torch.from_numpy(x)
        with torch.no_grad():
            e = enc.encode_batch(t, lens).squeeze(1).cpu().numpy()
        embs[rows] = e
        batch.clear(); rows.clear()
    for j, i in enumerate(idx):
        s = segs[i]
        mid = (s["s"] + s["e"]) / 2
        a, b = max(s["s"], mid - 3), min(s["e"], mid + 3)
        batch.append(read_pcm(wav, a, b)); rows.append(j)
        if len(batch) == 64:
            flush()
    if batch:
        flush()
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    from sklearn.cluster import AgglomerativeClustering
    rng = np.random.default_rng(0)
    samp = rng.choice(len(idx), size=min(3000, len(idx)), replace=False) if len(idx) else np.array([], int)
    if len(samp) < 10:
        for s in segs:
            s["spk"] = -1
        return {}
    lab = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=0.6).fit_predict(embs[samp])
    cents = []
    for c, cnt in Counter(lab).items():
        if cnt >= max(10, 0.01 * len(samp)):
            v = embs[samp][lab == c].mean(0)
            cents.append(v / (np.linalg.norm(v) + 1e-9))
    C = np.stack(cents) if cents else np.zeros((0, 192))
    sims = embs @ C.T if len(C) else np.zeros((len(idx), 0))
    raw = np.where(sims.max(1) >= 0.35, sims.argmax(1), -1) if len(C) else np.full(len(idx), -1)
    talk = Counter()
    for j, i in enumerate(idx):
        talk[int(raw[j])] += segs[i]["e"] - segs[i]["s"]
    order = [c for c, _ in talk.most_common() if c != -1]
    remap = {c: k for k, c in enumerate(order)}
    for s in segs:
        s["spk"] = -1
    for j, i in enumerate(idx):
        segs[i]["spk"] = remap.get(int(raw[j]), -1)
    return {k: talk[c] for c, k in remap.items()}


# ---------- scenes ----------
def _scene(args):
    video, t = args
    from facecam import find_cam
    try:
        r = find_cam(str(video), t, t + 12)
    except Exception as ex:
        return {"t": t, "mode": "error", "reason": str(ex)[:80]}
    mode = r.get("mode")
    if mode == "none" and r.get("face") and r["face"][2] > 0.15:
        mode = "fullcam"  # big talking head, no game
    return {"t": t, "mode": mode, "box": r.get("box"), "face": r.get("face")}


def detect_scenes(video, dur, step=90):
    from multiprocessing import Pool
    ts = [(video, t) for t in range(30, int(dur) - 15, step)]
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        out = pool.map(_scene, ts, chunksize=4)
    boxes = Counter(tuple(round(v, 2) for v in s["box"]) for s in out if s["mode"] == "cam" and s.get("box"))
    cam = list(boxes.most_common(1)[0][0]) if boxes else None
    return {"step": step, "cam_box": cam, "cam_votes": boxes.most_common(1)[0][1] if boxes else 0, "samples": out}


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=int, default=0, help="0 = whole VOD")
    ap.add_argument("--profile"); ap.add_argument("--chat-timeout", type=int, default=100)
    ap.add_argument("--windows", type=int, default=40)
    a = ap.parse_args()
    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
    prof = load_json(a.profile) if a.profile else {}
    info = ytdlp_json(a.url)
    meta = {k: info.get(k) for k in ("id", "title", "channel", "uploader_id", "webpage_url", "duration", "upload_date", "was_live")}
    save_json(d / "info.json", meta)
    window_s = a.max_minutes * 60 if a.max_minutes else 0
    log(f"{meta['title']} | {meta['duration']} s | window {'all' if not window_s else f'{a.max_minutes} min'}")

    th = []
    if info.get("was_live"):
        th.append(threading.Thread(target=download_chat, args=(a.url, d, window_s or meta["duration"], a.chat_timeout)))
    lowres = {}
    th.append(threading.Thread(target=lambda: lowres.update(v=download_lowres(a.url, d, a.max_minutes))))
    for t in th:
        t.start()

    wav = download_audio(a.url, d, a.max_minutes)
    n = int(wav_seconds(wav))
    log(f"audio ready: {n / 3600:.2f} h")
    wh = Whisper(prof.get("lang", "auto"))
    segs, langs = [], Counter()
    for c0 in range(0, n, CHUNK):
        part, lang = wh.chunk(read_pcm(wav, c0, min(n, c0 + CHUNK)), c0)
        segs += part; langs[lang] += 1
        if c0 // CHUNK % 6 == 0:
            log(f"transcribed {min(n, c0 + CHUNK) / 3600:.2f} h ({lang}), {len(segs)} segments")
    talk = speaker_clusters(wav, segs)
    log(f"speakers: {len(talk)} clusters, talk time {[(k, round(v / 60)) for k, v in sorted(talk.items())][:8]} min")
    words = [dict(w, spk=s["spk"]) for s in segs for w in s["words"]]
    save_json(d / "transcript.json", words)
    save_json(d / "segments.json", [{k: s[k] for k in ("s", "e", "text", "lang", "spk")} for s in segs])

    for t in th:
        t.join()
    scenes = detect_scenes(lowres["v"], n) if lowres.get("v") else {"cam_box": None, "samples": []}
    save_json(d / "scenes.json", scenes)
    log(f"scenes: cam box {scenes.get('cam_box')} ({scenes.get('cam_votes', 0)} votes)")

    # signals per second
    lw = lang_words()
    hype_rx = {L: rx(v.get("hype", [])) for L, v in lw.items()}
    event_rx = {L: rx(v.get("event", [])) for L, v in lw.items()}
    db = loudness(wav, n)
    loud = smooth(np.clip(z(db - smooth(db, 30)), 0, None), 3)
    hype, event = np.zeros(n), np.zeros(n)
    for w in words:
        i = int(w["s"])
        if i >= n:
            continue
        hr, er = hype_rx.get(w["lang"]) or hype_rx["en"], event_rx.get(w["lang"]) or event_rx["en"]
        if hr and hr.search(w["w"]): hype[i] += 1
        if er and er.search(w["w"]): event[i] += 1
    chat = chat_series(d, n, None, info)
    burst = np.convolve(chat, np.ones(15) / 15, "same") / (np.convolve(chat, np.ones(600) / 600, "same") + 1)
    chatz = np.clip(z(burst), 0, None) if chat.any() else np.zeros(n)
    score = loud + 0.8 * z(smooth(hype, 5)).clip(0) + 1.2 * z(smooth(event, 8)).clip(0) + 1.5 * chatz

    # chat text per second for context
    chat_txt = {}
    f = next(iter(d.glob("chat*.live_chat.json")), None)
    if f:
        for line in open(f, encoding="utf-8", errors="ignore"):
            try:
                act = json.loads(line)["replayChatItemAction"]
                t = int(int(act["videoOffsetTimeMsec"]) / 1000)
                r = act["actions"][0]["addChatItemAction"]["item"]["liveChatTextMessageRenderer"]
                chat_txt.setdefault(t, []).append("".join(x.get("text", "") for x in r["message"]["runs"]))
            except Exception:
                continue
    samples = scenes.get("samples", [])
    def scene_at(t):
        return min(samples, key=lambda s: abs(s["t"] - t))["mode"] if samples else "unknown"

    wins, used = [], np.zeros(n, bool)
    for p in np.argsort(-score):
        if len(wins) >= a.windows or score[p] <= 0:
            break
        s0, e0 = max(0, p - 45), min(n, p + 25)
        if used[s0:e0].any():
            continue
        used[max(0, s0 - 30):min(n, e0 + 30)] = True
        inwin = [s for s in segs if s["e"] > s0 and s["s"] < e0]
        if inwin:
            s0, e0 = max(0, int(inwin[0]["s"])), min(n, int(np.ceil(inwin[-1]["e"])) + 1)
        cw = Counter(w.upper() for t in range(p - 10, p + 20) for m in chat_txt.get(t, []) for w in re.findall(r"\w+", m))
        wins.append({"peak": int(p), "start": s0, "end": e0, "score": round(float(score[p]), 2),
                     "signals": {"loud": round(float(loud[p]), 2), "hype": int(hype[max(0, p - 5):p + 5].sum()),
                                 "event": int(event[max(0, p - 15):p + 15].sum()), "chat_burst": round(float(burst[p]), 2)},
                     "scene": scene_at(p), "chat_top": [w for w, _ in cw.most_common(8)],
                     "lines": [f"[{s['s']:.1f} spk{s['spk']}] {s['text']}" for s in inwin]})
    wins.sort(key=lambda w: w["start"])
    save_json(d / "windows.json", wins)

    spk = {}
    for s in segs:
        k = s["spk"]
        x = spk.setdefault(k, {"talk_s": 0.0, "segments": 0, "langs": Counter(), "samples": []})
        x["talk_s"] += s["e"] - s["s"]; x["segments"] += 1; x["langs"][s["lang"]] += 1
        x["samples"].append((s["e"] - s["s"], s["s"], s["text"]))
    for k, x in spk.items():
        x["talk_s"] = round(x["talk_s"]); x["langs"] = dict(x["langs"])
        top = sorted(x["samples"], reverse=True)[:6] + x["samples"][:: max(1, len(x["samples"]) // 8)][:8]
        x["samples"] = [f"[{t:.0f}s] {txt}" for _, t, txt in top]
    save_json(d / "speakers.json", {str(k): v for k, v in sorted(spk.items())})
    for p in (d / "audio.wav", d / "lowres.mp4"):
        p.unlink(missing_ok=True)
    log(f"done: {len(words)} words, langs {dict(langs)}, {len(wins)} windows -> {d}")


if __name__ == "__main__":
    main()
