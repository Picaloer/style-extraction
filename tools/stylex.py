#!/usr/bin/env python3
"""stylex 2026-10-13.4: tools for distilling a visual style from videos / images into an Amber-Diorama (琥珀微缩) style document.

Subcommands (run `stylex.py <cmd> -h` for options):
  probe    inventory inputs, assign codes A, B, C...          -> W/inputs.json
  sample   sample frames per video, detect shots, contact sheets, before|after pairs of plain cuts -> W/frames/, W/sheets/
  overlay  find static watermarks / overlays, suggest crop boxes  -> W/overlay.json, W/overlay/*.jpg
  crop     set crop boxes by hand (C=x0,y0,x1,y1), overriding overlay
  grab     extract full-resolution frames by id (C_140.0) or time (C@2:20)
  zoom     view a frame / region with a pixel-coordinate grid (to read boxes; nearest-neighbour enlargement)
  sheet    contact sheet from arbitrary images (layout and pages chosen from their number)
  measure  colour / light / lens measurements on frames          -> W/measure.json
  profile  sharpness profile + vignette averaged over a time range -> W/measure.json
  pixel    estimate pixel-art scale (screen px per art px) in a region
  pick     median colour of small named patches (objects, skin, signs, UI); draws the boxes  -> W/measure.json, W/zoom/
  drop     remove entries (a bad pick, a trial measurement) from measure.json
  light    falloff of a light pool or glow around a point                 -> W/measure.json
  post     lateral chromatic aberration, map of channel splits, grain      -> W/measure.json
  audio    loudness, segments, tempo / key / spectrum per input    -> W/audio/
  facts    every measured number on one screen; with content.json, which colours are not measured
  crops    plates and specimens for the page                      -> W/img/
  build    content.json -> self-contained HTML in the Amber form
  pdf      HTML -> PDF (paged, dark background kept; empty pages removed; lists text drawn with system fonts)
  shots    QA screenshots of the HTML and PDF pages
Only numpy + Pillow + ffmpeg are required; Playwright or Chromium is needed for pdf / shots.
"""
import argparse, base64, glob, html, html.parser, io, json, math, os, re, shutil, subprocess, sys, tempfile, urllib.parse, urllib.request, warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VERSION = "2026-10-13.4"
HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".flv", ".m4v", ".ts", ".wmv", ".mpg", ".mpeg", ".gif"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


# ───────────────────────── helpers ─────────────────────────
def die(msg):
    print("error: " + msg, file=sys.stderr)
    sys.exit(1)


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, **kw)


def jload(p, default=None):
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return default


def jsave(p, obj):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def mmss(t):
    t = max(0.0, float(t))
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}" if abs(s - round(s)) > 0.05 else f"{int(m)}:{int(round(s)):02d}"


def parse_time(s):
    s = str(s).strip()
    if ":" in s:
        parts = [float(x) for x in s.split(":")]
        t = 0.0
        for p in parts:
            t = t * 60 + p
        return t
    return float(s)


def font(size):
    for p in ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/PingFang.ttc", "C:/Windows/Fonts/msyh.ttc"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def inputs(W):
    d = jload(os.path.join(W, "inputs.json"))
    if not d:
        die(f"{W}/inputs.json not found: run `probe` first")
    return d


def item(W, code):
    for it in inputs(W)["items"]:
        if it["code"] == code:
            return it
    die(f"no input with code {code}")


def frame_id(code, t=None):
    return code if t is None else f"{code}_{t:.1f}"


def parse_frame_ref(ref):
    """'C_140.0' / 'C@140' / 'C@2:20' / 'C' (image) -> (code, t or None)"""
    ref = ref.strip()
    m = re.fullmatch(r"([A-Z]{1,2})(?:[_@ ]([0-9:.]+))?", ref)
    if not m:
        die(f"bad frame reference {ref!r}; use C_140.0, C@2:20 or C (image)")
    return m.group(1), (parse_time(m.group(2)) if m.group(2) else None)


# ───────────────────────── colour maths ─────────────────────────
def srgb_to_lin(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb8):
    """rgb8: (...,3) uint8 or 0-255 floats -> Lab (...,3)"""
    lin = srgb_to_lin(np.asarray(rgb8, dtype=np.float64) / 255.0)
    xyz = lin @ _M.T / _WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def luminance(rgb8):
    lin = srgb_to_lin(np.asarray(rgb8, dtype=np.float64) / 255.0)
    return lin @ np.array([0.2126, 0.7152, 0.0722])


def hexc(rgb):
    r, g, b = [int(round(min(255, max(0, v)))) for v in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def lch(rgb):
    L, a, b = rgb_to_lab(np.array(rgb, dtype=np.float64))
    return round(float(L), 1), round(float(math.hypot(a, b)), 1), round(float(math.degrees(math.atan2(b, a)) % 360), 0)


def kmeans(X, k, iters=25, seed=1):
    """Plain k-means with k-means++ init. Returns centres (k,d) and labels."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    c = [X[rng.integers(n)]]
    d2 = ((X - c[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = d2 / d2.sum() if d2.sum() > 0 else None
        c.append(X[rng.choice(n, p=p)])
        d2 = np.minimum(d2, ((X - c[-1]) ** 2).sum(1))
    c = np.array(c)
    for _ in range(iters):
        lab = ((X[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        nc = np.array([X[lab == i].mean(0) if (lab == i).any() else c[i] for i in range(k)])
        if np.allclose(nc, c):
            break
        c = nc
    lab = ((X[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
    return c, lab


def clusters(rgb_px, k, seed=1, max_n=40000):
    """Cluster pixels in Lab; report the mean sRGB of each cluster with its share, sorted by share."""
    rgb_px = np.asarray(rgb_px).reshape(-1, 3)
    if len(rgb_px) == 0:
        return []
    rng = np.random.default_rng(seed)
    if len(rgb_px) > max_n:
        rgb_px = rgb_px[rng.choice(len(rgb_px), max_n, replace=False)]
    lab = rgb_to_lab(rgb_px)
    _, labels = kmeans(lab, k, seed=seed)
    out = []
    for i in np.unique(labels):
        m = labels == i
        mean = rgb_px[m].mean(0)
        L, C, h = lch(mean)
        out.append({"hex": hexc(mean), "share": round(float(m.mean()), 3), "L": L, "C": C, "h": h})
    return sorted(out, key=lambda d: -d["share"])


def circ_mean_deg(h_deg, w):
    if len(h_deg) == 0 or w.sum() <= 0:
        return None
    r = np.radians(h_deg)
    return round(float(np.degrees(np.arctan2((np.sin(r) * w).sum(), (np.cos(r) * w).sum())) % 360), 0)


# ───────────────────────── probe ─────────────────────────
def ffprobe(path):
    try:
        out = run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path]).stdout
        return json.loads(out)
    except Exception:
        return None


def classify(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXT or (not ext and _is_still(path)):
        if _is_still(path):
            return "image"
    pr = ffprobe(path)
    if not pr:
        return None
    vs = [s for s in pr.get("streams", []) if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic", 0) == 0]
    au = [s for s in pr.get("streams", []) if s.get("codec_type") == "audio"]
    dur = float(pr.get("format", {}).get("duration", 0) or 0)
    if vs and dur > 0.5:
        return "video"
    if vs and _is_still(path):
        return "image"
    if au:
        return "audio"
    return None


def _is_still(path):
    try:
        with Image.open(path) as im:
            return getattr(im, "n_frames", 1) == 1
    except Exception:
        return False


def cmd_probe(a):
    W = a.work
    os.makedirs(W, exist_ok=True)
    files = []
    for p in a.inputs:
        if os.path.isdir(p):
            for q in sorted(glob.glob(os.path.join(p, "**", "*"), recursive=True)):
                if os.path.isfile(q):
                    files.append(q)
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"warning: {p} not found, skipped", file=sys.stderr)
    old = jload(os.path.join(W, "inputs.json"), {"items": []})
    known = {it["path"]: it for it in old["items"]}
    items, used = [], set()
    letters = [chr(65 + i) for i in range(26)] + [chr(65 + i) + chr(65 + j) for i in range(26) for j in range(26)]
    for it in old["items"]:
        used.add(it["code"])
    for f in files:
        f = os.path.abspath(f)
        if f in known:
            items.append(known[f])
            continue
        kind = classify(f)
        if kind is None:
            print(f"skip (not video/image/audio): {f}", file=sys.stderr)
            continue
        code = next(c for c in letters if c not in used)
        used.add(code)
        it = {"code": code, "kind": kind, "path": f, "name": os.path.basename(f), "bytes": os.path.getsize(f)}
        if kind == "image":
            with Image.open(f) as im:
                it.update(width=im.width, height=im.height)
        else:
            pr = ffprobe(f)
            fmt = pr.get("format", {})
            it["duration"] = round(float(fmt.get("duration", 0) or 0), 2)
            vs = [s for s in pr["streams"] if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic", 0) == 0]
            au = [s for s in pr["streams"] if s.get("codec_type") == "audio"]
            if vs:
                v = vs[0]
                num, den = (v.get("avg_frame_rate") or "0/1").split("/")
                it.update(width=v.get("width"), height=v.get("height"), fps=round(float(num) / float(den), 2) if float(den) else None, vcodec=v.get("codec_name"))
            it["has_audio"] = bool(au)
            if au:
                it.update(audio_rate=int(au[0].get("sample_rate", 0) or 0), audio_channels=au[0].get("channels"), acodec=au[0].get("codec_name"))
        items.append(it)
    jsave(os.path.join(W, "inputs.json"), {"items": items})
    print(f"{'code':4} {'kind':5} {'size':>11} {'dur':>8} {'fps':>6} audio  name")
    for it in items:
        size = f"{it.get('width', '-')}x{it.get('height', '-')}" if it.get("width") else "-"
        dur = mmss(it["duration"]) if it.get("duration") else "-"
        print(f"{it['code']:4} {it['kind']:5} {size:>11} {dur:>8} {str(it.get('fps') or '-'):>6} {'yes' if it.get('has_audio') else 'no ':5}  {it['name']}")


# ───────────────────────── frame decoding ─────────────────────────
def decode_stream(path, fps, width, t0=None, t1=None):
    """Yield (t, rgb uint8 array) at the given fps, scaled to `width` (None = native)."""
    pr = ffprobe(path)
    v = [s for s in pr["streams"] if s.get("codec_type") == "video"][0]
    W0, H0 = int(v["width"]), int(v["height"])
    if width and width < W0:
        w = int(width) // 2 * 2
        h = int(round(H0 * w / W0 / 2)) * 2
        vf = f"fps={fps},scale={w}:{h}:flags=area"
    else:
        w, h = W0, H0
        vf = f"fps={fps}"
    cmd = ["ffmpeg", "-v", "error"]
    if t0 is not None:
        cmd += ["-ss", f"{t0:.3f}"]
    if t1 is not None:
        cmd += ["-to", f"{t1:.3f}"]
    cmd += ["-i", path, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n, size = 0, w * h * 3
    base = t0 or 0.0
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        yield base + n / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        n += 1
    p.wait()


def grab_frame(W, code, t=None, force=False):
    """Full-resolution PNG for a video time or an image; returns its path."""
    it = item(W, code)
    fid = frame_id(code, None if it["kind"] == "image" else t)
    out = os.path.join(W, "frames", "full", fid + ".png")
    if os.path.exists(out) and not force:
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if it["kind"] == "image":
        Image.open(it["path"]).convert("RGB").save(out)
    elif it["kind"] == "video":
        if t is None:
            die(f"{code} is a video: give a time, e.g. {code}@1:20")
        t = min(max(0.0, t), max(0.0, it["duration"] - 0.05))
        run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", it["path"], "-frames:v", "1", "-y", out])
        if not os.path.exists(out):
            die(f"could not grab {code} at {t}")
    else:
        die(f"{code} is audio only")
    return out


def load_frame(W, ref, crop=True):
    """ref = frame id / path. Applies the input's crop box from overlay.json unless crop=False."""
    if os.path.exists(ref):
        return np.asarray(Image.open(ref).convert("RGB")), None
    code, t = parse_frame_ref(ref)
    p = grab_frame(W, code, t)
    a = np.asarray(Image.open(p).convert("RGB"))
    box = crop_box(W, code) if crop else None
    if box:
        x0, y0, x1, y1 = box
        a = a[y0:y1, x0:x1]
    return a, box


def crop_box(W, code):
    ov = jload(os.path.join(W, "overlay.json"), {})
    b = ov.get(code, {}).get("crop")
    return [int(v) for v in b] if b else None


# ───────────────────────── sample ─────────────────────────
def contact_sheets(W, ids, labels, prefix, cols=4, rows=4, cell=480):
    os.makedirs(os.path.join(W, "sheets"), exist_ok=True)
    f = font(18)
    outs = []
    per = cols * rows
    for s in range(0, len(ids), per):
        chunk = list(zip(ids[s:s + per], labels[s:s + per]))
        ims = []
        for fid, lab in chunk:
            p = os.path.join(W, "frames", "thumbs", fid + ".jpg")
            im = Image.open(p).convert("RGB")
            im = im.resize((cell, max(1, round(im.height * cell / im.width))), Image.LANCZOS)
            ims.append((im, lab))
        rh = max(i.height for i, _ in ims) + 30
        nrows = (len(ims) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * (cell + 8) + 8, nrows * rh + 8), (24, 24, 30))
        d = ImageDraw.Draw(sheet)
        for i, (im, lab) in enumerate(ims):
            x, y = 8 + (i % cols) * (cell + 8), 8 + (i // cols) * rh
            sheet.paste(im, (x, y))
            d.text((x + 2, y + im.height + 4), lab, font=f, fill=(235, 230, 215))
        out = os.path.join(W, "sheets", f"{prefix}_{s // per + 1:02d}.jpg")
        sheet.save(out, quality=82)
        outs.append(out)
    return outs


def cut_sheets(W, code, tmpdir, ts, idxs, cell=320, per=8, cap=24):
    """One before | after pair for each plain-cut moment (one picture change that is no fade or flash): the quick check on
    what changed (a camera cut, a pan, a menu opening, an effect) before the motion chapter calls it editing."""
    if not idxs:
        return []
    if len(idxs) > cap:
        idxs = [idxs[j] for j in sorted(set(np.linspace(0, len(idxs) - 1, cap).round().astype(int)))]
    f = font(16)
    outs = []
    os.makedirs(os.path.join(W, "sheets"), exist_ok=True)
    for s0 in range(0, len(idxs), per):
        pairs = []
        for i in idxs[s0:s0 + per]:
            ims = [Image.open(os.path.join(tmpdir, f"{k:05d}.jpg")).convert("RGB") for k in (i - 1, i)]
            ims = [im.resize((cell, max(1, round(im.height * cell / im.width))), Image.LANCZOS) for im in ims]
            pairs.append((ims, f"{code} {mmss(ts[i - 1])} → {mmss(ts[i])}"))
        ph = max(im.height for ims, _ in pairs for im in ims) + 28
        cols = 2
        rows = (len(pairs) + cols - 1) // cols
        pw_ = 2 * cell + 4
        sheet = Image.new("RGB", (cols * (pw_ + 12) + 12, rows * ph + 12), (24, 24, 30))
        d = ImageDraw.Draw(sheet)
        for j, (ims, lab) in enumerate(pairs):
            x, y = 12 + (j % cols) * (pw_ + 12), 12 + (j // cols) * ph
            sheet.paste(ims[0], (x, y))
            sheet.paste(ims[1], (x + cell + 4, y))
            d.text((x, y + ims[0].height + 4), lab, font=f, fill=(235, 230, 215))
        p = os.path.join(W, "sheets", f"{code}_cuts_{s0 // per + 1:02d}.jpg")
        sheet.save(p, quality=82)
        outs.append(p)
    return outs


def cmd_sample(a):
    W = a.work
    its = inputs(W)["items"]
    if a.codes:
        its = [it for it in its if it["code"] in a.codes.split(",")]
    os.makedirs(os.path.join(W, "frames", "thumbs"), exist_ok=True)
    image_ids = []
    for it in its:
        code = it["code"]
        if it["kind"] == "image":
            im = Image.open(it["path"]).convert("RGB")
            im.thumbnail((960, 960))
            im.save(os.path.join(W, "frames", "thumbs", code + ".jpg"), quality=88)
            image_ids.append(code)
            continue
        if it["kind"] != "video":
            continue
        dur = it["duration"]
        fps = min(a.fps, max(0.2, a.cap / max(dur, 1)))
        feats, ts, sharp, lum = [], [], [], []
        tmpdir = os.path.join(W, "frames", "_stream", code)
        os.makedirs(tmpdir, exist_ok=True)
        for t, fr in decode_stream(it["path"], fps, 480):
            g = fr.mean(2)
            small = np.asarray(Image.fromarray(fr).resize((32, 18), Image.BILINEAR), dtype=np.float32) / 255
            feats.append(small.reshape(-1))
            ts.append(t)
            sharp.append(float(np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean()))
            lum.append(float(g.mean() / 255))
            Image.fromarray(fr).save(os.path.join(tmpdir, f"{len(ts) - 1:05d}.jpg"), quality=85)
        if not ts:
            print(f"{code}: no frames decoded", file=sys.stderr)
            continue
        F = np.array(feats)
        diff = np.r_[0, np.abs(np.diff(F, axis=0)).mean(1)]
        thr = max(a.cut, float(np.median(diff) * 4))
        cuts = [i for i in range(1, len(ts)) if diff[i] > thr]
        black = [i for i, l in enumerate(lum) if l < 0.03]
        # most "cuts" in gameplay are transitions (fade to black around battles and menus, white flashes),
        # not camera cuts; count them apart so the motion chapter does not report transitions as editing
        la = np.array(lum)
        near = max(1, round(fps))                        # about one second either side
        kind = {}
        for i in cuts:
            around = la[max(0, i - near):i + near + 1]
            kind[i] = "black" if around.min() < 0.04 else "flash" if (around.max() > 0.85 or abs(la[i] - la[i - 1]) > 0.35) else "cut"
        bounds = [0] + cuts + [len(ts)]
        shots = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]
        picks = set()
        for s0, s1 in shots:
            lo, hi = s0 + (s1 - s0) // 5, max(s0 + 1, s1 - (s1 - s0) // 5)
            cand = [i for i in range(lo, hi) if i not in black] or [i for i in range(s0, s1) if i not in black]
            if cand and (s1 - s0) / fps >= 0.6:
                picks.add(max(cand, key=lambda i: sharp[i]))
        every = a.every or min(10.0, max(2.0, dur / 30))
        step = max(1, int(round(every * fps)))
        for i in range(step // 2, len(ts), step):
            if i not in black:
                picks.add(i)
        picks = sorted(picks)
        kept = []
        for i in picks:
            if kept and np.abs(F[i] - F[kept[-1]]).mean() < a.dedup and (ts[i] - ts[kept[-1]]) < every * 2:
                continue
            kept.append(i)
        if len(kept) > a.max:
            idx = np.linspace(0, len(kept) - 1, a.max).round().astype(int)
            kept = [kept[j] for j in sorted(set(idx))]
        ids, labels = [], []
        for i in kept:
            fid = frame_id(code, round(ts[i], 1))
            shutil.copy(os.path.join(tmpdir, f"{i:05d}.jpg"), os.path.join(W, "frames", "thumbs", fid + ".jpg"))
            ids.append(fid)
            labels.append(f"{code} {mmss(ts[i])}")
        act = [round(float(x), 4) for x in diff]
        shot_t = [[round(ts[s0], 1), round(ts[s1 - 1] + 1 / fps, 1)] for s0, s1 in shots]
        lens = [b - a_ for a_, b in shot_t]
        per_min = lambda n: round(n / max(dur / 60, 1e-6), 2)
        nk = {k: sum(1 for v in kind.values() if v == k) for k in ("cut", "black", "flash")}
        plain_i = [i for i in cuts if kind[i] == "cut"]
        moment_i = [i for j, i in enumerate(plain_i) if j == 0 or ts[i] - ts[plain_i[j - 1]] > 1.5]
        moments = len(moment_i)    # bursts = effects or menus
        cutsheets = cut_sheets(W, code, tmpdir, ts, moment_i)
        shutil.rmtree(tmpdir, ignore_errors=True)
        info = {"code": code, "fps_sampled": round(fps, 3), "cuts": [round(ts[i], 1) for i in cuts],
                "cut_kinds": {k: [round(ts[i], 1) for i in cuts if kind[i] == k] for k in ("cut", "black", "flash")},
                "black": [round(ts[i], 1) for i in black], "shots": shot_t, "picks": ids,
                "times": [round(t, 2) for t in ts], "activity": act,
                "summary": {"duration": dur, "shots": len(shots), "cuts_per_min": per_min(len(cuts)), "plain_cuts_per_min": per_min(nk["cut"]), "plain_cut_moments_per_min": per_min(moments),
                            "median_shot_s": round(float(np.median(lens)), 1) if lens else None,
                            "black_frames": len(black), "mean_activity": round(float(np.mean(diff[1:])) if len(diff) > 1 else 0, 4)}}
        jsave(os.path.join(W, "frames", f"{code}.json"), info)
        sheets = contact_sheets(W, ids, labels, code)
        s = info["summary"]
        print(f"{code}: {len(ids)} frames kept of {len(ts)} sampled @ {fps:.2f} fps; {s['shots']} shots, median shot {s['median_shot_s']} s, {len(black)} black samples")
        print(f"   picture changes {s['cuts_per_min']}/min: {nk['black']} through black (fades, battle and menu transitions), {nk['flash']} flashes, "
              f"{nk['cut']} plain cuts at {moments} separate moments ({s['plain_cut_moments_per_min']}/min; changes less than 1.5 s apart count once). "
              "Plain cuts include effects, menus and UI cards: look at the sheets before calling any of them camera cuts.")
        for sh in sheets:
            print("   sheet", sh)
        for sh in cutsheets:
            print("   cut pairs", sh, "(before | after for each plain-cut moment: a camera cut, or a pan, a menu, an effect?)")
    if image_ids:
        sheets = contact_sheets(W, image_ids, [f"{c} · {item(W, c)['name'][:28]}" for c in image_ids], "images", cols=3, rows=3, cell=620)
        for sh in sheets:
            print("images sheet", sh)


# ───────────────────────── overlay (watermarks, static HUD) ─────────────────────────
def corner_sheets(W, its, frac=0.22, tw=360, per=6):
    """Stills have no motion to find overlays from: the four corners of every image side by side, so a watermark,
    logo or frame shows at a glance. Each tile is the outer 22% x 22% of that corner."""
    os.makedirs(os.path.join(W, "overlay"), exist_ok=True)
    f = font(16)
    names = ("top-left", "top-right", "bottom-left", "bottom-right")
    outs = []
    for s0 in range(0, len(its), per):
        rows = []
        for it in its[s0:s0 + per]:
            im = Image.open(it["path"]).convert("RGB")
            w, h = im.size
            cw, ch = max(8, round(w * frac)), max(8, round(h * frac))
            boxes = [(0, 0, cw, ch), (w - cw, 0, w, ch), (0, h - ch, cw, h), (w - cw, h - ch, w, h)]
            rows.append((it["code"], [im.crop(b).resize((tw, max(1, round(tw * ch / cw))), Image.LANCZOS) for b in boxes], (w, h)))
        th = max(r[1][0].height for r in rows)
        sheet = Image.new("RGB", (4 * (tw + 6) + 6, len(rows) * (th + 24) + 6), (24, 24, 30))
        d = ImageDraw.Draw(sheet)
        for i, (code, tiles, (w, h)) in enumerate(rows):
            for j, t in enumerate(tiles):
                x, y = 6 + j * (tw + 6), 6 + i * (th + 24)
                sheet.paste(t, (x, y + 20))
                d.text((x + 2, y + 1), f"{code} {names[j]}" + (f"  ({w}x{h})" if j == 0 else ""), font=f, fill=(235, 230, 215))
        out = os.path.join(W, "overlay", f"images_corners_{s0 // per + 1:02d}.jpg")
        sheet.save(out, quality=85)
        outs.append(out)
    return outs


def cmd_overlay(a):
    W = a.work
    ov = jload(os.path.join(W, "overlay.json"), {})
    os.makedirs(os.path.join(W, "overlay"), exist_ok=True)
    for it in inputs(W)["items"]:
        if it["kind"] != "video" or (a.codes and it["code"] not in a.codes.split(",")):
            continue
        code, dur = it["code"], it["duration"]
        frames = [fr for _, fr in decode_stream(it["path"], a.n / max(dur, 1), 640)]
        if len(frames) < 4:
            continue
        g = np.stack(frames).astype(np.float32).mean(3)
        dh, dw = g.shape[1:]                             # decoded size (640 wide)
        e = np.abs(np.diff(g, axis=2))[:, :-1, :] + np.abs(np.diff(g, axis=1))[:, :, :-1]
        pres = (e > a.edge).mean(0)                      # share of samples with an edge at this pixel
        mask = pres >= 0.7
        h, w = mask.shape
        cs = max(4, round(dw / 64))                      # square cells that tile the whole frame
        gh, gw = math.ceil(h / cs), math.ceil(w / cs)
        padm = np.zeros((gh * cs, gw * cs), bool)
        padm[:h, :w] = mask
        cells = padm.reshape(gh, cs, gw, cs).mean((1, 3)) > 0.015
        static_share = float(cells.mean())
        d = cells.copy()                                 # dilate one cell so text and logo merge
        d[1:] |= cells[:-1]; d[:-1] |= cells[1:]; d[:, 1:] |= cells[:, :-1]; d[:, :-1] |= cells[:, 1:]
        seen = np.zeros_like(d)
        boxes = []
        for y in range(gh):
            for x in range(gw):
                if d[y, x] and not seen[y, x]:
                    st, comp = [(y, x)], []
                    seen[y, x] = True
                    while st:
                        yy, xx = st.pop()
                        comp.append((yy, xx))
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                ny, nx = yy + dy, xx + dx
                                if 0 <= ny < gh and 0 <= nx < gw and d[ny, nx] and not seen[ny, nx]:
                                    seen[ny, nx] = True
                                    st.append((ny, nx))
                    ys, xs = [c[0] for c in comp], [c[1] for c in comp]
                    # the dilated cells only join the pieces; the box itself hugs the static edges, plus a small margin
                    cx0, cy0, cx1, cy1 = min(xs) * cs, min(ys) * cs, (max(xs) + 1) * cs, (max(ys) + 1) * cs
                    sub = padm[cy0:cy1, cx0:cx1]
                    ry, rx = np.where(sub.any(1))[0], np.where(sub.any(0))[0]
                    if len(ry) and len(rx):
                        mg = max(2, round(0.012 * dh))
                        boxes.append([max(0, cx0 + rx[0] - mg), max(0, cy0 + ry[0] - mg), cx0 + rx[-1] + 1 + mg, cy0 + ry[-1] + 1 + mg])
        Wd, Hd = it["width"], it["height"]
        sx, sy = Wd / dw, Hd / dh                        # decoded -> full-frame pixels
        px = [[int(b[0] * sx), int(b[1] * sy), min(Wd, int(math.ceil(b[2] * sx))), min(Hd, int(math.ceil(b[3] * sy)))] for b in boxes]
        # watermarks and HUD live in the corners; keep small boxes whose centre is in a corner zone
        corner = [b for b in px if ((b[0] + b[2]) / 2 < 0.25 * Wd or (b[0] + b[2]) / 2 > 0.75 * Wd)
                  and ((b[1] + b[3]) / 2 < 0.25 * Hd or (b[1] + b[3]) / 2 > 0.75 * Hd)
                  and (b[2] - b[0]) * (b[3] - b[1]) < 0.08 * Wd * Hd]
        # a watermark is there in every sample; scenery is "static" only while the camera holds still (a fixed-camera
        # dialogue, a pause screen), so its strongest edges are present in fewer samples
        always, part = [], []
        for b in corner:
            m = pres[int(b[1] * dh / Hd):int(b[3] * dh / Hd), int(b[0] * dw / Wd):int(b[2] * dw / Wd)]
            p90 = float(np.percentile(m[m >= 0.3], 90)) if (m >= 0.3).any() else 0.0
            (always if p90 >= 0.92 else part).append((b, p90))
        static = static_share > 0.12
        conf = "low: the footage itself is mostly static" if static else ("low: short clip" if dur < 40 else "medium")
        keepb = [b for b, _ in always]
        crop = best_crop(Wd, Hd, keepb, 16 / 9)
        ov.setdefault(code, {})
        if ov[code].get("auto", True):
            ov[code].update({"boxes": keepb, "part_time": [b for b, _ in part], "crop": crop, "confidence": conf, "auto": True})
        mid = frames[len(frames) // 2]
        pv = Image.fromarray(mid).resize((960, round(960 * Hd / Wd)))
        k = 960 / Wd
        dr = ImageDraw.Draw(pv)
        for b in keepb:
            dr.rectangle([b[0] * k, b[1] * k, b[2] * k, b[3] * k], outline=(255, 60, 60), width=3)
        for b, _ in part:
            dr.rectangle([b[0] * k, b[1] * k, b[2] * k, b[3] * k], outline=(255, 190, 40), width=2)
        dr.rectangle([crop[0] * k, crop[1] * k, crop[2] * k - 1, crop[3] * k - 1], outline=(80, 220, 120), width=3)
        heat = Image.fromarray((np.clip(pres, 0, 1) * 255).astype(np.uint8)).resize(pv.size)
        both = Image.new("RGB", (pv.width, pv.height * 2 + 6), (0, 0, 0))
        both.paste(pv, (0, 0))
        both.paste(heat.convert("RGB"), (0, pv.height + 6))
        outp = os.path.join(W, "overlay", f"{code}.jpg")
        both.save(outp, quality=85)
        kept = (crop[2] - crop[0]) * (crop[3] - crop[1]) / (Wd * Hd)
        print(f"{code}: overlays {keepb} -> crop {crop}, keeps {kept:.0%} of the frame  [{conf}; static share {static_share:.0%}]  preview {outp}")
        for b, p90 in part:
            print(f"   not used: {b} is static in only ~{p90:.0%} of samples (scenery while the camera held still?). "
                  "If it is a real overlay that fades in and out, set the crop by hand.")
        if kept < 0.7:
            print(f"   warning: the crop keeps only {kept:.0%} of the frame. Check the red boxes on the preview; plates can also use "
                  "their own boxes in crops.json as long as they stay clear of the red boxes.")
        if not ov[code].get("auto", True):
            print(f"   (your hand-set crop {ov[code]['crop']} is kept)")
    jsave(os.path.join(W, "overlay.json"), ov)
    vids = [it for it in inputs(W)["items"] if it["kind"] == "video" and not (a.codes and it["code"] not in a.codes.split(","))]
    imgs = [it["code"] for it in inputs(W)["items"] if it["kind"] == "image" and not (a.codes and it["code"] not in a.codes.split(","))]
    if vids:
        print("Look at every preview (top: red = overlay, orange = static only part of the time, not used; green = crop; "
              "bottom: white = edges that never move). Set a crop by hand with `crop C=x0,y0,x1,y1`; hand-set crops are kept on reruns.")
    if imgs:
        outs = corner_sheets(W, [it for it in inputs(W)["items"] if it["code"] in imgs])
        print(f"images ({', '.join(imgs)}): a still has no motion to find overlays from, so there are no boxes. "
              f"Look at the corner sheets instead (one row per image, outer 22% of each corner): {', '.join(outs)}. "
              "If an image has a watermark or frame, set its crop with `crop A=x0,y0,x1,y1`.")


def best_crop(Wd, Hd, boxes, ratio):
    """Largest box of the given aspect ratio that avoids all overlay boxes (searching trims from each side)."""
    if not boxes:
        return fit_ratio([0, 0, Wd, Hd], ratio)
    xs = sorted({0} | {b[2] for b in boxes if b[2] < Wd * 0.5} | {b[2] for b in boxes})
    xe = sorted({Wd} | {b[0] for b in boxes})
    ys = sorted({0} | {b[3] for b in boxes})
    ye = sorted({Hd} | {b[1] for b in boxes})
    best, area = fit_ratio([0, 0, Wd, Hd], ratio), -1
    for x0 in xs:
        for x1 in xe:
            for y0 in ys:
                for y1 in ye:
                    if x1 - x0 < Wd * 0.4 or y1 - y0 < Hd * 0.4:
                        continue
                    c = fit_ratio([x0, y0, x1, y1], ratio)
                    if any(not (c[2] <= b[0] or c[0] >= b[2] or c[3] <= b[1] or c[1] >= b[3]) for b in boxes):
                        continue
                    ar = (c[2] - c[0]) * (c[3] - c[1])
                    if ar > area:
                        best, area = c, ar
    return best


def fit_ratio(b, ratio):
    x0, y0, x1, y1 = b
    w, h = x1 - x0, y1 - y0
    if w / h > ratio:
        nw = int(h * ratio)
        x0 += (w - nw) // 2
        return [x0, y0, x0 + nw, y1]
    nh = int(w / ratio)
    y0 += (h - nh) // 2
    return [x0, y0, x1, y0 + nh]


def cmd_overlay_set(a):
    ov = jload(os.path.join(a.work, "overlay.json"), {})
    for s in a.set:
        code, box = s.split("=")
        ov.setdefault(code, {})["crop"] = [int(v) for v in box.split(",")]
        ov[code]["auto"] = False
        print(code, "crop", ov[code]["crop"])
    jsave(os.path.join(a.work, "overlay.json"), ov)


# ───────────────────────── grab / zoom / sheet ─────────────────────────
def cmd_grab(a):
    for ref in a.refs:
        code, t = parse_frame_ref(ref)
        print(grab_frame(a.work, code, t, force=a.force))


def nice_step(span):
    for k in range(0, 6):
        for m in (1, 2, 5):
            if m * 10 ** k >= span:
                return m * 10 ** k
    return 100000


def cmd_zoom(a):
    """Region (or whole frame) resized for viewing, with a grid labelled in full-frame pixel coordinates,
    so boxes for crops / specimens / pixel can be read straight off the image."""
    arr, _ = load_frame(a.work, a.ref, crop=False)
    H, Wd = arr.shape[:2]
    if a.box and len(a.box) != 4:
        die("box is x0 y0 x1 y1 (full-frame pixels)")
    x0, y0, x1, y1 = a.box or [0, 0, Wd, H]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(Wd, x1), min(H, y1)
    im = Image.fromarray(arr[y0:y1, x0:x1])
    sc = a.scale or min(8.0, 1600 / im.width, 1600 / im.height)
    im = im.resize((max(1, round(im.width * sc)), max(1, round(im.height * sc))), Image.NEAREST if sc >= 1 else Image.LANCZOS)
    if not a.nogrid:
        step = a.grid or nice_step(max(x1 - x0, y1 - y0) / 10)
        d = ImageDraw.Draw(im)
        f = font(14)
        for gx in range(math.ceil(x0 / step) * step, x1, step):
            X = (gx - x0) * sc
            d.line([(X, 0), (X, im.height)], fill=(255, 0, 200), width=1)
            d.text((X + 3, 2), str(gx), font=f, fill=(255, 240, 80), stroke_width=2, stroke_fill=(0, 0, 0))
        for gy in range(math.ceil(y0 / step) * step, y1, step):
            Y = (gy - y0) * sc
            d.line([(0, Y), (im.width, Y)], fill=(255, 0, 200), width=1)
            d.text((3, Y + 2), str(gy), font=f, fill=(255, 240, 80), stroke_width=2, stroke_fill=(0, 0, 0))
    out = a.out or os.path.join(a.work, "zoom", f"{a.ref.replace('@', '_').replace(':', '-')}_{x0}_{y0}_{x1}_{y1}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)    # --out may be a bare file name
    im.save(out)
    print(out, im.size, f"scale {sc:.2f}" + ("" if a.nogrid else f", grid every {step} px (full-frame coordinates)"))


def cmd_sheet(a):
    """Contact sheet of images. Columns and cell width follow from the number of images (about 1600 px wide, at most
    about 1100 px high); more than fit make several sheets OUT_01.jpg, OUT_02.jpg ... Give --cols/--cell to choose."""
    if not a.out:
        if len(a.images) < 2:
            die("usage: sheet IMAGES... -o OUT.jpg   (or: sheet OUT.jpg IMAGES...)")
        a.out, a.images = a.images[0], a.images[1:]
    missing = [p for p in a.images if not os.path.isfile(p)]
    if missing:
        die("no such image: " + ", ".join(missing[:4]) + " (a wildcard that matched nothing is passed on as it is)")
    raw = [(Image.open(p).convert("RGB"), os.path.splitext(os.path.basename(p))[0]) for p in a.images]
    n = len(raw)
    aspect = float(np.median([im.height / im.width for im, _ in raw]))
    if a.cols or a.cell:    # chosen by hand: one sheet unless --per says otherwise
        cols = a.cols or max(1, min(n, 1600 // ((a.cell or 420) + 8)))
        cell = a.cell or int(min(560, 1600 / cols - 8))
        single = not a.per
    else:    # the biggest cells that put every image on one sheet of about 1600 x 1100 px, else pages that fill up to 1100 px at 280 px
        cell = cols = 0
        for c in range(560, 190, -10):
            k = max(1, min(n, 1600 // (c + 8)))
            if -(-n // k) * (c * aspect + 28) + 8 <= 1100:
                cell, cols = c, k
                break
        single = bool(cell) and not a.per
        if not cell:
            cell, cols = 280, max(1, min(n, 5))
    ims_all = [(im.resize((cell, max(1, round(im.height * cell / im.width))), Image.LANCZOS), lab) for im, lab in raw]
    if single:
        pages = [ims_all]
    elif a.per:
        pages = [ims_all[i:i + a.per] for i in range(0, n, a.per)]
    else:    # fill a page row by row until the next row would pass 1100 px
        pages, cur, hgt = [], [], 8
        for r0 in range(0, n, cols):
            row = ims_all[r0:r0 + cols]
            rh = max(i.height for i, _ in row) + 28
            if cur and hgt + rh > 1100:
                pages.append(cur)
                cur, hgt = [], 8
            cur += row
            hgt += rh
        pages.append(cur)
    outs = []
    f = font(16)
    for pi, ims in enumerate(pages):
        rows = [ims[i:i + cols] for i in range(0, len(ims), cols)]
        Wd = min(cols, len(ims)) * (cell + 8) + 8
        Hd = sum(max(i.height for i, _ in r) + 28 for r in rows) + 8
        o = Image.new("RGB", (Wd, Hd), (24, 24, 30))
        d = ImageDraw.Draw(o)
        y = 8
        for r in rows:
            for c, (im, lab) in enumerate(r):
                x = 8 + c * (cell + 8)
                o.paste(im, (x, y))
                d.text((x, y + im.height + 4), lab, font=f, fill=(230, 230, 230))
            y += max(i.height for i, _ in r) + 28
        out = a.out if len(pages) == 1 else f"{os.path.splitext(a.out)[0]}_{pi + 1:02d}{os.path.splitext(a.out)[1] or '.jpg'}"
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        o.save(out, quality=85)
        outs.append(out)
        print(out, o.size, f"({len(ims)} images, {cols} columns, cell {cell} px)")
    return outs


# ───────────────────────── measure ─────────────────────────
SHADOW_L, LIGHT_L = 20.0, 55.0


def sharp_profile(gray, bands=12, valid=None):
    """Mean squared gradient per horizontal band, top to bottom; `valid` masks out pixels (HUD, dialogue boxes)."""
    ex = np.abs(np.diff(gray, axis=1))[:-1, :]
    ey = np.abs(np.diff(gray, axis=0))[:, :-1]
    e = (ex ** 2 + ey ** 2)
    h = e.shape[0]
    cuts = [(int(h * i / bands), int(h * (i + 1) / bands)) for i in range(bands)]
    if valid is None:
        return np.array([e[a:b].mean() for a, b in cuts])
    v = valid[:-1, :-1]
    return np.array([e[a:b][v[a:b]].mean() if v[a:b].sum() > 50 else np.nan for a, b in cuts])


def vignette(rgb):
    h, w = rgb.shape[:2]
    Y = luminance(rgb)
    L = rgb_to_lab(rgb)[..., 0]
    ph, pw = max(2, int(h * 0.12)), max(2, int(w * 0.12))
    corners = [(slice(0, ph), slice(0, pw)), (slice(0, ph), slice(w - pw, w)), (slice(h - ph, h), slice(0, pw)), (slice(h - ph, h), slice(w - pw, w))]
    cy, cx = slice(int(h * 0.375), int(h * 0.625)), slice(int(w * 0.375), int(w * 0.625))
    cY, cL = Y[cy, cx].mean(), L[cy, cx].mean()
    rY = [float(Y[s].mean() / max(cY, 1e-6)) for s in corners]
    rL = [float(L[s].mean() / max(cL, 1e-6)) for s in corners]
    return {"Y_ratio": round(float(np.mean(rY)), 3), "L_ratio": round(float(np.mean(rL)), 3),
            "Y_corners_tl_tr_bl_br": [round(v, 2) for v in rY]}


def measure_array(rgb, k_all=8, seed=1):
    h, w = rgb.shape[:2]
    if w > 1280:   # measurements don't need more than 1280 px; keeps clustering fast
        im = Image.fromarray(rgb).resize((1280, round(h * 1280 / w)), Image.BILINEAR)
        small = np.asarray(im)
    else:
        small = rgb
    px = small.reshape(-1, 3)
    lab = rgb_to_lab(px)
    L, A, B = lab[:, 0], lab[:, 1], lab[:, 2]
    C = np.hypot(A, B)
    H = np.degrees(np.arctan2(B, A)) % 360
    res = {"size": [int(w), int(h)]}
    tiers = []
    for name, m in (("shadow", L < SHADOW_L), ("mid", (L >= SHADOW_L) & (L < LIGHT_L)), ("light", L >= LIGHT_L)):
        pct = float(m.mean() * 100)
        cl = clusters(px[m], 3, seed) if m.sum() >= 50 else []
        tiers.append({"tier": name, "pct": round(pct, 1), "colors": [[c["hex"], c["share"]] for c in cl],
                      "mean_C": round(float(C[m].mean()), 1) if m.any() else None,
                      "hue": circ_mean_deg(H[m & (C > 4)], C[m & (C > 4)]),
                      "mean_a": round(float(A[m].mean()), 1) if m.any() else None, "mean_b": round(float(B[m].mean()), 1) if m.any() else None})
    res["tiers"] = tiers
    res["palette"] = clusters(px, k_all, seed)
    acc = [c for c in clusters(px, 16, seed + 1) if c["C"] >= 35 and c["share"] < 0.05]
    res["accents"] = sorted(acc, key=lambda c: -c["C"])[:4]
    hi = L >= max(60.0, float(np.percentile(L, 97)))
    res["highlights"] = clusters(px[hi], 3, seed) if hi.sum() >= 30 else []
    res["core_white_pct"] = round(float((L >= 95).mean() * 100), 2)
    res["L"] = {"mean": round(float(L.mean()), 1), "p5": round(float(np.percentile(L, 5)), 1), "p50": round(float(np.percentile(L, 50)), 1),
                "p95": round(float(np.percentile(L, 95)), 1)}
    res["contrast_L_p95_p5"] = round(res["L"]["p95"] - res["L"]["p5"], 1)
    res["chroma"] = {"mean": round(float(C.mean()), 1), "p90": round(float(np.percentile(C, 90)), 1)}
    res["vignette"] = vignette(small)
    gray = small.mean(2) / 255.0
    prof = sharp_profile(gray)
    res["sharpness_bands"] = [round(float(v / max(prof.max(), 1e-12)), 2) for v in prof]
    res["detail"] = round(float(np.sqrt(sharp_profile(gray, 1)[0])), 4)
    return res


def fmt_tiers(t):
    return "  ".join(f"{x['tier'][:2]} {x['pct']:4.1f}% " + " ".join(f"{h}({s:.0%})" for h, s in x["colors"]) for x in t)


def split_box(refs, box):
    """Allow a box straight after the frame (`measure A@1:20 x0 y0 x1 y1`), like zoom; or --box."""
    if len(refs) >= 5 and all(re.fullmatch(r"-?\d+", r) for r in refs[-4:]):
        if box:
            die("give the box once: either after the frame or with --box")
        return refs[:-4], [int(v) for v in refs[-4:]]
    return refs, (list(box) if box else None)


def cmd_measure(a):
    W = a.work
    db = jload(os.path.join(W, "measure.json"), {"frames": {}, "profiles": {}})
    refs, box = split_box(a.refs, a.box)
    if a.name and len(refs) > 1:
        die("--name needs a single frame")
    for ref in refs:
        if box:    # full-frame coordinates, as on the zoom grid
            arr, _ = load_frame(W, ref, crop=False)
            H, Wd = arr.shape[:2]
            x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(Wd, box[2]), min(H, box[3])
            if x1 - x0 < 8 or y1 - y0 < 8:
                die(f"box {box} is outside the {Wd}x{H} frame or too small")
            arr, region = arr[y0:y1, x0:x1], [x0, y0, x1, y1]
        else:
            arr, region = load_frame(W, ref, crop=not a.nocrop)
        r = measure_array(arr)
        r["crop"] = region
        r["region"] = region                 # full-frame pixels actually measured (None = whole frame)
        fid = ref_id(ref)
        # a region never overwrites the whole-frame entry: it gets its own name
        name = a.name or (f"{fid}@{region[0]},{region[1]},{region[2]},{region[3]}" if box else fid)
        r["frame"] = fid
        db["frames"][name] = r
        where = f"  region {region}" if box else ""
        print(f"── {name}  {r['size'][0]}x{r['size'][1]}{where}  L mean {r['L']['mean']} (p5 {r['L']['p5']}, p95 {r['L']['p95']})  C* mean {r['chroma']['mean']} p90 {r['chroma']['p90']}")
        print(f"   tiers  {fmt_tiers(r['tiers'])}")
        print(f"   hue    shadow {r['tiers'][0]['hue']}°  mid {r['tiers'][1]['hue']}°  light {r['tiers'][2]['hue']}°   (Lab hue angle, chroma-weighted)")
        print(f"   palette " + " ".join(f"{c['hex']}({c['share']:.0%})" for c in r["palette"]))
        print(f"   accents " + (" ".join(f"{c['hex']}(C{c['C']:.0f},{c['share']:.1%})" for c in r["accents"]) or "-"))
        print(f"   highlights " + (" ".join(f"{c['hex']}({c['share']:.0%})" for c in r["highlights"]) or "-") + f"   white cores {r['core_white_pct']}%")
        if not box:
            print(f"   vignette corners/centre: Y {r['vignette']['Y_ratio']}  L* {r['vignette']['L_ratio']}   sharpness bands {r['sharpness_bands']}")
    jsave(os.path.join(W, "measure.json"), db)


def cmd_profile(a):
    """Average the sharpness profile (24 bands, top to bottom) and vignette over many frames of one stretch of video,
    or over one or more still images (codes joined with commas: A,B,C). --exclude masks HUD, dialogue boxes,
    and bright windows or chandeliers that would read as sharp at any depth."""
    W = a.work
    codes = a.code.split(",")
    its = [item(W, c) for c in codes]
    if all(it["kind"] == "image" for it in its):
        t0 = t1 = 0.0
        frames = (load_frame(W, c, crop=not a.nocrop) for c in codes)
    elif len(its) == 1 and its[0]["kind"] == "video":
        it = its[0]
        t0 = parse_time(a.t0) if a.t0 else 0.0
        t1 = parse_time(a.t1) if a.t1 else it["duration"]
        box = None if a.nocrop else crop_box(W, a.code)
        frames = ((fr[box[1]:box[3], box[0]:box[2]] if box else fr, box)
                  for _, fr in decode_stream(it["path"], a.n / max(t1 - t0, 0.1), None, t0, t1))
    else:
        die("profile takes one video code (with an optional time range) or one or more image codes, e.g. A,B,C")
    nb = a.bands
    profs, vigs, vigsL, used, masked = [], [], [], 0, 0.0
    for fr, box in frames:
        g = fr.mean(2) / 255.0
        if g.mean() < 0.03:
            continue
        valid = None
        if a.exclude:
            valid = np.ones(g.shape, bool)
            ox, oy = (box[0], box[1]) if box else (0, 0)
            for x0, y0, x1, y1 in a.exclude:    # full-frame pixels -> this (cropped) frame
                valid[max(0, y0 - oy):max(0, y1 - oy), max(0, x0 - ox):max(0, x1 - ox)] = False
            masked = 1 - valid.mean()
        p = sharp_profile(g, nb, valid)
        profs.append(p / max(np.nanmax(p), 1e-12))
        v = vignette(fr[::2, ::2])
        vigs.append(v["Y_ratio"])
        vigsL.append(v["L_ratio"])
        used += 1
    if not profs:
        die("no usable frames in that range")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)    # bands that are wholly excluded
        prof = np.nanmean(profs, 0)
    prof = np.nan_to_num(prof / np.nanmax(prof))
    name = a.name or (f"{a.code}_{t0:.0f}_{t1:.0f}" if t1 else a.code.replace(",", "_"))
    db = jload(os.path.join(W, "measure.json"), {"frames": {}, "profiles": {}})
    sharp_rows = [i for i, v in enumerate(prof) if v >= a.keep]
    band = [round(min(sharp_rows) * 100 / nb), round((max(sharp_rows) + 1) * 100 / nb)] if sharp_rows else None
    db.setdefault("profiles", {})[name] = {"code": a.code, "t0": round(t0, 1), "t1": round(t1, 1), "frames": used, "bands": nb,
                                         "profile": [round(float(v), 2) for v in prof], "sharp_band_pct": band,
                                         "top": round(float(prof[0]), 2), "bottom": round(float(prof[-1]), 2),
                                         "vignette_Y": round(float(np.mean(vigs)), 3), "vignette_L": round(float(np.mean(vigsL)), 3), "exclude": a.exclude or []}
    jsave(os.path.join(W, "measure.json"), db)
    span = f"{mmss(t0)}–{mmss(t1)}" if t1 else "stills"
    print(f"{name}: {used} frames {span}  profile(top→bottom, {nb} bands) {[round(float(v), 2) for v in prof]}")
    print(f"   sharp band (≥{a.keep:.0%} of peak): {band}% of frame height from top (±{50 / nb:.0f}%);  top {prof[0]:.2f}  bottom {prof[-1]:.2f};"
          f"  vignette corners/centre L* {np.mean(vigsL):.2f} (linear Y {np.mean(vigs):.2f})" + (f";  {masked:.0%} of the frame excluded" if a.exclude else ""))


def cmd_pixel(a):
    """Pixel-art scale: dominant period of column/row edge energy in a region (screen px per art px)."""
    W = a.work
    refs, box = split_box([a.ref] + (a.box_pos or []), a.box)
    arr, _ = load_frame(W, refs[0], crop=False)
    if box:
        x0, y0, x1, y1 = box
        arr = arr[y0:y1, x0:x1]
    g = arr.astype(np.float64).mean(2)
    out = {}
    for axis, name in ((1, "x"), (0, "y")):
        d = np.abs(np.diff(g, axis=axis))
        s_ = d.mean(axis=1 - axis) if axis == 1 else d.mean(axis=1)
        s_ = s_ - s_.mean()
        n = len(s_)
        if n < 32:
            continue
        spec = np.abs(np.fft.rfft(s_ * np.hanning(n), n=8 * n))
        freqs = np.fft.rfftfreq(8 * n)
        m = (freqs >= 1 / 16) & (freqs <= 1 / 1.8)
        fs, ps = freqs[m], spec[m]
        order = np.argsort(-ps)
        cands = []
        for i in order:
            per = 1 / fs[i]
            if all(abs(per - c[0]) > 0.25 for c in cands):
                cands.append((round(float(per), 2), round(float(ps[i] / ps.mean()), 1)))
            if len(cands) == 3:
                break
        out[name] = cands
    fid = ref_id(refs[0])
    name = a.name or (f"{fid}@{box[0]},{box[1]},{box[2]},{box[3]}" if box else fid)
    db = jload(os.path.join(W, "measure.json"), {"frames": {}, "profiles": {}})
    db.setdefault("pixel", {})[name] = {"frame": fid, "box": box, "size": [arr.shape[1], arr.shape[0]], **out}
    jsave(os.path.join(W, "measure.json"), db)
    print(f"{name}: region {arr.shape[1]}x{arr.shape[0]}: period candidates (screen px, prominence×mean)")
    for k, v in out.items():
        print(f"   {k}: {v}")
    print("   A strong period that matches in x and y is the art-pixel size; harmonics (p/2, 2p) can also appear: check with `zoom`.\n"
          "   Box one sprite at a time: in a perspective (HD-2D) scene the scale changes with depth. Weak or mismatched peaks are"
          " common below 1080p, on soft textures and in motion blur; then say the scale was judged by eye (目测).")


def parse_patch(s):
    """'name=x0,y0,x1,y1', 'name=x,y' (a 5x5 patch around a point) or 'name=x,y,r' (a square of 2r+1 px around a point;
    r = 0 is one pixel, for a thin line or bar); the name is optional."""
    name, _, box = s.rpartition("=")
    try:
        v = [int(round(float(x))) for x in box.split(",")]
    except ValueError:
        v = []
    if len(v) == 2:
        v = [v[0] - 2, v[1] - 2, v[0] + 3, v[1] + 3]
    elif len(v) == 3 and v[2] >= 0:
        v = [v[0] - v[2], v[1] - v[2], v[0] + v[2] + 1, v[1] + v[2] + 1]
    if len(v) != 4:
        die(f"bad patch {s!r}: use name=x0,y0,x1,y1, name=x,y or name=x,y,r (full-frame pixels)")
    return name or box, v


def ref_id(ref):
    if os.path.exists(ref):
        return os.path.splitext(os.path.basename(ref))[0]
    return frame_id(*parse_frame_ref(ref))


def pick_sheet(arr, shown, path, tile=200):
    """What each pick measured: the box drawn on an enlarged piece of the frame around it, so a patch that was aimed
    next to the object is seen at once. `shown` is a list of (name, hex, (x0, y0, x1, y1))."""
    H, Wd = arr.shape[:2]
    f = font(14)
    tiles = []
    for name, hx, (x0, y0, x1, y1) in shown[:24]:
        m = int(min(max(24, 4 * max(x1 - x0, y1 - y0)), Wd, H))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        wx0, wy0 = min(max(0, cx - m // 2), Wd - m), min(max(0, cy - m // 2), H - m)
        sc = tile / m
        im = Image.fromarray(arr[wy0:wy0 + m, wx0:wx0 + m]).resize((tile, tile), Image.NEAREST if sc >= 1 else Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.rectangle([(x0 - wx0) * sc - 1, (y0 - wy0) * sc - 1, (x1 - wx0) * sc, (y1 - wy0) * sc], outline=(255, 0, 200), width=2)
        t = Image.new("RGB", (tile, tile + 34), (24, 24, 30))
        t.paste(im, (0, 0))
        td = ImageDraw.Draw(t)
        td.text((3, tile + 2), name[:26], font=f, fill=(235, 230, 215))
        td.rectangle([3, tile + 19, 17, tile + 31], fill=tuple(hex_to_rgb(hx)))
        td.text((22, tile + 18), f"{hx}  [{x0},{y0}]", font=f, fill=(190, 190, 200))
        tiles.append(t)
    cols = min(4, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tile + 8) + 8, rows * (tile + 42) + 8), (24, 24, 30))
    for i, t in enumerate(tiles):
        sheet.paste(t, (8 + (i % cols) * (tile + 8), 8 + (i // cols) * (tile + 42)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path)
    return path


def cmd_pick(a):
    """Median colour of small named patches: the local colours of objects, skin, hair, signs and UI, for swatches.
    A patch that holds two colours (an object and what is behind it) is split in two clusters: --main takes the
    colour of the bigger one. The box of every pick is drawn on an enlarged crop (zoom/pick_FRAME.png)."""
    W = a.work
    arr, _ = load_frame(W, a.ref, crop=False)
    H, Wd = arr.shape[:2]
    fid = ref_id(a.ref)
    db = jload(os.path.join(W, "measure.json"), {"frames": {}, "profiles": {}})
    picks = db.setdefault("picks", {})
    shown = []
    for s in a.patches:
        name, (x0, y0, x1, y1) = parse_patch(s)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(Wd, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            die(f"{name}: box is outside the {Wd}x{H} frame")
        px = arr[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
        lab = rgb_to_lab(px)
        med = np.median(px, 0)
        spread = float(np.sqrt(((lab - rgb_to_lab(med)) ** 2).sum(-1)).mean())
        cl = None
        if spread > 6 and len(px) >= 8:
            _, lbl = kmeans(lab, 2, seed=1)
            cnt = np.bincount(lbl, minlength=2)
            big = int(cnt.argmax())
            if cnt.min() > 0:
                m1, m2 = np.median(px[lbl == big], 0), np.median(px[lbl != big], 0)
                sp1 = float(np.sqrt(((lab[lbl == big] - rgb_to_lab(m1)) ** 2).sum(-1)).mean())
                cl = {"hex": hexc(m1), "share": round(float(cnt[big] / len(px)), 2), "other_hex": hexc(m2), "spread": round(sp1, 1)}
        use_main = bool(a.main and cl)
        col = np.array(hex_to_rgb(cl["hex"])) if use_main else med
        L, C, h = lch(col)
        sp = cl["spread"] if use_main else round(spread, 1)
        picks[f"{fid}:{name}"] = {"frame": fid, "name": name, "box": [x0, y0, x1, y1], "hex": hexc(col),
                                  "L": L, "C": C, "h": h, "spread": sp, **({"main_cluster_share": cl["share"]} if use_main else {}),
                                  **({"clusters": cl} if cl and not use_main else {})}
        shown.append((name, hexc(col), (x0, y0, x1, y1)))
        note = ""
        if use_main:
            note = f"   main colour of the patch ({cl['share']:.0%}; the rest is {cl['other_hex']})"
        elif sp > 12 and cl:
            note = (f"   mixed: {cl['share']:.0%} {cl['hex']} and {1 - cl['share']:.0%} {cl['other_hex']}: tighten the box, aim at the pure part, or use --main for the first colour;"
                    f" a thin line or bar: `name=x,y,0` is one pixel. `drop -w W pick {fid}:{name}` removes it")
        elif sp > 12:
            note = f"   mixed colours: tighten the box (the same name replaces this one), or `drop -w W pick {fid}:{name}`"
        print(f"{name:<18} {hexc(col)}  L*{L:5.1f}  C*{C:5.1f}  h {h:3.0f}°  spread ΔE {sp:4.1f}  [{x0},{y0},{x1},{y1}]{note}")
    jsave(os.path.join(W, "measure.json"), db)
    if not a.noshow and shown:
        print("   boxes on the frame:", pick_sheet(arr, shown, os.path.join(W, "zoom", f"pick_{fid}.png")),
              "(each patch drawn in magenta on an enlarged crop: check that it sits on the object)")


def cmd_drop(a):
    """Remove entries from measure.json: a pick that could not be made clean, a trial measurement.
    drop -w W pick G:sign_tea I:m_red"""
    path = os.path.join(a.work, "measure.json")
    db = jload(path, {})
    kinds = {"frame": "frames", "frames": "frames", "profile": "profiles", "profiles": "profiles", "pick": "picks", "picks": "picks",
             "light": "lights", "lights": "lights", "post": "post", "pixel": "pixel"}
    k = kinds.get(a.kind)
    if not k:
        die(f"kind must be one of frame, profile, pick, light, post, pixel (got '{a.kind}')")
    sec = db.get(k) or {}
    for n in a.names:
        if n in sec:
            del sec[n]
            print(f"dropped {k[:-1] if k.endswith('s') else k}: {n}")
        else:
            near = [x for x in sec if n in x][:6]
            print(f"not found: {n}   " + (f"similar: {', '.join(near)}" if near else f"existing: {', '.join(list(sec)[:8]) or '-'}"))
    jsave(path, db)


def cmd_light(a):
    """Falloff of a light pool or glow around (cx, cy): median L* and colour per ring, and the radii where it has
    fallen half and three quarters of the way to its surroundings."""
    W = a.work
    arr, _ = load_frame(W, a.ref, crop=False)
    H, Wd = arr.shape[:2]
    R = int(a.r or H * 0.3)
    step = max(2, round(R / 20))
    x0, x1, y0, y1 = max(0, a.cx - R), min(Wd, a.cx + R + 1), max(0, a.cy - R), min(H, a.cy + R + 1)
    sub = arr[y0:y1, x0:x1].astype(np.float64)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(xx - a.cx, yy - a.cy)
    L = rgb_to_lab(sub)[..., 0]
    rings = []
    for r0 in range(0, R, step):
        m = (r >= r0) & (r < r0 + step)
        if m.sum() >= 3:
            rings.append([r0 + step / 2, round(float(np.median(L[m])), 1), hexc(np.median(sub[m], 0))])
    if len(rings) < 4:
        die("too few rings: move the centre inside the frame or raise --r")
    peak, floor = rings[0][1], float(np.median([v for _, v, _ in rings[-3:]]))

    def reach(frac):
        target = floor + (peak - floor) * frac
        return next((rr for rr, v, _ in rings if v <= target), None)

    half, quarter = reach(0.5), reach(0.25)
    Ls = [v for _, v, _ in rings]
    lows = np.minimum.accumulate(Ls)
    rise = [(rr, v, lows[i - 1]) for i, (rr, v, _) in enumerate(rings) if i >= 2 and v - lows[i - 1] > 3]
    quarter_ok = not rise and (quarter is None or quarter < rings[-1][0] - step * 1.5)
    name = a.name or f"{ref_id(a.ref)}@{a.cx},{a.cy}"
    pct = lambda v: None if v is None else round(v * 100 / H, 1)
    e = {"frame": ref_id(a.ref), "centre": [a.cx, a.cy], "centre_hex": rings[0][2], "centre_L": peak,
         "surround_hex": rings[-1][2], "surround_L": round(floor, 1), "half_px": half, "half_pct_h": pct(half),
         "quarter_px": quarter, "quarter_pct_h": pct(quarter), "quarter_ok": bool(quarter_ok), "rings": rings}
    db = jload(os.path.join(W, "measure.json"), {"frames": {}, "profiles": {}})
    db.setdefault("lights", {})[name] = e
    jsave(os.path.join(W, "measure.json"), db)
    print(f"{name}: centre {e['centre_hex']} L*{peak}  →  surroundings {e['surround_hex']} L*{floor:.1f}")
    print(f"   half-way at r = {half} px ({pct(half)}% of frame height); three-quarters at r = {quarter} px ({pct(quarter)}%)"
          + ("" if quarter_ok else "  (not reliable, see below)"))
    print("   rings (r px: L* hex) " + "  ".join(f"{rr:.0f}:{v:.0f} {hx}" for rr, v, hx in rings[::2]))
    top = max(rings, key=lambda x: x[1])
    if top[1] > peak + 3:
        print(f"   warning: the brightest ring is at r = {top[0]:.0f} px (L*{top[1]}), not the centre: put the centre on the light")
    elif peak - floor < 8:
        print("   warning: the glow is less than 8 L* above its surroundings: probably not a light source")
    if rise:
        rr, v, lo = rise[0]
        print(f"   warning: the rings get brighter again at r = {rr:.0f} px (L*{v:.0f} after L*{lo:.0f} nearer the light): another light or a bright "
              "surface is in the way, so the three-quarter radius means little. Quote only the half radius, or measure with a smaller --r.")
    elif not quarter_ok:
        print(f"   warning: the fall-off is not finished within r = {rings[-1][0]:.0f} px: the three-quarter radius is only the edge of the measured area. "
              "Raise --r, or quote only the half radius.")
    if floor > 60:
        print(f"   note: the surroundings are bright (L*{floor:.0f}): against a bright wall or sky a glow has little fall-off to measure, so the radii are rough")


def gblur(x, s):
    r = max(1, int(round(3 * s)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
    k /= k.sum()
    h, w = x.shape
    p = np.pad(x, ((r, r), (0, 0)), mode="edge")
    y = sum(k[i] * p[i:i + h] for i in range(2 * r + 1))
    p = np.pad(y, ((0, 0), (r, r)), mode="edge")
    return sum(k[i] * p[:, i:i + w] for i in range(2 * r + 1))


def rb_offset(arr, xs, ys, thr=24, win=4, minn=40):
    """Lateral chromatic aberration in a region. For every strong vertical-ish edge (a local maximum of the green
    channel's horizontal gradient) it takes the sub-pixel position of the red edge minus that of the blue edge; the
    result is the median over all such edges, in px (positive: the red edge lies to the right of the blue one).
    Only strong edges vote, so smooth shading and grain, whose red and blue gradients always agree, cannot hide a
    fringe that is only a pixel or two wide. None when the region has fewer than `minn` usable edges."""
    sub = arr[ys, xs].astype(np.float64)
    g = np.diff(sub, axis=1)
    gg = g[..., 1]
    mag = np.abs(gg)
    cand = (mag[:, 1:-1] >= mag[:, :-2]) & (mag[:, 1:-1] > mag[:, 2:]) & (mag[:, 1:-1] >= thr)
    yy, xx = np.nonzero(cand)
    xx = xx + 1
    ok = (xx >= win) & (xx < g.shape[1] - win)
    yy, xx = yy[ok], xx[ok]
    if len(yy) < minn:
        return None
    pos = np.arange(-win, win + 1)
    seg = np.clip(g[yy[:, None], xx[:, None] + pos[None, :], :] * np.sign(gg[yy, xx])[:, None, None], 0, None)    # gradient in the edge's own direction
    sr, sb = seg[..., 0], seg[..., 2]
    ok = (sr.sum(1) >= thr) & (sb.sum(1) >= thr)
    if ok.sum() < minn:
        return None
    pr = (sr[ok] * pos).sum(1) / sr[ok].sum(1)
    pb = (sb[ok] * pos).sum(1) / sb[ok].sum(1)
    return round(float(np.median(pr - pb)), 2)


def split_map(arr, nx=6, ny=4, maxs=16, min_peak=0.3):
    """Where the colour channels are displaced against each other, cell by cell. In every cell of an nx x ny grid the
    red and the blue channel (high-passed) are shifted against the green one, +-maxs px, for the best match: red
    minus blue in px (positive: the red image lies to the right of the blue one), None where the cell has too little
    structure for a clear peak. A hit, glitch or transition effect splits the channels by many pixels, usually more
    toward the sides, which the strips at the edge (rb_offset: edges of one or two pixels) cannot see; a cell that
    holds a static HUD correlates at zero shift and shows no split."""
    h, w = arr.shape[:2]
    out = []
    for j in range(ny):
        row = []
        for i in range(nx):
            sub = arr[j * h // ny:(j + 1) * h // ny, i * w // nx:(i + 1) * w // nx].astype(np.float64)
            ch = [sub[..., c] - gblur(sub[..., c], 3.0) for c in range(3)]
            G = ch[1]
            sh, gain = [], 0.0
            for c in (0, 2):
                C = ch[c]
                v = []
                for s_ in range(-maxs, maxs + 1):
                    a_, b_ = (C[:, s_:], G[:, :G.shape[1] - s_]) if s_ >= 0 else (C[:, :s_], G[:, -s_:])
                    v.append(float((a_ * b_).sum() / (math.sqrt(float((a_ ** 2).sum()) * float((b_ ** 2).sum())) + 1e-9)))
                v = np.array(v)
                k = int(v.argmax())
                if v[k] < min_peak:
                    sh = None
                    break
                d = 0.0
                if 0 < k < len(v) - 1 and (v[k - 1] - 2 * v[k] + v[k + 1]) < 0:
                    d = 0.5 * (v[k - 1] - v[k + 1]) / (v[k - 1] - 2 * v[k] + v[k + 1])
                sh.append(k - maxs + d)
                gain += v[k] - v[maxs]    # how much better than "no shift" the best shift matches
            r = None if sh is None else round(float(sh[0] - sh[1]), 1)
            if r is not None and abs(r) >= 2.5 and gain < 0.12:
                r = None    # flat or repetitive structure (a HUD, a pattern, a plain wall): no clear peak
            row.append(r)
        out.append(row)
    return out


def grain(arr, tile=32):
    """Noise sigma (8-bit levels) in the flattest mid-tone tiles, by Immerkaer's estimator: a 3x3 mask that cancels
    smooth shading and most texture, so what is left is grain, dither or noise that survived compression."""
    g = arr.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    N = np.abs(g[:-2, :-2] - 2 * g[:-2, 1:-1] + g[:-2, 2:] - 2 * g[1:-1, :-2] + 4 * g[1:-1, 1:-1]
               - 2 * g[1:-1, 2:] + g[2:, :-2] - 2 * g[2:, 1:-1] + g[2:, 2:])
    base = gblur(g, 2.0)[1:-1, 1:-1]
    h, w = N.shape
    rows = []
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            bt = base[y:y + tile, x:x + tile]
            if 16 <= bt.mean() <= 240:
                rows.append((float(bt.std()), math.sqrt(math.pi / 2) * float(N[y:y + tile, x:x + tile].mean()) / 6, float(bt.mean())))
    if len(rows) < 8:
        return None
    rows.sort()
    flat = rows[:max(8, len(rows) // 7)]
    sig = float(np.median([n for _, n, _ in flat]))
    dark = [n for _, n, m in flat if m < 90]
    light = [n for _, n, m in flat if m >= 90]
    return {"sigma": round(sig, 2), "tiles": len(flat),
            "dark": round(float(np.median(dark)), 2) if dark else None, "light": round(float(np.median(light)), 2) if light else None}


def cmd_post(a):
    """Lens and post effects that colour statistics miss: lateral chromatic aberration (red/blue edge offset at the
    left, centre and right of the frame, and in the outermost 6% strips, where a lens fringe is strongest), a map of
    channel splits of several pixels (hit and glitch effects) and grain (noise in flat areas)."""
    W = a.work
    db = jload(os.path.join(W, "measure.json"), {"frames": {}, "profiles": {}})
    f2 = lambda v: "  n/a" if v is None else f"{v:+.2f}"
    side = lambda v: "" if v is None else (" (none)" if abs(v) < 0.3 else " (red fringe outward)" if v > 0 else " (blue fringe outward)")
    for ref in a.refs:
        arr, _ = load_frame(W, ref, crop=not a.nocrop)
        h, w = arr.shape[:2]
        ys = slice(int(h * 0.1), int(h * 0.9))
        bw = max(32, int(w * 0.15))
        off = [rb_offset(arr, slice(x, x + bw), ys) for x in (0, (w - bw) // 2, w - bw)]
        ca = None if off[0] is None or off[2] is None else round((off[2] - off[0]) / 2, 2)
        ew = max(24, int(w * 0.06))
        edge = [rb_offset(arr, slice(0, ew), ys), rb_offset(arr, slice(w - ew, w), ys)]
        ca_edge = None if None in edge else round((edge[1] - edge[0]) / 2, 2)
        gr = grain(arr)
        fid = ref_id(ref)
        db.setdefault("post", {})[fid] = {"rb_offset_left_centre_right": off, "ca_px": ca, "rb_offset_edge_left_right": edge,
                                          "ca_edge_px": ca_edge, "width": w, "grain": gr}
        print(f"── {fid}  {w}x{h}")
        print(f"   red-vs-blue edge offset (px)   left 15% {f2(off[0])}   centre {f2(off[1])}   right 15% {f2(off[2])}   →  lateral CA {f2(ca)} px{side(ca)}")
        print(f"   outermost 6% strips            left {f2(edge[0])}   right {f2(edge[1])}   →  edge CA {f2(ca_edge)} px{side(ca_edge)}")
        if None in off or None in edge:
            print("   n/a: fewer than 40 sharp vertical edges in that part (soft, blurred or flat picture): look at the sides with `zoom` instead")
        sm = split_map(arr)
        big = [(abs(v), v, i, j) for j, row in enumerate(sm) for i, v in enumerate(row) if v is not None and abs(v) >= 2.5]
        if big:
            print("   channel-split map (red minus blue, px; 6x4 cells, left to right, top to bottom; n/a = too little structure):")
            for row in sm:
                print("      " + " ".join("   n/a" if v is None else f"{v:+6.1f}" for v in row))
            top = max(big)
            print(f"   {len(big)} cells split by 2.5 px or more, the largest {top[1]:+.1f} px at x {top[2] * w // 6}–{(top[2] + 1) * w // 6}, y {top[3] * h // 4}–{(top[3] + 1) * h // 4}: "
                  "that is no lens fringe but a channel split of the picture (a hit, glitch or transition effect), usually growing toward the sides; static HUD parts stay at 0. "
                  "Look at it with `zoom`, and describe it as an effect on this frame, with the frame, not as the game's lens")
        db["post"][fid]["split_map"] = sm
        db["post"][fid]["split_max_px"] = max((v for row in sm for v in row if v is not None), key=abs, default=None)
        if not big and ca_edge is not None and abs(ca_edge) >= 0.3:
            print("   a lens fringe grows toward the sides: quote the centre and the outermost values, and confirm it on a `zoom` of a side")
        if gr:
            print(f"   grain σ {gr['sigma']} levels in the {gr['tiles']} flattest tiles (dark {gr['dark']}, light {gr['light']}):"
                  f" < 0.5 clean or erased by compression, 0.8–1.5 light grain, > 2 strong grain or dither")
        else:
            print("   grain: not enough flat mid-tone areas")
    jsave(os.path.join(W, "measure.json"), db)


# ───────────────────────── audio ─────────────────────────
SR = 22050
PC = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PAGE_CMAP = np.array([[0.0, 11, 13, 24], [0.35, 38, 34, 82], [0.62, 125, 72, 90], [0.82, 243, 166, 91], [1.0, 239, 229, 201]])


def load_audio(path, sr=SR):
    out = run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]).stdout
    return np.frombuffer(out, np.float32).copy()    # float32 keeps an hour of audio near 300 MB


def stft_mag(y, n_fft, hop):
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    n = 1 + (len(y) - n_fft) // hop
    win = np.hanning(n_fft).astype(np.float32)
    out = np.empty((n_fft // 2 + 1, n), np.float32)
    for s in range(0, n, 512):
        e = min(n, s + 512)
        idx = np.arange(n_fft)[None, :] + hop * np.arange(s, e)[:, None]
        out[:, s:e] = np.abs(np.fft.rfft(y[idx].astype(np.float32) * win, axis=1)).T
    return out


def per_second_db(y, sr=SR):
    n = len(y) // sr
    r = []
    for i in range(n):
        seg = y[i * sr:(i + 1) * sr]
        r.append(round(float(20 * np.log10(np.sqrt(np.mean(np.square(seg, dtype=np.float64))) + 1e-10)), 2))
    return r


def segments_from_db(db, thr=6.0, win=5, min_len=4):
    e = 10 ** (np.array(db) / 10)
    n = len(e)
    if n < 2 * win + 2:
        return [[0, n]]
    diff = np.zeros(n)
    for i in range(win, n - win):
        a_ = 10 * np.log10(e[i - win:i].mean() + 1e-12)
        b_ = 10 * np.log10(e[i:i + win].mean() + 1e-12)
        diff[i] = b_ - a_
    cuts = []
    for i in range(win, n - win):
        if abs(diff[i]) >= thr and abs(diff[i]) == np.abs(diff[max(0, i - 3):i + 4]).max():
            if not cuts or i - cuts[-1] >= min_len:
                cuts.append(i)
    b = [0] + cuts + [n]
    return [[b[i], b[i + 1]] for i in range(len(b) - 1) if b[i + 1] - b[i] >= 1]


def silence_gaps(y, sr=SR, min_dur=0.25):
    """Near-silent stretches, relative to the clip's own level (quiet recordings are common)."""
    hop = int(sr * 0.05)
    n = len(y) // hop
    if n == 0:
        return []
    db = 20 * np.log10(np.sqrt(np.square(y[:n * hop].reshape(n, hop), dtype=np.float64).mean(1)) + 1e-10)
    q = db < min(-50.0, float(np.median(db)) - 30)
    gaps, s = [], None
    for i, v in enumerate(np.r_[q, False]):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if (i - s) * 0.05 >= min_dur:
                gaps.append([round(s * 0.05, 2), round(i * 0.05, 2)])
            s = None
    return gaps


def tempo_candidates(y, sr=SR):
    hop = 512
    S = stft_mag(y, 2048, hop)
    flux = np.maximum(0, np.diff(np.log1p(10 * S), axis=1)).sum(0)
    if len(flux) < 64:
        return [], 0.0
    flux = flux - np.convolve(flux, np.ones(16) / 16, mode="same")
    flux = np.maximum(flux, 0)
    fr = sr / hop
    onset_rate = float(((flux > flux.mean() + flux.std()) & (np.r_[0, np.diff(np.sign(np.diff(flux))), 0] < 0)).sum() / (len(flux) / fr))
    f = flux - flux.mean()
    ac = np.correlate(f, f, mode="full")[len(f) - 1:]
    ac = ac / (ac[0] + 1e-12)
    out = []
    lags = np.arange(int(fr * 60 / 220), int(fr * 60 / 45) + 1)
    bpm = 60 * fr / lags
    prior = np.exp(-0.5 * (np.log2(bpm / 120.0)) ** 2)
    score = ac[lags] * prior
    for i in np.argsort(-score):
        b = float(bpm[i])
        if all(abs(b - o[0]) / o[0] > 0.06 for o in out):
            out.append((round(b, 1), round(float(ac[lags[i]]), 3), round(float(score[i]), 3)))
        if len(out) == 3:
            break
    return out, round(onset_rate, 2)


def key_estimate(y, sr=SR):
    n_fft, hop = 8192, 4096
    S = stft_mag(y, n_fft, hop)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    m = (freqs >= 55) & (freqs <= 2000)
    pcs = (np.round(69 + 12 * np.log2(freqs[m] / 440.0)).astype(int)) % 12
    E = S[m].mean(1)
    chroma = np.array([E[pcs == k].sum() for k in range(12)])
    if chroma.sum() <= 0:
        return [], chroma.tolist()
    chroma = chroma / chroma.max()
    res = []
    for k in range(12):
        for prof, mode in ((KS_MAJ, "major"), (KS_MIN, "minor")):
            r = np.corrcoef(chroma, np.roll(prof, k))[0, 1]
            res.append((round(float(r), 3), f"{PC[k]} {mode}"))
    res.sort(reverse=True)
    return [[k, r] for r, k in res[:3]], [round(float(v), 2) for v in chroma]


def scale_info(keys, chroma):
    """The 7 strongest pitch classes, listed from the best key's tonic; for a minor key the strength of the flat seventh
    against the leading tone (> 1.15: natural minor; < 0.87: harmonic minor or a raised seventh); the minor third against
    the major third (the test that settles major or minor); and `key_doubt`: reasons not to trust the label (the two
    best keys score within 0.03 of each other, which relative major and minor keys do as they share their notes, or the
    third disagrees with the mode)."""
    if not keys or len(chroma) != 12:
        return {}
    tonic, mode = keys[0][0].split()
    t = PC.index(tonic)
    top = sorted(range(12), key=lambda k: -chroma[k])[:7]
    out = {"scale_notes": " ".join(PC[k] for k in sorted(top, key=lambda k: (k - t) % 12))}
    if mode == "minor":
        out["b7_over_7"] = round(chroma[(t + 10) % 12] / max(chroma[(t + 11) % 12], 1e-6), 2)
    m3, M3 = chroma[(t + 3) % 12], chroma[(t + 4) % 12]
    out["m3_over_M3"] = round(m3 / max(M3, 1e-6), 2)
    doubt = []
    if len(keys) > 1 and keys[0][1] - keys[1][1] < 0.03:
        doubt.append(f"{keys[0][0]} {keys[0][1]} and {keys[1][0]} {keys[1][1]} score alike")
    if mode == "major" and m3 > 1.05 * M3:
        doubt.append(f"the minor third {PC[(t + 3) % 12]} is stronger than the major third {PC[(t + 4) % 12]} (x{out['m3_over_M3']})")
    elif mode == "minor" and M3 > 1.05 * m3:
        doubt.append(f"the major third {PC[(t + 4) % 12]} is stronger than the minor third {PC[(t + 3) % 12]} (x{round(1 / max(out['m3_over_M3'], 1e-6), 2)})")
    else:
        third = PC[(t + (4 if mode == "major" else 3)) % 12]
        if third not in out["scale_notes"].split():
            doubt.append(f"the {mode} third {third} is not among the 7 strongest notes")
    if doubt:
        out["key_doubt"] = doubt
    return out


def fmt_tempo(c):
    """Best tempo candidate with its pulse strength (autocorrelation of the onset curve), `weak` below 0.15: no clear
    pulse, so no tempo should be quoted. Candidates are ranked by pulse x a mild preference for 120 bpm, so the
    second one may show the larger pulse."""
    if not c:
        return "-"
    s_ = f"{c[0][0]:g} bpm (pulse {c[0][1]:.2f})" + ("  WEAK: no clear pulse, quote no tempo" if c[0][1] < 0.15 else "")
    return s_ + (f"; next {c[1][0]:g} ({c[1][1]:.2f})" if len(c) > 1 else "")


def fmt_key(o, n=2):
    k = o.get("key_candidates") or []
    if not k:
        return "-"
    s_ = " / ".join(f"{x[0]} ({x[1]})" for x in k[:n])
    return s_ + ("  ⚠ " + "; ".join(o["key_doubt"]) + ": judge the mode by ear or write both" if o.get("key_doubt") else "")


def percussive_ratio(y, sr=SR, max_s=40):
    if len(y) > max_s * sr:
        c = len(y) // 2
        y = y[c - max_s * sr // 2:c + max_s * sr // 2]
    S = stft_mag(y, 2048, 512)
    if S.shape[1] < 20:
        return None
    k = 17
    pt = np.pad(S, ((0, 0), (k // 2, k // 2)), mode="edge")
    H = np.median(np.lib.stride_tricks.sliding_window_view(pt, k, axis=1), axis=-1)
    pf = np.pad(S, ((k // 2, k // 2), (0, 0)), mode="edge")
    P = np.median(np.lib.stride_tricks.sliding_window_view(pf, k, axis=0), axis=-1)
    mp = P ** 2 / (H ** 2 + P ** 2 + 1e-12)
    return round(float((mp * S ** 2).sum() / ((S ** 2).sum() + 1e-12)), 3)


def band_shares(y, sr=SR, max_s=60):
    if len(y) > max_s * sr:
        c = len(y) // 2
        y = y[c - max_s * sr // 2:c + max_s * sr // 2]
    S = stft_mag(y, 2048, 1024) ** 2
    f = np.fft.rfftfreq(2048, 1 / sr)
    tot = S.sum() + 1e-12
    bands = {"<150Hz": (20, 150), "150–500": (150, 500), "0.5–2k": (500, 2000), "2–6k": (2000, 6000), ">6k": (6000, sr / 2)}
    shares = {k: round(float(S[(f >= lo) & (f < hi)].sum() / tot), 3) for k, (lo, hi) in bands.items()}
    centroid = float((f[:, None] * S).sum() / tot)
    return shares, round(centroid)


def middle(y, seconds, sr=SR):
    n = int(seconds * sr)
    if len(y) <= n:
        return y
    c = len(y) // 2
    return y[c - n // 2:c + n // 2]


def features(y, sr=SR):
    if len(y) < sr * 2:
        return {"note": "too short"}
    tempo, onset_rate = tempo_candidates(y[:sr * 90] if len(y) > sr * 90 else y, sr)
    keys, chroma = key_estimate(middle(y, 240, sr), sr)
    shares, centroid = band_shares(y, sr)
    return {"tempo_bpm_candidates": tempo, "onsets_per_s": onset_rate, "key_candidates": keys,
            "chroma_C_to_B": chroma, **scale_info(keys, chroma), "percussive_ratio": percussive_ratio(y, sr), "band_energy": shares,
            "centroid_hz": centroid, "db": round(float(10 * np.log10(np.mean(np.square(y, dtype=np.float64)) + 1e-12)), 1)}


def spectrum_image(W, code, y, t0, dur, fmax, sr=SR, scale="log", fmin=40.0):
    """Spectrogram strip in the page palette. The log axis gives each octave the same height, so bass and kick
    (where battle and calm music differ most) are as readable as the highs."""
    a_ = int(t0 * sr)
    seg = y[a_:a_ + int(dur * sr)]
    n_fft = 4096 if scale == "log" else 2048
    S = stft_mag(seg, n_fft, 256)
    S = 20 * np.log10(S / (S.max() + 1e-12) + 1e-10)
    fmax = min(fmax, sr / 2)
    if scale == "log":
        fb = np.fft.rfftfreq(n_fft, 1 / sr)
        rows = fmin * (fmax / fmin) ** (np.arange(330) / 329.0)
        S = np.stack([np.interp(rows, fb, S[:, j]) for j in range(S.shape[1])], 1)[::-1]
    else:
        nb = int(fmax / (sr / 2) * S.shape[0])
        S = S[:nb][::-1]
    v = np.clip((S + 75) / 75, 0, 1)
    rgb = np.stack([np.interp(v, PAGE_CMAP[:, 0], PAGE_CMAP[:, i]) for i in (1, 2, 3)], -1).astype(np.uint8)
    name = f"spec_{code}_{int(t0)}"
    os.makedirs(os.path.join(W, "img"), exist_ok=True)
    Image.fromarray(rgb).resize((1250, 330), Image.BILINEAR).save(os.path.join(W, "img", name + ".jpg"), quality=86)
    man = jload(os.path.join(W, "img", "manifest.json"), {})
    man[name] = {"file": name + ".jpg", "w": 1250, "h": 330, "kind": "spectrum", "code": code, "t0": t0, "t1": t0 + dur,
                 "fmax": fmax, "scale": scale, "fmin": fmin if scale == "log" else 0}
    jsave(os.path.join(W, "img", "manifest.json"), man)
    return name


def cmd_audio(a):
    W = a.work
    os.makedirs(os.path.join(W, "audio"), exist_ok=True)
    its = [it for it in inputs(W)["items"] if it["kind"] == "audio" or it.get("has_audio")]
    if a.codes:
        its = [it for it in its if it["code"] in a.codes.split(",")]
    elif a.spectrum:    # a spectrum call analyses only the inputs it draws, and only if they were never analysed
        its = [it for it in its if any(sp.strip().startswith(it["code"] + "@") for sp in a.spectrum)]
    if not its and not a.spectrum:
        print("no input has an audio track")
        return
    for it in its:
        code = it["code"]
        if a.spectrum and not a.ranges and os.path.exists(os.path.join(W, "audio", f"{code}.json")):
            print(f"{code}: analysis already in audio/{code}.json, making spectra only (run without --spectrum to redo it)")
            continue
        y = load_audio(it["path"])
        if len(y) < SR:
            continue
        db = per_second_db(y)
        jsave(os.path.join(W, "audio", f"{code}_loud.json"), [{"t": i, "db": v} for i, v in enumerate(db)])
        segs = segments_from_db(db, thr=a.thr)
        out = {"code": code, "duration": round(len(y) / SR, 2), "overall": features(y), "segments": [], "gaps": silence_gaps(y)}
        e = 10 ** (np.array(db) / 10)
        for s0, s1 in segs:
            lvl = round(float(10 * np.log10(e[s0:s1].mean() + 1e-12)), 1)
            f = features(y[s0 * SR:s1 * SR]) if (s1 - s0) >= 4 else {}
            out["segments"].append({"a": s0, "b": s1, "db": lvl, **f})
        if a.ranges:
            out["ranges"] = []
            for r in a.ranges.split(","):
                r0, r1 = [parse_time(x) for x in (r.split("~") if "~" in r else r.split("-"))]
                out["ranges"].append({"a": r0, "b": r1, **features(y[int(r0 * SR):int(r1 * SR)])})
        jsave(os.path.join(W, "audio", f"{code}.json"), out)
        o = out["overall"]
        print(f"══ {code} ({mmss(out['duration'])})  mean {o.get('db')} dBFS  tempo {fmt_tempo(o.get('tempo_bpm_candidates'))}  key {fmt_key(o)}")
        print(f"   overall bands {o.get('band_energy')} centroid {o.get('centroid_hz')} Hz  percussive {o.get('percussive_ratio')}  onsets/s {o.get('onsets_per_s')}")
        print(f"   overall notes {o.get('scale_notes', '-')}" + (f"  ♭7/7 {o['b7_over_7']}" if "b7_over_7" in o else "") + (f"  minor/major third {o['m3_over_M3']}" if "m3_over_M3" in o else ""))
        st = 5 if len(db) <= 300 else int(math.ceil(len(db) / 300) * 5)
        print(f"   loudness per {st} s (dB): " + "  ".join(f"{mmss(i)} {10 * np.log10(e[i:i + st].mean() + 1e-12):.0f}" for i in range(0, len(db), st)))
        for s in out["segments"]:
            kk = fmt_key(s, 1) + f"  notes {s.get('scale_notes')}" + (f"  ♭7/7 {s['b7_over_7']}" if "b7_over_7" in s else "") if s.get("key_candidates") else "-"
            print(f"   {mmss(s['a']):>6}–{mmss(s['b']):<6} {s['db']:6.1f} dB  tempo {fmt_tempo(s.get('tempo_bpm_candidates'))}  key {kk}  perc {s.get('percussive_ratio', '-')}  >6k {s.get('band_energy', {}).get('>6k', '-')}")
        if out["gaps"]:
            print(f"   silences ≥0.25 s: {out['gaps'][:12]}{' …' if len(out['gaps']) > 12 else ''}")
        for r in out.get("ranges", []):
            print(f"   range {mmss(r['a'])}–{mmss(r['b'])}: {r.get('db')} dB tempo {fmt_tempo(r.get('tempo_bpm_candidates'))} key {fmt_key(r)} perc {r.get('percussive_ratio')}")
    for spec in a.spectrum or []:
        m = re.fullmatch(r"([A-Z]{1,2})@([0-9:.]+)\+([0-9.]+)(?:/([0-9]+))?", spec.strip())
        if not m:
            die(f"--spectrum {spec!r}: use CODE@START+SECONDS[/FMAX], e.g. B@3:25+10/9000")
        code, t0, dur = m.group(1), parse_time(m.group(2)), float(m.group(3))
        fmax = float(m.group(4)) if m.group(4) else 9000
        y = load_audio(item(W, code)["path"])
        print("spectrum image:", spectrum_image(W, code, y, t0, dur, fmax, scale="linear" if a.linear else "log"),
              f"({'linear' if a.linear else 'log'} frequency axis, {'0' if a.linear else '40 Hz'}–{fmax:g} Hz)")
    print("Numbers are estimates from compressed, mixed audio (music + voices + SFX). Tag them 实测 and keep the caveat; "
          "instrument names need a classifier (模型识别) or a source (资料).")


# ───────────────────────── facts: every measured number on one screen ─────────────────────────
def cmd_facts(a):
    """All measurements as one compact list, to check the prose against; with content.json also the colour check."""
    W = a.work
    db = jload(os.path.join(W, "measure.json"), {})
    o = []
    if db.get("frames"):
        o.append("FRAMES (measure): L* mean (p5–p95) · C* mean · shadow / mid / light % and main colour · vignette corners/centre as L* (linear Y) · accents")
        for n, r in db["frames"].items():
            tiers = " · ".join(f"{x['tier'][:2]} {x['pct']:.0f}% {x['colors'][0][0] if x['colors'] else '-'}" + (f" h{x['hue']:.0f}°" if x.get("hue") is not None else "") for x in r["tiers"])
            vg = r.get("vignette", {}) if not r.get("region") or r.get("region") == r.get("crop") else {}
            vig = f"L* {vg['L_ratio']} (Y {vg['Y_ratio']})" if vg.get("L_ratio") is not None else None
            acc = " ".join(c["hex"] for c in r.get("accents", [])[:3]) or "-"
            o.append(f"  {n:<30} L* {r['L']['mean']:.0f} ({r['L']['p5']:.0f}–{r['L']['p95']:.0f})  C* {r['chroma']['mean']:.0f}  {tiers}"
                     + (f"  vig {vig}" if vig is not None else "") + f"  acc {acc}  hi {' '.join(c['hex'] for c in r.get('highlights', [])[:2]) or '-'}")
    if db.get("profiles"):
        o.append("PROFILES (profile): sharp band % from top · top / bottom sharpness · vignette corners/centre as L* (linear Y) · frames")
        for n, r in db["profiles"].items():
            vl = f"L* {r['vignette_L']} (Y {r.get('vignette_Y')})" if r.get("vignette_L") is not None else f"Y {r.get('vignette_Y')} (re-run profile for L*)"
            span = f"{mmss(r.get('t0', 0))}–{mmss(r.get('t1', 0))}" if r.get("t1") else "stills"
            o.append(f"  {n:<30} band {r.get('sharp_band_pct')}  top {r.get('top')} bottom {r.get('bottom')}  vig {vl}  "
                     f"{r.get('frames')} frames {span}" + ("  (with exclusions)" if r.get("exclude") else ""))
    if db.get("picks"):
        o.append("PICKS (pick): hex · L* C* h · spread")
        for n, r in db["picks"].items():
            o.append(f"  {n:<30} {r['hex']}  L* {r['L']:.0f} C* {r['C']:.0f} h {r['h']:.0f}°  spread {r['spread']}" + ("  ⚠ mixed colours" if r["spread"] > 12 else ""))
    if db.get("lights"):
        o.append("LIGHTS (light): centre → surroundings · radius where it has fallen half / three quarters")
        for n, r in db["lights"].items():
            o.append(f"  {n:<30} {r['centre_hex']} L*{r['centre_L']:.0f} → {r['surround_hex']} L*{r['surround_L']:.0f}  "
                     f"half {r['half_px']} px ({r['half_pct_h']}% H)  3/4 {r['quarter_px']} px ({r['quarter_pct_h']}% H)"
                     + ("" if r.get("quarter_ok", True) else "  ⚠ 3/4 not reliable"))
    if db.get("post"):
        o.append("POST (post): lateral CA px over the frame (and over the outer 6% strips) · grain σ")
        for n, r in db["post"].items():
            g = r.get("grain") or {}
            fm = lambda v: "n/a" if v is None else f"{v:+.2f}"
            sp = r.get("split_max_px")
            o.append(f"  {n:<30} CA {fm(r.get('ca_px'))} px" + (f" (outermost edges {fm(r['ca_edge_px'])})" if "ca_edge_px" in r else " (re-run post for the edge value)")
                     + (f"  channel split up to {sp:+.1f} px ⚠ effect, not lens" if sp is not None and abs(sp) >= 2.5 else "") + f"  grain σ {g.get('sigma', '-')}")
    if db.get("pixel"):
        o.append("PIXEL (pixel): candidate periods in x / y, screen px per art px, with prominence ×: a clear pixel size shows the same period in x and y (sprites); textures and bricks do not")
        for n, r in db["pixel"].items():
            cand = lambda c: " ".join(f"{p:g}(×{w:g})" for p, w in (c or [])) or "-"
            o.append(f"  {n:<30} x {cand(r.get('x'))}  y {cand(r.get('y'))}")
    bands = lambda be: " ".join(f"{k} {v}" for k, v in (be or {}).items()) or "-"
    for f in sorted(glob.glob(os.path.join(W, "audio", "*.json"))):
        if f.endswith("_loud.json"):
            continue
        d = jload(f)
        ov = d.get("overall", {})
        o.append(f"AUDIO {d['code']} ({mmss(d['duration'])}): mean {ov.get('db')} dB  tempo {fmt_tempo(ov.get('tempo_bpm_candidates'))}  key {fmt_key(ov)}  notes {ov.get('scale_notes', '-')}"
                 f"  centroid {ov.get('centroid_hz')} Hz  percussive {ov.get('percussive_ratio')}")
        o.append(f"      bands (share of energy) {bands(ov.get('band_energy'))}")
        for sg in d.get("segments", []) + [dict(r, range=True) for r in d.get("ranges", [])]:
            o.append(f"  {'range' if sg.get('range') else 'seg':<5} {mmss(sg['a'])}–{mmss(sg['b'])}  {sg.get('db', '-')} dB  "
                     f"tempo {fmt_tempo(sg.get('tempo_bpm_candidates'))}  key {fmt_key(sg, 2) if sg.get('key_candidates') else '-'}  perc {sg.get('percussive_ratio', '-')}"
                     f"  bands {bands(sg.get('band_energy'))}")
    if a.content:
        for sec in (jload(a.content) or {}).get("sections", []):
            for b in walk_blocks(sec.get("blocks", [])):
                if b.get("type") != "loudness":
                    continue
                lm = loud_means(W, b)
                if lm is None:
                    continue
                segs, names = lm
                o.append(f"LOUDNESS in content.json ({sec.get('id', '?')}, series {b['series']}; energy mean, the numbers the chart prints: quote these, not the auto-segment dB above)")
                o.append("  per segment: " + " · ".join(f"{sg.get('name', '')} {mmss(sg['a'])}–{mmss(sg['b'])} {v:.1f}" for sg, v in zip(b.get("segments", []), segs) if v is not None))
                o.append("  per name:    " + " · ".join(f"{k} {v:.1f} dB ({c} seg, {n} s)" for k, (v, c, n) in names.items()))
    if not o:
        o.append("nothing measured yet")
    print("\n".join(o))
    if a.content:
        content = jload(a.content)
        tpl = open(os.path.join(HERE, "amber.css"), encoding="utf-8").read() + open(os.path.join(HERE, "amber_ext.css"), encoding="utf-8").read()
        n, bad, near = colour_check(W, content, os.path.dirname(os.path.abspath(a.content)), tpl)
        print(f"COLOURS: {n - len(bad) - len(near)} of {n} hex values in the document match a measured colour (ΔE 2 or less); "
              f"{len(near)} are close (ΔE 2–6: check it is the same thing); {len(bad)} match nothing measured")
        for h, nr, d, src_, where in sorted(near, key=lambda x: -x[2]):
            print(f"  close   {h} in {', '.join(where)}: ≈ {nr} (ΔE {d}) measured as {src_}")
        for h, nr, d, where in bad:
            print(f"  none    {h} in {', '.join(where)}: " + (f"nearest measured {nr} (ΔE {d})" if nr else "nothing measured"))


# ───────────────────────── crops: plates and specimens ─────────────────────────
def chips_of(arr, k=6, n=4):
    a = np.asarray(arr, dtype=np.float64).reshape(-1, 3)
    if len(a) > 20000:
        a = a[np.random.default_rng(1).choice(len(a), 20000, replace=False)]
    c, lab = kmeans(a, k, seed=1)
    share = np.bincount(lab, minlength=len(c)) / len(a)
    sat = c.max(1) - c.min(1)
    order = np.argsort(-(share * (0.35 + sat / 255)))
    return [hexc(c[i]) for i in order[:n]]


def cmd_crops(a):
    W = a.work
    spec = jload(a.spec)
    out_dir = os.path.join(W, "img")
    os.makedirs(out_dir, exist_ok=True)
    man = jload(os.path.join(out_dir, "manifest.json"), {})

    def src_array(ref, use_overlay):
        if os.path.exists(ref):
            return np.asarray(Image.open(ref).convert("RGB")), None
        code, t = parse_frame_ref(ref)
        arr = np.asarray(Image.open(grab_frame(W, code, t)).convert("RGB"))
        return arr, (crop_box(W, code) if use_overlay else None)

    for kind in ("hero", "plates", "specimens"):
        group = spec.get(kind) or {}
        if kind == "hero" and group:
            group = {"hero": group}
        for name, s in group.items():
            if isinstance(s, str):
                s = {"src": s}
            arr, ovbox = src_array(s["src"], kind != "specimens")
            box = s.get("box") or ovbox or [0, 0, arr.shape[1], arr.shape[0]]
            x0, y0, x1, y1 = [int(v) for v in box]
            if kind == "specimens":
                im = Image.fromarray(arr[y0:y1, x0:x1])
                sc = s.get("scale", 2)
                mode = Image.NEAREST if s.get("mode", "nearest") == "nearest" else Image.LANCZOS
                if sc != 1:
                    im = im.resize((round(im.width * sc), round(im.height * sc)), mode)
                fmt = s.get("fmt") or ("jpg" if s.get("mode") == "smooth" else "png")
                f = f"{name}.{fmt}"
                (im.save(os.path.join(out_dir, f), quality=88) if fmt == "jpg" else im.save(os.path.join(out_dir, f), optimize=True))
                entry = {"file": f, "w": im.width, "h": im.height, "kind": "specimen", "src": s["src"], "box": [x0, y0, x1, y1], "scale": sc}
                if s.get("chips", True):
                    entry["chips"] = chips_of(arr[y0:y1, x0:x1])
            else:
                ratio = 2.2 if kind == "hero" else s.get("ratio", 16 / 9)
                bw, bh = x1 - x0, y1 - y0
                if bw / bh > ratio:
                    nw = int(bh * ratio)
                    fx = s.get("focus_x", 0.5)
                    x0 = int(min(max(x0, x0 + (bw - nw) * fx), x1 - nw))
                    x1 = x0 + nw
                else:
                    nh = int(bw / ratio)
                    fy = s.get("focus_y", 0.5)
                    y0 = int(min(max(y0, y0 + (bh - nh) * fy), y1 - nh))
                    y1 = y0 + nh
                im = Image.fromarray(arr[y0:y1, x0:x1])
                tw = s.get("width") or min(1600, im.width)    # never invent pixels by default
                im = im.resize((tw, round(tw / ratio)), Image.LANCZOS)
                f = f"{name}.jpg"
                im.save(os.path.join(out_dir, f), quality=s.get("quality", 82), optimize=True, progressive=True)
                entry = {"file": f, "w": im.width, "h": im.height, "kind": "hero" if kind == "hero" else "plate", "src": s["src"], "box": [x0, y0, x1, y1]}
            if s.get("px_per_art"):
                entry["px_per_art"] = s["px_per_art"]
            if not os.path.exists(s["src"]):    # a plate or specimen that reaches into a watermark or HUD box
                ovl = jload(os.path.join(W, "overlay.json"), {}).get(parse_frame_ref(s["src"])[0], {})
                b, tol = entry["box"], max(8, 0.015 * arr.shape[0])    # overlay boxes carry a small margin
                hit = [o for o in ovl.get("boxes", []) if min(b[2], o[2]) - max(b[0], o[0]) > tol and min(b[3], o[3]) - max(b[1], o[1]) > tol]
                if hit:
                    print(f"   warning: {name} {b} overlaps the overlay box {hit[0]}: the watermark or HUD will show. Move the box or set focus_x/focus_y")
            old = man.get(name)
            if old and old.get("file") != entry["file"] and os.path.exists(os.path.join(out_dir, old["file"])):
                os.remove(os.path.join(out_dir, old["file"]))    # e.g. m_window.jpg left behind by m_window.png
            man[name] = entry
            print(f"{kind[:-1] if kind != 'hero' else 'hero'} {name}: {entry['w']}x{entry['h']} {os.path.getsize(os.path.join(out_dir, entry['file'])) // 1024} KB"
                  + (f" chips {entry['chips']}" if entry.get("chips") else ""))
    keep = ({"hero"} if spec.get("hero") else set()) | set(spec.get("plates") or {}) | set(spec.get("specimens") or {})
    stale = [n for n, e in man.items() if e.get("kind") in ("hero", "plate", "specimen") and n not in keep]
    for n in stale:
        f = os.path.join(out_dir, man.pop(n)["file"])
        if os.path.exists(f):
            os.remove(f)
    if stale:
        print(f"removed {len(stale)} images that are no longer in {os.path.basename(a.spec)}: {', '.join(stale)}")
    jsave(os.path.join(out_dir, "manifest.json"), man)


# ───────────────────────── build: content.json -> HTML ─────────────────────────
CANON = ["define", "palette", "l1", "l2", "l3", "l4", "l5", "l6", "l7", "motion", "sound", "rules", "recipe"]
NO_SUB = {"pillars", "lede", "note", "h3"}
FONT_LINK = "https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=IBM+Plex+Mono:wght@400;500&family=Noto+Serif+SC:wght@400;600;900&display=swap"
INLINE_TAGS = "a|abbr|b|br|cite|code|del|em|i|ins|kbd|mark|q|s|small|span|strong|sub|sup|u|wbr"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"


def tx(s):
    """Inline text: HTML allowed; shorthands {{实测}} -> tag, {{dot:#hex}} -> colour dot, **b**, `code`."""
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"<(?!/?(?:" + INLINE_TAGS + r")(?=[\s/>]))", "&lt;", s)
    s = re.sub(r"\{\{dot:(#[0-9A-Fa-f]{3,8})\}\}", r'<span class="dot" style="background:\1"></span>', s)
    # a tag and the punctuation right after it never wrap apart
    s = re.sub(r"\{\{([^{}]{1,24})\}\}([，。；：、！？）」』,.;:!?)]+)", r'<span class="nw"><span class="tag">\1</span>\2</span>', s)
    s = re.sub(r"\{\{([^{}]{1,24})\}\}", r'<span class="tag">\1</span>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


def attr(s):
    return html.escape(re.sub(r"<[^>]+>", "", str(s or "")), quote=True)


def autodot(s):
    m = re.match(r"\s*(#[0-9A-Fa-f]{6})\b", s)
    return (f'<span class="dot" style="background:{m.group(1)}"></span>' + s) if m and "class=\"dot\"" not in s else s


class Ctx:
    def __init__(self, W, content, base="."):
        self.W = W
        self.c = content
        self.base = base
        self.man = jload(os.path.join(W, "img", "manifest.json"), {})
        self.db = jload(os.path.join(W, "measure.json"), {"frames": {}, "profiles": {}})
        self.warn = []
        self.errors = []
        self.notes = []
        self.uris = {}

    def raw(self, b, key):
        """b[key], or the text of the file named by b[key + '_file'] (relative to content.json)."""
        if b.get(key):
            return b[key]
        f = b.get(key + "_file")
        if not f:
            return ""
        p = f if os.path.isabs(f) else os.path.join(self.base, f)
        if not os.path.exists(p):
            self.errors.append(f"{key}_file '{f}' not found")
            return ""
        return open(p, encoding="utf-8").read()

    def img(self, name):
        if name in self.uris:
            return self.uris[name]
        e = self.man.get(name)
        path = os.path.join(self.W, "img", e["file"]) if e else None
        if not path:
            for ext in (".jpg", ".png", ".jpeg", ".webp"):
                if os.path.exists(os.path.join(self.W, "img", name + ext)):
                    path = os.path.join(self.W, "img", name + ext)
        if not path or not os.path.exists(path):
            self.errors.append(f"image '{name}' not found in {self.W}/img (run crops)")
            self.uris[name] = ("data:,", 1, 1, {})
            return self.uris[name]
        with Image.open(path) as im:
            w, h = im.size
        mime = "image/png" if path.lower().endswith(".png") else "image/webp" if path.lower().endswith(".webp") else "image/jpeg"
        uri = f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()
        self.uris[name] = (uri, w, h, e or {})
        return self.uris[name]


def wrap(b, inner, sub):
    head = ""
    if b.get("h3"):
        head += f'<h3 class="h3">{tx(b["h3"])}</h3>'
    if b.get("intro"):
        head += f'<p class="muted intro">{tx(b["intro"])}</p>'
    tail = f'<p class="note">{tx(b["note"])}</p>' if b.get("note") and b["type"] not in ("tiers", "lineup", "spectrum", "loudness", "diagram", "sharpness", "cast") else ""
    if head or tail or sub:
        cls = "sub" if sub else "blk"
        return f'<div class="{cls}">{head}{inner}{tail}</div>'
    return inner


def plate_html(p, ctx, extra=""):
    uri, w, h, _ = ctx.img(p["img"])
    marks = "".join(f'<div class="mk" style="left:{x}%;top:{y}%" aria-hidden="true"><span>{i + 1}</span></div>' for i, (x, y) in enumerate(p.get("marks", [])))
    cap = ""
    if p.get("caption") or p.get("meta"):
        cap = f'<figcaption><span>{tx(p.get("caption", ""))}</span><span class="meta">{tx(p.get("meta", ""))}</span></figcaption>'
    return (f'<figure class="plate"><div class="img{" mkset" if marks else ""}"><img src="{uri}" alt="{attr(p.get("alt", p.get("caption", "")))}" width="{w}" height="{h}">'
            f'{marks}{extra}</div>{cap}</figure>')


def grey_hex(hx):
    r, g, b = [v / 255 for v in hex_to_rgb(hx)]
    y = float(srgb_to_lin(np.array([r, g, b])) @ np.array([0.2126, 0.7152, 0.0722]))
    v = int(round(float(lin_to_srgb(y)) * 255))
    return f"#{v:02X}{v:02X}{v:02X}"


def tier_row(label, tiers, grey=False):
    cells = []
    for t, (pct, cols) in enumerate(tiers):
        for i, (hx, sh) in enumerate(cols):
            col = grey_hex(hx) if grey else hx
            cells.append(f'<i class="{"cut" if t and not i else ""}" style="flex-grow:{pct * sh:.2f};background:{col}" title="{col} · {pct * sh:.1f}%"></i>')
    meta = " · ".join(str(round(t[0])) for t in tiers)
    return (f'<div class="tier-row"><span>{tx(label)}</span><div class="tier-bar" role="img" aria-label="{attr(label)}：暗部 {tiers[0][0]}%，中间调 {tiers[1][0]}%，亮部 {tiers[2][0]}%">'
            f'{"".join(cells)}</div><span class="meta">{meta}</span></div>')


def tiers_from(ctx, r):
    if r.get("tiers"):
        return r["tiers"]
    fr = ctx.db.get("frames", {}).get(r.get("frame", ""))
    if not fr:
        ctx.errors.append(f"tiers: no measurement for frame '{r.get('frame')}' (run measure)")
        return [[0, []], [0, []], [0, []]]
    return [[t["pct"], t["colors"]] for t in fr["tiers"]]


def loud_means(W, b):
    """Energy-mean level (dB) of each segment of a `loudness` block, and per segment name over all its segments: the numbers
    the chart prints and the prose may quote. None when the series has no audio/CODE_loud.json."""
    data = jload(os.path.join(W, "audio", f"{b['series']}_loud.json"))
    if not data:
        return None
    t0, t1 = b.get("t0", 0), b.get("t1", data[-1]["t"] + 1)
    data = [d for d in data if t0 <= d["t"] < t1]
    t = np.array([d["t"] for d in data], dtype=float) + 0.5
    e = 10 ** (np.array([d["db"] for d in data], dtype=float) / 10)
    segs, byname = [], {}
    for sg in b.get("segments", []):
        m = (t >= sg["a"]) & (t < sg["b"])
        segs.append(float(10 * np.log10(e[m].mean() + 1e-12)) if m.any() else None)
        n = byname.setdefault(sg.get("name", ""), [np.zeros(len(t), bool), 0])
        n[0] |= m
        n[1] += 1
    names = {k: (float(10 * np.log10(e[m].mean() + 1e-12)), c, int(m.sum())) for k, (m, c) in byname.items() if m.any()}
    return segs, names


def svg_loudness(ctx, b):
    code = b["series"]
    data = jload(os.path.join(ctx.W, "audio", f"{code}_loud.json"))
    if not data:
        ctx.errors.append(f"loudness: {ctx.W}/audio/{code}_loud.json missing (run audio)")
        return ""
    t0 = b.get("t0", 0)
    t1 = b.get("t1", data[-1]["t"] + 1)
    data = [d for d in data if t0 <= d["t"] < t1]
    t = np.array([d["t"] for d in data], dtype=float) + 0.5
    db = np.array([d["db"] for d in data], dtype=float)
    e = 10 ** (db / 10)
    es = np.convolve(e, np.ones(3) / 3, mode="same")
    dbs = 10 * np.log10(es + 1e-12)
    lo, hi = b.get("range") or [math.floor((np.percentile(dbs, 2) - 3) / 5) * 5, math.ceil((dbs.max() + 3) / 5) * 5]
    X = lambda s: 70 + 900 * (s - t0) / max(t1 - t0, 1e-6)
    Y = lambda v: 250 - 220 * (min(max(v, lo), hi) - lo) / (hi - lo)
    pal = {"loud": ("rgba(243,166,91,.14)", "#f3a65b"), "calm": ("rgba(115,184,234,.08)", "#73b8ea"), "other": ("rgba(217,189,127,.07)", "#d9bd7f")}
    out = [f'<svg viewBox="0 0 1000 290" role="img" aria-label="{attr(b.get("aria", b.get("label", "loudness")))}">']
    segs = b.get("segments", [])
    lvls = {i: v for i, v in enumerate((loud_means(ctx.W, b) or ([], {}))[0]) if v is not None}
    ctx.notes.append(f"loudness chart {code} (energy mean per segment, quote these): " + "; ".join(
        f'{s.get("name", "")} {mmss(s["a"])}–{mmss(s["b"])} {lvls[i]:.1f} dB' for i, s in enumerate(segs) if i in lvls))
    narrow = 0
    for i, s in enumerate(segs):
        fill, col = pal.get(s.get("kind", "other"), pal["other"])
        wd = X(s["b"]) - X(s["a"])
        out.append(f'<rect x="{X(s["a"]):.1f}" y="30" width="{max(0.5, wd):.1f}" height="220" fill="{fill}"/>')
        if wd > 28:
            out.append(f'<text class="d-t" x="{(X(s["a"]) + X(s["b"])) / 2:.1f}" y="48" text-anchor="middle" style="fill:{col}">{attr(s.get("name", ""))}</text>')
        elif i in lvls:    # too narrow to label inside: name and level above the plot, alternating rows
            ly = 12 if narrow % 2 else 24
            narrow += 1
            lab = f'{s.get("name", "")} {lvls[i]:.1f} dB'.replace("-", "−").strip()
            out.append(f'<line x1="{(X(s["a"]) + X(s["b"])) / 2:.1f}" y1="{ly + 2}" x2="{(X(s["a"]) + X(s["b"])) / 2:.1f}" y2="30" stroke="{col}" stroke-width="1"/>'
                       f'<text class="d-m" x="{(X(s["a"]) + X(s["b"])) / 2:.1f}" y="{ly}" text-anchor="middle" style="fill:{col}">{attr(lab)}</text>')
    step = 10 if hi - lo <= 40 else 20
    for v in range(int(math.ceil(lo / step) * step), int(hi) + 1, step):
        out.append(f'<line x1="70" y1="{Y(v):.1f}" x2="970" y2="{Y(v):.1f}" stroke="rgba(217,189,127,.18)"/>')
        out.append(f'<text class="d-m" x="62" y="{Y(v) + 4:.1f}" text-anchor="end">{str(v).replace("-", "−")} dB</text>')
    span = t1 - t0
    tick = next(s for s in (5, 10, 15, 30, 60, 120, 300, 600, 1e9) if span / s <= 9)
    k = math.ceil(t0 / tick) * tick
    while k <= t1:
        out.append(f'<text class="d-m" x="{X(k):.1f}" y="270" text-anchor="middle">{int(k // 60)}:{int(k % 60):02d}</text>')
        k += tick
    out.append(f'<path d="M{" L".join(f"{X(a_):.1f} {Y(v):.1f}" for a_, v in zip(t, dbs))}" fill="none" stroke="#efe5c9" stroke-width="1.6" stroke-linejoin="round"/>')
    for i, s in enumerate(segs):
        if i not in lvls:
            continue
        lvl = lvls[i]
        _, col = pal.get(s.get("kind", "other"), pal["other"])
        out.append(f'<line x1="{X(s["a"]) + 3:.1f}" y1="{Y(lvl):.1f}" x2="{X(s["b"]) - 3:.1f}" y2="{Y(lvl):.1f}" stroke="{col}" stroke-width="2.5"/>')
        if 28 < X(s["b"]) - X(s["a"]) <= 70:
            out.append(f'<text class="d-m" x="{(X(s["a"]) + X(s["b"])) / 2:.1f}" y="{Y(lvl) - 7:.1f}" text-anchor="middle" style="fill:{col};font-size:10px" stroke="#0b0d18" stroke-width="4" stroke-linejoin="round" paint-order="stroke">{f"{lvl:.0f}".replace("-", "−")}</text>')
        if X(s["b"]) - X(s["a"]) > 70:
            out.append(f'<text class="d-m" x="{(X(s["a"]) + X(s["b"])) / 2:.1f}" y="{Y(lvl) - 7:.1f}" text-anchor="middle" style="fill:{col}" stroke="#0b0d18" stroke-width="4" stroke-linejoin="round" paint-order="stroke">{f"{lvl:.1f}".replace("-", "−")} dB</text>')
    out.append('<text class="d-g" x="970" y="286" text-anchor="end">TIME →</text></svg>')
    return "".join(out)


def svg_spectrum(ctx, b):
    uri, w, h, e = ctx.img(b["img"])
    t0, t1, fmax = e.get("t0", b.get("t0", 0)), e.get("t1", b.get("t1", 10)), e.get("fmax", b.get("fmax", 9000))
    log = e.get("scale") == "log"
    fmin = e.get("fmin") or 40.0
    Y = (lambda f: 260 - 250 * math.log(max(f, fmin) / fmin) / math.log(fmax / fmin)) if log else (lambda f: 260 - 250 * f / fmax)
    out = [f'<svg viewBox="0 0 1000 300" role="img" aria-label="{attr(b.get("aria", b.get("label", "spectrum")))}">',
           f'<image href="{uri}" x="70" y="10" width="800" height="250" preserveAspectRatio="none"/>',
           '<rect x="70" y="10" width="800" height="250" fill="none" stroke="#7d6a45"/>']
    if log:
        ticks = [f for f in (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000) if fmin <= f <= fmax]
    else:
        fstep = 2000 if fmax > 5000 else 1000
        ticks = list(range(0, int(fmax) + 1, fstep))
    for f in ticks:
        lab = "0" if f == 0 else (f"{f / 1000:g} kHz" if f >= 1000 else f"{f} Hz")
        out.append(f'<text class="d-m" x="62" y="{Y(f) + 4:.1f}" text-anchor="end">{lab}</text>')
        if log:
            out.append(f'<line x1="70" y1="{Y(f):.1f}" x2="76" y2="{Y(f):.1f}" stroke="#7d6a45"/>')
    for i in range(6):
        tt = t0 + (t1 - t0) * i / 5
        out.append(f'<text class="d-m" x="{70 + 800 * i / 5:.0f}" y="278" text-anchor="middle">{int(tt // 60)}:{tt % 60:04.1f}</text>' if (t1 - t0) < 20 else
                   f'<text class="d-m" x="{70 + 800 * i / 5:.0f}" y="278" text-anchor="middle">{int(tt // 60)}:{int(tt % 60):02d}</text>')
    for an in b.get("annotations", []):
        y0, y1 = Y(min(an["f1"], fmax)), Y(an["f0"])
        out.append(f'<path d="M878 {y0:.0f} h8 v{max(4, y1 - y0):.0f} h-8" fill="none" stroke="#efe5c9" stroke-width="1.5"/>')
        ty = (y0 + y1) / 2 - 6 * len(an.get("lines", []))
        out.append(f'<text class="d-t" x="894" y="{ty + 4:.0f}">{attr(an.get("title", ""))}</text>')
        for j, ln in enumerate(an.get("lines", [])):
            out.append(f'<text class="d-m" x="894" y="{ty + 19 + 14 * j:.0f}">{attr(ln)}</text>')
    out.append('<text class="d-g" x="870" y="296" text-anchor="end">TIME →</text></svg>')
    return "".join(out)


def walk_blocks(bs):
    for b in bs or []:
        yield b
        yield from walk_blocks(b.get("blocks"))
        yield from walk_blocks(b.get("left"))
        yield from walk_blocks(b.get("right"))


def render_block(b, ctx, sub=False):
    t = b.get("type")
    f = BLOCKS.get(t)
    if not f:
        ctx.errors.append(f"unknown block type {t!r}")
        return ""
    inner = f(b, ctx)
    return wrap(b, inner, sub)


def render_blocks(bs, ctx):
    parts = [render_block(b, ctx) for b in bs]
    return parts[0] if len(parts) == 1 else f'<div class="stack">{"".join(parts)}</div>'


def b_prose(b, ctx):
    paras = b.get("paras") or ([b["text"]] if b.get("text") else [])
    return f'<div class="prose">{"".join(f"<p>{tx(p)}</p>" for p in paras)}</div>'


def b_cols(b, ctx):
    lay = b.get("layout", "even")
    sides = [b.get("left", []), b.get("right", [])]
    has_tbl = any(x.get("type") == "table" for s in sides for x in s)
    cls = "cols" + (" wide-left" if lay == "wide-left" else "") + (" tblcols" if has_tbl else "")
    return f'<div class="{cls}">' + "".join(render_blocks(s, ctx) if s else "<div></div>" for s in sides) + "</div>"


def b_card(b, ctx):
    return '<dl class="card" aria-label="风格名片">' + "".join(f"<div><dt>{tx(k)}</dt><dd>{tx(v)}</dd></div>" for k, v in b["items"]) + "</dl>"


def b_pillars(b, ctx):
    lis = []
    for p in b["items"]:
        rule = f'<div class="rule"><b>{tx(b.get("rule_label", "规则"))}</b>{tx(p["rule"])}</div>' if p.get("rule") else ""
        lis.append(f'<li><div class="ph"><b>{tx(p["name"])}</b><span>{tx(p.get("en", ""))}</span></div><div><p>{tx(p["text"])}</p>{rule}</div></li>')
    return f'<ol class="pillars" aria-label="支柱">{"".join(lis)}</ol>'


def b_swatches(b, ctx):
    rows = []
    for g in b["groups"]:
        sw = []
        for c in g["colors"]:
            n, hx, role = (c["name"], c["hex"], c.get("role", "")) if isinstance(c, dict) else (list(c) + [""])[:3]
            hx = hx.upper() if re.fullmatch(r"#[0-9a-fA-F]{6}", hx) else hx
            sw.append(f'<button class="sw" type="button" data-hex="{hx}" aria-label="{attr(n)} {hx}，点击复制色值"><i style="background:{hx}"></i><b>{tx(n)}</b>'
                      f'<span class="meta">{hx}</span><span class="role">{tx(role)}</span></button>')
        rows.append(f'<div class="swgroup"><div><h4>{tx(g["name"])}<small>{tx(g.get("en", ""))}</small></h4><p>{tx(g.get("desc", ""))}</p></div><div class="swatches">{"".join(sw)}</div></div>')
    return f'<div class="swgroups">{"".join(rows)}</div>'


def b_tiers(b, ctx):
    rows = [tier_row(r.get("label", r.get("frame", "")), tiers_from(ctx, r)) for r in b["rows"]]
    gi = b.get("grey_compare")
    if gi is not None and 0 <= gi < len(b["rows"]):
        rows.append(tier_row(b.get("grey_label", "对照：去掉色相"), tiers_from(ctx, b["rows"][gi]), grey=True).replace('class="tier-row"', 'class="tier-row" style="margin-top:12px"', 1))
    key = ('<div class="tier-key"><span>暗部 L* &lt; 20</span><span>中间调</span><span>亮部 L* ≥ 55</span><span class="mono">右侧：暗 · 中 · 亮 %</span></div>')
    note = f'<p class="note">{tx(b["note"])}</p>' if b.get("note") else ""
    return f'<div class="spec"><div class="spec-label">{tx(b.get("label", "TIERED PALETTES · 实测"))}</div>{key}{"".join(rows)}{note}</div>'


def b_plate(b, ctx):
    return plate_html(b, ctx)


def b_plates(b, ctx):
    n = b.get("cols", 2)
    figs = "".join(plate_html(p, ctx) for p in b["items"])
    return f'<div class="cols">{figs}</div>' if n == 2 else f'<div class="pgrid" style="--n:{n}">{figs}</div>'


def b_spec(b, ctx):
    pad = f' style="padding:{b["pad"]}px"' if b.get("pad") is not None else ""
    return f'<div class="spec"{pad}><div class="spec-label">{tx(b.get("label", ""))}</div>{"".join(render_block(x, ctx) for x in b.get("blocks", []))}</div>'


def b_diagram(b, ctx):
    note = f'<p class="note">{tx(b["note"])}</p>' if b.get("note") else ""
    return f'<div class="spec" style="padding:14px"><div class="spec-label">{tx(b.get("label", ""))}</div><div class="diagram">{ctx.raw(b, "svg")}</div>{note}</div>'


def b_table(b, ctx):
    v = set(b.get("v", []))
    n = len(b["head"])
    rows = []
    for r in b["rows"]:
        if isinstance(r, dict):
            rows.append(f'<tr class="grp"><td colspan="{n}">{tx(r.get("group", ""))}</td></tr>')
            continue
        cells = [f'<td class="v">{autodot(tx(c))}</td>' if i in v else f"<td>{tx(c)}</td>" for i, c in enumerate(r)]
        rows.append(f"<tr>{''.join(cells)}</tr>")
    head = "".join(f"<th>{tx(h)}</th>" for h in b["head"])
    return f'<div class="tbl"><table class="params {attr(b.get("cls", ""))}"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def b_marks(b, ctx):
    return '<ol class="marks">' + "".join(f"<li><span>{tx(x)}</span></li>" for x in b["items"]) + "</ol>"


def art_size(b, c, e, axis):
    """Display size that gives every source the same art-pixel size: (screen px / px_per_art) x art_px."""
    ppa = c.get("px_per_art") or e.get("px_per_art")
    box = e.get("box")
    if not (ppa and box):
        return None
    return round((box[2] - box[0] if axis == "w" else box[3] - box[1]) / ppa * b.get("art_px", 3))


def b_cast(b, ctx):
    items = b["items"]
    figs = []
    for c in items:
        uri, w, h, e = ctx.img(c["img"])
        cw = c.get("width") or art_size(b, c, e, "w") or round(w / 2)
        chips = c.get("chips") or e.get("chips") or []
        ch = "".join(f'<i style="background:{x}" title="{x}"></i>' for x in chips)
        figs.append(f'<figure><img class="px" src="{uri}" alt="{attr(c.get("alt", c.get("name", "")))}" width="{cw}" style="width:{cw}px">'
                    f'<figcaption><b>{tx(c.get("name", ""))}</b><span>{tx(c.get("role", ""))}</span><div class="chips">{ch}</div></figcaption></figure>')
    note = f'<p class="note">{tx(b["note"])}</p>' if b.get("note") else ""
    return f'<div class="cast" style="--n:{min(len(items), 9)}" aria-label="角色一览">{"".join(figs)}</div>{note}'


def b_matgrid(b, ctx):
    figs = []
    for m in b["items"]:
        uri, w, h, _ = ctx.img(m["img"])
        if h and abs(w / h - 1) > 0.15 and uri != "data:,":
            ctx.warn.append(f"matgrid shows square tiles, so '{m['img']}' ({w}x{h}) loses its edges: give it a square box in crops.json")
        figs.append(f'<figure><img class="{"px" if m.get("px", True) else ""}" src="{uri}" alt="{attr(m.get("alt", m.get("name", "")))}" width="{w}" height="{h}">'
                    f'<figcaption>{tx(m.get("name", ""))}<span class="meta">{tx(m.get("meta", ""))}</span></figcaption></figure>')
    return f'<div class="matgrid" style="--n:{b.get("cols", 4)}">{"".join(figs)}</div>'


def b_lineup(b, ctx):
    figs = []
    for m in b["items"]:
        uri, w, h, e = ctx.img(m["img"])
        hh = m.get("height") or art_size(b, m, e, "h") or h
        figs.append(f'<figure><img class="px lu" src="{uri}" alt="{attr(m.get("alt", m.get("caption", "")))}" style="--h:{hh}"><figcaption>{tx(m.get("caption", ""))}</figcaption></figure>')
    note = f'<p class="note">{tx(b["note"])}</p>' if b.get("note") else ""
    return f'<div class="spec"><div class="spec-label">{tx(b.get("label", "LINEUP · TRUE SCALE"))}</div><div class="lineup center">{"".join(figs)}</div>{note}</div>'


def b_figure_note(b, ctx):
    uri, w, h, _ = ctx.img(b["img"])
    meta = f'<p class="meta muted" style="margin:12px 0 0">{tx(b["meta"])}</p>' if b.get("meta") else ""
    return (f'<div class="spec"><div class="spec-label">{tx(b.get("label", ""))}</div><div class="texel"><img class="{"px" if b.get("px", True) else ""}" src="{uri}" alt="{attr(b.get("alt", ""))}" width="{w}" height="{h}">'
            f'<p class="muted" style="font-size:14px;margin:0;line-height:1.75">{tx(b.get("text", ""))}</p></div>{meta}</div>')


def b_image(b, ctx):
    uri, w, h, _ = ctx.img(b["img"])
    marks = "".join(f'<div class="mk" style="left:{x}%;top:{y}%" aria-hidden="true"><span>{i + 1}</span></div>' for i, (x, y) in enumerate(b.get("marks", [])))
    style = f' style="width:min(100%,{b["width"]}px)"' if b.get("width") else ""
    cap = f'<p class="note">{tx(b["caption"])}</p>' if b.get("caption") else ""
    return f'<div class="{attr(b.get("cls", "anat"))} mkset"{style}><img class="{"px" if b.get("px", True) else ""}" src="{uri}" alt="{attr(b.get("alt", ""))}" width="{w}" height="{h}">{marks}</div>{cap}'


def b_sharpness(b, ctx):
    figs = []
    for it in b["items"]:
        prof = it.get("profile")
        bd = it.get("band") or b.get("band")
        if not prof and it.get("measure"):
            p = ctx.db.get("profiles", {}).get(it["measure"])
            if not p:
                ctx.errors.append(f"sharpness: profile '{it['measure']}' not in measure.json (run profile)")
                prof = [0] * 12
            else:
                prof = p["profile"]
                bd = bd or p.get("sharp_band_pct")
        bd = bd or [40, 65]
        nb = len(prof)
        bars = "".join(f'<b class="pb" style="top:{i * 100 / nb:.3f}%;height:{100 / nb:.3f}%;width:{v * 24:.1f}%"></b>' for i, v in enumerate(prof))
        ov = (f'<div class="band on" aria-hidden="true"><div class="z" style="top:{bd[0]}%;height:{bd[1] - bd[0]}%"></div>'
              f'<i style="top:{bd[0]}%"></i><em class="left" style="top:{bd[0]}%">{bd[0]}%</em><i style="top:{bd[1]}%"></i><em class="left" style="top:{bd[1]}%">{bd[1]}%</em>'
              f'<span class="pl">清晰度 →</span>{bars}</div>')
        fig = plate_html(it, ctx, extra=ov)
        if it.get("note"):
            fig = f'<div>{fig}<p class="note">{tx(it["note"])}</p></div>'
        figs.append(fig)
    note = f'<p class="note">{tx(b["note"])}</p>' if b.get("note") else ""
    return f'<div class="cols sharpgrid{" three" if len(figs) == 3 else ""}">{"".join(figs)}</div>{note}'


def b_script(b, ctx):
    out = []
    for it in b["items"]:
        uri, w, h, _ = ctx.img(it["img"])
        ro = it.get("readout")
        if not ro and it.get("frame"):
            fr = ctx.db.get("frames", {}).get(it["frame"])
            if fr:
                tt = fr["tiers"]
                ro = [["暗部面积", f'{tt[0]["pct"]}%'], ["暗部主色", tt[0]["colors"][0][0] if tt[0]["colors"] else "-"],
                      ["中间调主色", tt[1]["colors"][0][0] if tt[1]["colors"] else "-"], ["亮部面积", f'{tt[2]["pct"]}%'],
                      ["亮部主色", tt[2]["colors"][0][0] if tt[2]["colors"] else "-"], ["平均彩度 C*", str(fr["chroma"]["mean"])]]
            else:
                ctx.errors.append(f"script: no measurement for {it['frame']}")
        dl = "".join(f"<dt>{tx(k)}</dt><dd>{autodot(tx(v))}</dd>" for k, v in (ro or []))
        note = f'<p class="note">{tx(it["note"])}</p>' if it.get("note") else ""
        out.append(f'<div class="script"><div class="skybox"><img src="{uri}" alt="{attr(it.get("alt", ""))}" width="{w}" height="{h}"></div>'
                   f'<div class="spec"><div class="timebig">{tx(it.get("big", ""))}</div><dl class="readout" style="margin-top:12px">{dl}</dl>{note}</div></div>')
    return f'<div class="stack scripts">{"".join(out)}</div>'


def b_rules(b, ctx):
    do = "".join(f"<li>{tx(x)}</li>" for x in b.get("do", []))
    dont = "".join(f"<li>{tx(x)}</li>" for x in b.get("dont", []))
    return f'<div class="rules"><div class="do"><h3>DO</h3><ul>{do}</ul></div><div class="dont"><h3>DON\'T</h3><ul>{dont}</ul></div></div>'


def b_check(b, ctx):
    return '<ul class="check">' + "".join(f"<li>{tx(x)}</li>" for x in b["items"]) + "</ul>"


def b_prompts(b, ctx):
    out = []
    for p in b["items"]:
        lang = p.get("lang", "")
        en = bool(re.fullmatch(r"[A-Za-z ·\-]+", lang))
        out.append(f'<div class="prompt"><span class="lang">{attr(lang)}</span><button class="btn" type="button" data-copy>{"Copy" if en else "复制"}</button>'
                   f'<span class="ptext">{html.escape(p["text"])}</span></div>')
    return f'<div class="stack">{"".join(out)}</div>'


def b_loudness(b, ctx):
    note = f'<p class="note">{tx(b["note"])}</p>' if b.get("note") else ""
    return f'<div class="spec" style="padding:14px"><div class="spec-label">{tx(b.get("label", "LOUDNESS"))}</div><div class="diagram">{svg_loudness(ctx, b)}</div>{note}</div>'


def b_spectrum(b, ctx):
    note = f'<p class="note">{tx(b["note"])}</p>' if b.get("note") else ""
    return f'<div class="spec" style="padding:14px"><div class="spec-label">{tx(b.get("label", "SPECTRUM"))}</div><div class="diagram">{svg_spectrum(ctx, b)}</div>{note}</div>'


def b_typespec(b, ctx):
    rows = "".join(f'<div class="row"><span>{tx(r.get("label", ""))}</span><div class="{attr(r.get("cls", ""))}" style="{attr(r.get("style", ""))}">{tx(r.get("sample", ""))}</div></div>' for r in b["rows"])
    return f'<div class="typespec">{rows}</div>'


def b_uiwall(b, ctx):
    return f'<div class="uiwall {attr(b.get("cls", ""))}">{ctx.raw(b, "html")}</div>'


def b_html(b, ctx):
    return ctx.raw(b, "html")


def b_lede(b, ctx):
    return f'<p class="lede">{tx(b["text"])}</p>'


def b_note(b, ctx):
    return f'<p class="note">{tx(b["text"])}</p>'


def b_h3(b, ctx):
    return f'<h3 class="h3">{tx(b["text"])}</h3>'


BLOCKS = {"prose": b_prose, "cols": b_cols, "card": b_card, "pillars": b_pillars, "swatches": b_swatches, "tiers": b_tiers,
          "plate": b_plate, "plates": b_plates, "spec": b_spec, "diagram": b_diagram, "table": b_table, "marks": b_marks,
          "cast": b_cast, "matgrid": b_matgrid, "lineup": b_lineup, "figure_note": b_figure_note, "image": b_image,
          "sharpness": b_sharpness, "script": b_script, "rules": b_rules, "check": b_check, "prompts": b_prompts,
          "loudness": b_loudness, "spectrum": b_spectrum, "typespec": b_typespec, "uiwall": b_uiwall, "html": b_html,
          "lede": b_lede, "note": b_note, "h3": b_h3}

COPY_JS = """<script>
(function(){
  function copy(text, el, done){
    var fb=function(){ if(el){ var r=document.createRange(); r.selectNodeContents(el); var s=getSelection(); s.removeAllRanges(); s.addRange(r);} done(false); };
    try{ navigator.clipboard.writeText(text).then(function(){done(true)}, fb); }catch(e){ fb(); }
  }
  document.querySelectorAll('.sw[data-hex]').forEach(function(b){ b.addEventListener('click', function(){
    var m=b.querySelector('.meta'), hex=b.getAttribute('data-hex');
    copy(hex, m, function(ok){ if(!ok) return; b.classList.add('copied'); m.textContent='已复制'; setTimeout(function(){ b.classList.remove('copied'); m.textContent=hex; },1200); });
  }); });
  document.querySelectorAll('.prompt [data-copy]').forEach(function(b){ b.addEventListener('click', function(){
    var el=b.parentNode.querySelector('.ptext'), label=b.textContent, en=label==='Copy';
    copy(el.textContent, el, function(ok){ b.textContent= ok ? (en?'Copied':'已复制') : (en?'Selected':'已选中'); setTimeout(function(){ b.textContent=label; },1400); });
  }); });
})();
</script>"""


def fetch(url, binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    return data if binary else data.decode("utf-8")


LATIN_FONTS = {"Cinzel", "IBM Plex Mono"}


def embed_fonts(families):
    """Download Google Fonts subsets for exactly the characters used and inline them (base64).
    families: (family, spec, text) - each family gets only its own text (a class font only its class's text).
    Latin-only families get only the non-CJK characters (Google answers 400 for a CJK-only subset of them);
    a failed chunk is skipped and reported, the rest still embeds. Returns (css, failures)."""
    css_out, failed = [], []

    def repl(m):
        data = fetch(m.group(1), binary=True)
        fmt = "woff2" if data[:4] == b"wOF2" else "woff" if data[:4] == b"wOFF" else "truetype"
        mime = {"woff2": "font/woff2", "woff": "font/woff", "truetype": "font/ttf"}[fmt]
        return f"url(data:{mime};base64,{base64.b64encode(data).decode()}) format('{fmt}')"

    for fam, spec, text in families:
        chars = sorted(set(ch for ch in text if ch not in " \t\r\n\f\v") | set(" "))    # keeps U+3000 and U+00A0: they are glyphs
        cs = [ch for ch in chars if ord(ch) < 0x2E80] if fam in LATIN_FONTS else chars
        for i in range(0, len(cs), 220):
            chunk = "".join(cs[i:i + 220])
            url = f"https://fonts.googleapis.com/css2?family={fam.replace(' ', '+')}{(':' + spec) if spec else ''}&text={urllib.parse.quote(chunk, safe='')}&display=swap"
            try:
                css_out.append(re.sub(r"url\((https://[^)]+)\)\s*format\('[^']+'\)", repl, fetch(url)))
            except Exception as e:
                failed.append(f"{fam} chunk {i // 220 + 1} ({e})")
    return "\n".join(css_out), failed


VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


class ClassText(html.parser.HTMLParser):
    """All text inside elements carrying a class, nested elements included (for per-class font subsets)."""

    def __init__(self, cls):
        super().__init__(convert_charrefs=True)
        self.cls, self.stack, self.depth, self.out = cls, [], 0, []

    def handle_starttag(self, tag, attrs):
        if tag in VOID_TAGS:
            return
        on = self.cls in (dict(attrs).get("class") or "").split()
        self.stack.append(on)
        self.depth += on

    def handle_endtag(self, tag):
        if tag not in VOID_TAGS and self.stack:
            self.depth -= self.stack.pop()

    def handle_data(self, d):
        if self.depth > 0:
            self.out.append(d)


def class_text(body, cls):
    p = ClassText(cls)
    p.feed(body)
    p.close()
    return "".join(p.out)


PRINT_MQ = [("@media (max-width:860px)", "@media screen and (max-width:860px)"),
            ("@media (max-width:520px)", "@media screen and (max-width:520px)"),
            ("@media (max-width:1199px)", "@media print, (max-width:1199px)"),
            ("@media (min-width:861px)", "@media print, (min-width:861px)"),
            ("@media (min-width:1000px)", "@media print, (min-width:1000px)")]


def print_safe(css):
    """Some PDF backends evaluate media queries against Letter width (816px), not the 1120px @page.
    Pin print to the layout a 1120px-wide page should get; screen rendering is unchanged."""
    for a_, b_ in PRINT_MQ:
        css = css.replace(a_, b_)
    return css


def css_selectors(css):
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    for sel in re.findall(r"([^{}@;]+)\{", css):
        for one in sel.split(","):
            one = one.strip()
            if one and not re.match(r"^(from|to|\d+%)$", one) and "(" not in one.split(":")[0]:
                yield one


def sel_parts(sel):
    """Compound selectors left to right, each as its set of classes."""
    return [set(re.findall(r"\.(-?[A-Za-z_][\w-]*)", c)) for c in re.split(r"\s*[\s>+~]\s*", sel.strip()) if c]


UTILITY = {"kai", "sans", "nw", "tag", "dot", "mono", "muted", "meta", "note", "px", "uiwall"}   # template classes meant for use
WRAPPERS = {"uiwall", "sec", "book", "stack", "blk", "sub", "spec", "cols"}                         # what agent HTML sits inside


def css_clashes(tpl_css, extra_css, agent_html):
    """(template classes the extra CSS restyles everywhere, template rules that also hit classes used in agent HTML)."""
    tpl_sel = list(css_selectors(tpl_css))
    tpl_all = set().union(*[set().union(*sel_parts(x)) for x in tpl_sel]) if tpl_sel else set()
    own = tpl_all - UTILITY
    restyle = set()
    for sel in css_selectors(extra_css or ""):
        parts = sel_parts(sel)
        if not parts:
            continue
        scoped = len(parts) > 1 and (parts[0] - own)          # starts from one of the agent's own classes (or .uiwall)
        if not scoped:
            restyle |= parts[-1] & own
    used = set(c for v in re.findall(r'class="([^"]*)"', agent_html or "") for c in v.split())
    hits = []
    for sel in tpl_sel:
        parts = sel_parts(sel)
        if parts and parts[-1] & used - UTILITY:
            need = set().union(*parts)
            if need <= used | WRAPPERS:
                hits.append(sel)
    return sorted(restyle), sorted(set(hits))


def lab_of_hex(hx):
    return rgb_to_lab(np.array(hex_to_rgb(hx), dtype=np.float64))


def colour_check(W, content, base, tpl_css=""):
    """Hex values in the document (content.json and the files it names) against every measured colour.
    Returns (n_total, unmatched [(hex, nearest, dE, where)], near [(hex, nearest, dE, source, where)]): `near` are colours
    2 to 6 DeltaE from a measured one (a similar colour, maybe of another object: the source is named), `unmatched` more."""
    mdb = jload(os.path.join(W, "measure.json"), {})
    mdb = dict(mdb, picks={k: v for k, v in (mdb.get("picks") or {}).items() if v.get("spread", 0) <= 12})    # a mixed-colour pick is no measurement
    src = {}    # measured hex -> where it was measured

    def walk(o, label):
        if isinstance(o, dict):
            for v in o.values():
                walk(v, label)
        elif isinstance(o, list):
            for v in o:
                walk(v, label)
        elif isinstance(o, str):
            for h in re.findall(r"#[0-9A-Fa-f]{6}\b", o):
                src.setdefault(h.upper(), []).append(label)
    kinds = {"frames": "frame", "profiles": "profile", "picks": "pick", "lights": "light", "post": "post", "pixel": "pixel"}
    for sec, ents in mdb.items():
        for name, v in (ents.items() if isinstance(ents, dict) else []):
            walk(v, f"{kinds.get(sec, sec)} {name}")
    for name, v in jload(os.path.join(W, "img", "manifest.json"), {}).items():
        walk(v, f"image {name}")
    pool = set(src)
    page = set(h.upper() for h in re.findall(r"#[0-9A-Fa-f]{6}\b", tpl_css))     # the page's own colours are not claims
    where = {}
    for sec in content.get("sections", []):
        for h in re.findall(r"#[0-9A-Fa-f]{6}\b", json.dumps(sec, ensure_ascii=False)):
            where.setdefault(h.upper(), set()).add(sec.get("id", "?"))
    for h in re.findall(r"#[0-9A-Fa-f]{6}\b", json.dumps({k: v for k, v in content.items() if k != "sections"}, ensure_ascii=False)):
        where.setdefault(h.upper(), set()).add("top")

    def files(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k.endswith("_file") and isinstance(v, str):
                    yield v
                else:
                    yield from files(v)
        elif isinstance(o, list):
            for v in o:
                yield from files(v)
    for f in set(files(content)):
        pth = f if os.path.isabs(f) else os.path.join(base, f)
        if os.path.exists(pth):
            for h in re.findall(r"#[0-9A-Fa-f]{6}\b", open(pth, encoding="utf-8", errors="replace").read()):
                where.setdefault(h.upper(), set()).add(os.path.basename(f))
    claims = sorted(h for h in where if h not in page)
    if not pool:
        return len(claims), [(h, None, None, sorted(where[h])) for h in claims], []
    P = np.array([lab_of_hex(h) for h in sorted(pool)])
    names = sorted(pool)
    bad, near = [], []
    for h in claims:
        d = np.sqrt(((P - lab_of_hex(h)) ** 2).sum(1))
        i = int(np.argmin(d))
        if d[i] > 6:
            bad.append((h, names[i], round(float(d[i]), 1), sorted(where[h])))
        elif d[i] > 2:
            near.append((h, names[i], round(float(d[i]), 1), " / ".join(sorted(set(src[names[i]]))[:2]), sorted(where[h])))
    return len(claims), bad, near


def cmd_build(a):
    content = jload(a.content)
    if content is None:
        die(f"{a.content} not found")
    W = a.work
    ctx = Ctx(W, content, os.path.dirname(os.path.abspath(a.content)))
    secs = content.get("sections", [])
    ids = [s.get("id") for s in secs]
    order = [i for i in ids if i in CANON]
    if content.get("check_sections", True):    # false for pages that are not style documents
        if order != sorted(order, key=CANON.index):
            ctx.warn.append(f"sections are not in the 琥珀微缩 order {CANON}: {ids}")
        for need in ("define", "palette", "l1", "l2", "l3", "l4", "l5", "l6", "l7", "rules", "recipe"):
            if need not in ids:
                ctx.warn.append(f"section '{need}' missing")
    # hero
    hero = content.get("hero", {})
    huri, hw, hh, _ = ctx.img(hero.get("img", "hero"))
    focus = f' style="object-position:{attr(hero["focus"])}"' if hero.get("focus") else ""
    parts = [f'<header class="hero"><img src="{huri}" alt="{attr(hero.get("alt", ""))}" width="{hw}"{focus}>'
             f'<div class="title-block"><div class="kick">{tx(content.get("kicker", ""))}</div><h1>{tx(content.get("title", ""))}</h1>'
             f'<div class="en">{tx(content.get("title_en", ""))}</div><p class="thesis">{tx(content.get("thesis", ""))}</p></div></header>', '<main class="book">']
    sig = []
    for s in content.get("sig", [])[:4]:
        if isinstance(s, dict):
            v, u, l = s.get("value", ""), s.get("unit", ""), s.get("label", "")
        else:
            v, u, l = (s[0], "", s[1]) if len(s) == 2 else s[:3]
        sig.append(f'<div><b>{tx(v)}{f"<span class=u>{tx(u)}</span>" if u else ""}</b><span>{tx(l)}</span></div>')
    if len(sig) != 4:
        ctx.warn.append(f"sig should have exactly 4 numbers (has {len(sig)})")
    parts.append(f'<div class="sig" aria-label="风格的四个标志数字">{"".join(sig)}</div>')
    toc = []
    for s in secs:
        if s.get("toc") is False:
            continue
        num = f'<b>{tx(s["num"])}</b>' if s.get("num") else ""
        toc.append(f'<a href="#{attr(s["id"])}">{num}{tx(s.get("toc", s.get("title", s["id"])))}</a>')
    parts.append(f'<nav class="toc" aria-label="目录">{"".join(toc)}</nav>')
    for s in secs:
        head = f'<div class="eyebrow">{tx(s.get("eyebrow", ""))}</div><h2 class="h">{tx(s.get("title", ""))}</h2>'
        if s.get("lede"):
            head += f'<p class="lede">{tx(s["lede"])}</p>'
        body, first = [], True
        for b in s.get("blocks", []):
            sub = (not first) and b.get("type") not in NO_SUB and b.get("sub", True)
            body.append(render_block(b, ctx, sub=sub))
            if b.get("type") not in ("lede", "note", "h3"):
                first = False
        parts.append(f'<section class="sec" id="{attr(s["id"])}">{head}{"".join(body)}</section>')
    col = content.get("colophon", {})
    links = "".join(f'<a href="{attr(l["href"])}" target="_blank" rel="noopener">{tx(l["label"])} ↗</a>' for l in col.get("links", []))
    parts.append(f'<footer class="colophon"><span>{tx(col.get("text", ""))}</span>{links}</footer></main>')
    body = "\n".join(parts)
    css = print_safe(open(os.path.join(HERE, "amber.css"), encoding="utf-8").read())
    ext = print_safe(open(os.path.join(HERE, "amber_ext.css"), encoding="utf-8").read())
    extra = ctx.raw(content, "extra_css")
    agent_html = "".join(ctx.raw(b, "html") for sec in secs for b in walk_blocks(sec.get("blocks", [])) if b.get("type") in ("uiwall", "html"))
    restyle, hits = css_clashes(css + ext, extra, agent_html)
    if restyle:
        ctx.warn.append("extra_css restyles template classes everywhere: " + ", ".join("." + c for c in restyle[:8])
                        + ". Fine if you mean to change the template there; otherwise scope the rule under your own class (e.g. .op-menu .row)")
    if hits:
        ctx.warn.append("template rules also style classes in your HTML: " + "; ".join(hits[:6])
                        + ". Fine if you are reusing that template widget; otherwise rename the classes in your HTML (give them a prefix of your own, e.g. op-; the template already uses ui-)")
    # fonts: the template's three families, plus `fonts` entries; a font with a class styles that class
    plain = html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r'src="data:[^"]+"', "", body)))
    plain += "0123456789 已复制已选中复制 Copy Copied Selected"        # strings the copy script writes
    font_html = ""
    extra_fonts = content.get("fonts", [])
    texts = [class_text(body, f["class"]) if f.get("class") else plain for f in extra_fonts]
    for f, t in zip(extra_fonts, texts):
        if f.get("class") and not t.strip():
            ctx.warn.append(f"font {f['family']}: no text with class '{f['class']}' in the document")
    cls_css = "".join(f'.{f["class"]}{{font-family:"{f["family"]}",var(--f-{f["class"]},serif)}}' for f in extra_fonts if f.get("class"))
    if a.embed_fonts:
        fams = [("Cinzel", "wght@500;700", plain), ("IBM Plex Mono", "wght@400;500", plain), ("Noto Serif SC", "wght@400;600;900", plain)]
        fams += [(f["family"], f.get("spec", ""), t) for f, t in zip(extra_fonts, texts)]
        try:
            fcss, failed = embed_fonts(fams)
            font_html = "<style>\n" + fcss + "\n</style>" if fcss.strip() else ""
            if failed:
                ctx.warn.append("some font subsets could not be embedded (those characters use the next font in the stack): " + "; ".join(failed[:4]))
        except Exception as e:
            ctx.warn.append(f"font embedding failed ({e}); falling back to Google Fonts links")
            font_html = ""
    if not font_html:
        font_html = (f'<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
                     f'<link rel="stylesheet" href="{FONT_LINK}">')
        for f, t in zip(extra_fonts, texts):
            chars = "".join(sorted(set(ch for ch in t if ch not in " \t\r\n\f\v")))
            fam = f["family"].replace(" ", "+") + (":" + f["spec"] if f.get("spec") else "")
            font_html += f'<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family={fam}&text={urllib.parse.quote(chars, safe="")}&display=swap">'
    title = attr(content.get("doc_title") or content.get("title", "风格文档"))
    head_html = (f'<title>{title}</title>\n{font_html}\n<style>\n{css}\n</style>\n<style>\n{ext}\n</style>'
                 + (f"\n<style>{cls_css}</style>" if cls_css else "") + (f"\n<style>\n{extra}\n</style>" if extra else ""))
    if content.get("check_sections", True):
        n_hex, bad, near = colour_check(W, content, ctx.base, css + ext)
        if bad or near:
            ctx.notes.append(f"colours: {n_hex - len(bad) - len(near)} of {n_hex} hex values in the document match a measured colour (ΔE 2 or less)"
                             + (f"; {len(near)} only come close (ΔE 2–6), so check that each is the same thing and not a similar colour of something else: "
                                + ", ".join(f"{h} ({'/'.join(w)}) ≈ {n} of {src_} ΔE {d}" for h, n, d, src_, w in sorted(near, key=lambda x: -x[2])[:8]) if near else "")
                             + (f"; {len(bad)} match nothing measured: "
                                + ", ".join(f"{h} ({'/'.join(w)}; nearest {n} ΔE {d})" if n else f"{h} ({'/'.join(w)})" for h, n, d, w in bad[:10]) if bad else "")
                             + ". `facts content.json` lists them all. Tag the unmeasured ones 起调/目测, or use a measured value (measure, pick).")
    if a.fragment:
        doc = f"{head_html}\n{body}\n{COPY_JS}\n"
    else:
        doc = (f'<!doctype html>\n<html lang="{attr(content.get("lang", "zh-CN"))}">\n<head>\n<meta charset="utf-8">\n'
               f'<meta name="viewport" content="width=device-width,initial-scale=1">\n{head_html}\n</head>\n<body>\n{body}\n{COPY_JS}\n</body>\n</html>\n')
    leftovers = re.findall(r"\{\{[^}]*\}\}|\}\}", re.sub(r"<style>.*?</style>", "", doc, flags=re.S))
    if leftovers:
        ctx.warn.append(f"unexpanded shorthands: {leftovers[:5]}")
    tags = {}
    for m in re.findall(r'<span class="tag">([^<]+)</span>', body):
        tags[m] = tags.get(m, 0) + 1
    out = a.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    for w in ctx.warn:
        print("warning:", w)
    for e in ctx.errors:
        print("ERROR:", e)
    for n in ctx.notes:
        print("note:", n)
    print(f"{out}: {len(doc.encode()) / 1e6:.2f} MB, {len(secs)} sections, {len(ctx.uris)} images, evidence tags {tags}")
    if ctx.errors:
        sys.exit(2)


# ───────────────────────── browser: pdf + screenshots ─────────────────────────
PROBE_JS = r"""() => {
  const vw = document.documentElement.clientWidth, sec = el => (el.closest('section.sec') || {}).id || 'top';
  const name = el => el.tagName.toLowerCase() + (typeof el.className === 'string' && el.className ? '.' + el.className.trim().split(/\s+/).join('.') : '');
  const scrollers = [], wide = [];
  for (const el of document.querySelectorAll('body *')) {
    if (el.scrollWidth > el.clientWidth + 2 && ['auto', 'scroll'].includes(getComputedStyle(el).overflowX))
      scrollers.push({sec: sec(el), el: name(el), need: el.scrollWidth, have: el.clientWidth});
  }
  const page = document.documentElement.scrollWidth > vw + 1;
  if (page) for (const el of document.querySelectorAll('body *')) {
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.right > vw + 1 && !el.closest('[style*="overflow"], .tbl, .lineup, .diagram')) { wide.push({sec: sec(el), el: name(el), right: Math.round(r.right)}); if (wide.length >= 6) break; }
  }
  return {page, vw, scrollers: scrollers.slice(0, 40), wide};
}"""

FONT_TAG_JS = r"""() => {
  const out = [], sec = el => (el.closest('section.sec') || {}).id || 'top';
  const cls = el => typeof el.className === 'string' ? el.className : ((el.className && el.className.baseVal) || '');
  const name = el => el.tagName.toLowerCase() + (cls(el).trim() ? '.' + cls(el).trim().split(/\s+/).join('.') : '');
  for (const el of document.querySelectorAll('body *')) {
    if (['SCRIPT', 'STYLE', 'NOSCRIPT', 'TITLE'].includes(el.tagName)) continue;
    let own = '';
    for (const n of el.childNodes) if (n.nodeType === 3) own += n.textContent;
    own = own.replace(/\s+/g, ' ').trim();
    if (!own) continue;
    el.setAttribute('data-sxf', String(out.length));
    out.push({sec: sec(el), el: name(el), text: own.slice(0, 40)});
  }
  return out;
}"""

NODE_JOB = r"""
const pw = require(process.argv[2]);
const job = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));
(async () => {
  let br;
  try { br = await pw.chromium.launch({args: ['--no-sandbox']}); }
  catch (e) { if (!job.chrome) throw e; br = await pw.chromium.launch({args: ['--no-sandbox'], executablePath: job.chrome}); }
  job.probe = eval(job.probe_js);
  const open = async (w) => {
    const pg = await br.newPage({viewport: {width: w, height: 900}});
    await pg.goto(job.url, {waitUntil: 'networkidle', timeout: 180000});
    await pg.evaluate(() => document.fonts.ready.then(() => 1));
    await pg.waitForTimeout(400);
    return pg;
  };
  const fontcheck = async (pg) => {   // text drawn with a font that is not one of the page's own web fonts (CDP)
    try {
      await pg.emulateMedia({media: 'print'});
      const items = await pg.evaluate(eval(job.font_js));
      const cdp = await pg.context().newCDPSession(pg);
      await cdp.send('DOM.enable'); await cdp.send('CSS.enable');
      const root = (await cdp.send('DOM.getDocument', {depth: 0})).root.nodeId;
      const ids = (await cdp.send('DOM.querySelectorAll', {nodeId: root, selector: '[data-sxf]'})).nodeIds;
      const found = [];
      for (const nid of ids) {
        const fonts = ((await cdp.send('CSS.getPlatformFontsForNode', {nodeId: nid})).fonts) || [];
        const bad = fonts.filter(f => !f.isCustomFont);
        if (bad.length) {
          const at = (await cdp.send('DOM.getAttributes', {nodeId: nid})).attributes;
          found.push({...items[parseInt(at[at.indexOf('data-sxf') + 1])], fonts: bad.map(f => ({family: f.familyName || '?', glyphs: f.glyphCount || 0}))});
        }
      }
      await pg.evaluate(() => document.querySelectorAll('[data-sxf]').forEach(e => e.removeAttribute('data-sxf')));
      return found;
    } catch (e) { return {error: String(e && e.message || e)}; }
  };
  const probes = {};
  const widths = [...new Set((job.shots || []).map(s => s.width))];
  for (const w of widths) {
    const pg = await open(w);
    for (const s of job.shots.filter(s => s.width === w)) {
      if (s.sel) await pg.locator(s.sel).screenshot({path: s.path}); else await pg.screenshot({path: s.path, fullPage: true});
    }
    probes[w] = await pg.evaluate(job.probe); await pg.close();
  }
  if (job.pdf || job.fontcheck) {
    const pg = await open(job.pdf ? job.pdf.width : 1120);
    if (job.fontcheck) probes.__fonts = await fontcheck(pg);
    if (job.pdf) await pg.pdf({path: job.pdf.path, printBackground: true, preferCSSPageSize: true, width: job.pdf.width + 'px', height: job.pdf.height + 'px'});
    await pg.close();
  }
  if (job.result) require('fs').writeFileSync(job.result, JSON.stringify(probes));
  await br.close();
})().catch(e => { console.error(String(e && e.stack || e)); process.exit(1); });
"""


def find_chrome():
    cands = [os.environ.get("CHROME_PATH", "")]
    for base in [os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""), os.path.expanduser("~/.cache/ms-playwright"), "/opt/pw-browsers"]:
        if base:
            cands += sorted(glob.glob(os.path.join(base, "chromium-*", "chrome-linux*", "chrome")), reverse=True)
            cands += sorted(glob.glob(os.path.join(base, "chromium_headless_shell-*", "chrome-linux*", "headless_shell")), reverse=True)
    cands += [shutil.which(x) or "" for x in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "chrome")]
    cands += ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "C:/Program Files/Google/Chrome/Application/chrome.exe"]
    return next((c for c in cands if c and os.path.exists(c)), None)


def node_playwright():
    if not shutil.which("node"):
        return None
    roots = []
    try:
        roots.append(run(["npm", "root", "-g"], text=True).stdout.strip())
    except Exception:
        pass
    roots += [os.path.join(os.getcwd(), "node_modules"), "/usr/lib/node_modules", "/usr/local/lib/node_modules"]
    for r in roots:
        for name in ("playwright", "playwright-core"):
            if r and os.path.exists(os.path.join(r, name, "package.json")):
                return os.path.join(r, name)
    return None


def font_fallbacks_py(pg):
    """Elements of the print layout whose text is drawn with a font that is not one of the page's own web fonts (Chromium's
    CSS.getPlatformFontsForNode): [{sec, el, text, fonts: [{family, glyphs}]}], or {error} when the browser cannot say."""
    try:
        pg.emulate_media(media="print")
        items = pg.evaluate(FONT_TAG_JS)
        cdp = pg.context.new_cdp_session(pg)
        cdp.send("DOM.enable")
        cdp.send("CSS.enable")
        root = cdp.send("DOM.getDocument", {"depth": 0})["root"]["nodeId"]
        found = []
        for nid in cdp.send("DOM.querySelectorAll", {"nodeId": root, "selector": "[data-sxf]"})["nodeIds"]:
            bad = [f for f in cdp.send("CSS.getPlatformFontsForNode", {"nodeId": nid}).get("fonts", []) if not f.get("isCustomFont")]
            if bad:
                at = cdp.send("DOM.getAttributes", {"nodeId": nid})["attributes"]
                found.append({**items[int(at[at.index("data-sxf") + 1])],
                              "fonts": [{"family": f.get("familyName", "?"), "glyphs": f.get("glyphCount", 0)} for f in bad]})
        pg.evaluate("() => document.querySelectorAll('[data-sxf]').forEach(e => e.removeAttribute('data-sxf'))")
        return found
    except Exception as e:
        return {"error": str(e)}


def report_fonts(found, indent="   "):
    """Print what font_fallbacks found: which elements use a system font, and what to do."""
    if found is None:
        return
    if isinstance(found, dict):
        print(f"{indent}font check skipped: {found.get('error', '?')[:160]}")
        return
    if not found:
        print(f"{indent}fonts: all text of the print layout is drawn with the document's own fonts")
        return
    by = {}
    for it in found:
        for f in it["fonts"]:
            d = by.setdefault(f["family"], {"glyphs": 0, "els": []})
            d["glyphs"] += f["glyphs"]
            d["els"].append(it)
    for fam, d in sorted(by.items(), key=lambda kv: -kv[1]["glyphs"]):
        print(f"{indent}system font {fam}: {d['glyphs']} glyphs in {len(d['els'])} elements, e.g. "
              + "; ".join(f"#{e['sec']} {e['el']} “{e['text']}”" for e in d["els"][:5]) + (" …" if len(d["els"]) > 5 else ""))
    print(f"{indent}→ a character there is in none of the document's fonts, or the element's font stack has no font with that script: swap the character,"
          " put 'Noto Serif SC' (Chinese) or 'IBM Plex Mono' (Latin) into that element's font stack, or add a `fonts` entry that has it. A stray σ, ≤ or arrow is harmless")


def browser_job(job):
    job["chrome"] = find_chrome()
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            try:
                br = p.chromium.launch(args=["--no-sandbox"])
            except Exception:
                if not job["chrome"]:
                    raise
                br = p.chromium.launch(args=["--no-sandbox"], executable_path=job["chrome"])

            def open_(w):
                pg = br.new_page(viewport={"width": w, "height": 900})
                pg.goto(job["url"], wait_until="networkidle", timeout=180000)
                pg.evaluate("() => document.fonts.ready.then(() => 1)")
                pg.wait_for_timeout(400)
                return pg
            for w in sorted(set(s["width"] for s in job.get("shots", []))):
                pg = open_(w)
                for s in [s for s in job["shots"] if s["width"] == w]:
                    if s.get("sel"):
                        pg.locator(s["sel"]).screenshot(path=s["path"])
                    else:
                        pg.screenshot(path=s["path"], full_page=True)
                job.setdefault("probes", {})[str(w)] = pg.evaluate(PROBE_JS)
                pg.close()
            if job.get("pdf") or job.get("fontcheck"):
                pg = open_(job["pdf"]["width"] if job.get("pdf") else 1120)
                if job.get("fontcheck"):
                    job["fonts"] = font_fallbacks_py(pg)
                if job.get("pdf"):
                    pg.pdf(path=job["pdf"]["path"], print_background=True, prefer_css_page_size=True,
                           width=f'{job["pdf"]["width"]}px', height=f'{job["pdf"]["height"]}px')
                pg.close()
            br.close()
        return "python-playwright"
    except ImportError:
        pass
    pw = node_playwright()
    if pw:
        with tempfile.TemporaryDirectory() as td:
            js, jf, res = os.path.join(td, "job.js"), os.path.join(td, "job.json"), os.path.join(td, "probes.json")
            open(js, "w").write(NODE_JOB)
            json.dump({**job, "probe_js": PROBE_JS, "font_js": FONT_TAG_JS, "result": res}, open(jf, "w"))
            r = subprocess.run(["node", js, pw, jf], capture_output=True, text=True)
            if r.returncode != 0:
                die("node playwright failed: " + r.stderr[-1500:])
            if os.path.exists(res):
                job["probes"] = json.load(open(res))
                job["fonts"] = job["probes"].pop("__fonts", None)
        return "node-playwright"
    if job.get("pdf") and job["chrome"]:
        r = subprocess.run([job["chrome"], "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                            "--run-all-compositor-stages-before-draw", "--virtual-time-budget=30000",
                            f"--print-to-pdf={job['pdf']['path']}", job["url"]], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(job["pdf"]["path"]):
            die("chromium --print-to-pdf failed: " + r.stderr[-1500:])
        if job.get("shots"):
            print("warning: screenshots need Playwright (pip install playwright); skipped", file=sys.stderr)
        return "chromium-cli"
    die("No headless browser found. Install one: `pip install playwright && python -m playwright install chromium` "
        "(or set CHROME_PATH to a Chrome/Chromium binary).")


def pdf_pages(path):
    try:
        from pypdf import PdfReader
        r = PdfReader(path)
        b = r.pages[0].mediabox
        return len(r.pages), (float(b.width), float(b.height))
    except Exception:
        try:
            out = run(["pdfinfo", path], text=True).stdout
            n = int(re.search(r"Pages:\s+(\d+)", out).group(1))
            m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", out)
            return n, (float(m.group(1)), float(m.group(2)))
        except Exception:
            return None, None


def uncovered_chars(doc):
    """Characters of the page text that none of the document's embedded font faces has: they are drawn with a system font.
    None when that cannot be told (fonts linked, not embedded; fontTools missing)."""
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return None
    cov = set()
    for m in re.finditer(r"url\(data:font/(?:woff2|woff|ttf);base64,([^)]+)\)", doc):
        try:
            cov |= set(TTFont(io.BytesIO(base64.b64decode(m.group(1)))).getBestCmap())
        except Exception:
            return None
    if not cov:
        return None
    text = html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<(style|script)[^>]*>.*?</\1>", " ", doc, flags=re.S)))
    return sorted(ch for ch in set(text) if ch not in " \t\r\n\f\v" and ord(ch) > 31 and ord(ch) not in cov)


def static_fonts(doc):
    """Chromium writes variable fonts (Google serves Noto Serif SC and Cinzel that way) into a PDF as Type 3 outlines,
    about 5 MB for a Chinese document. Pinning each @font-face to a static instance of its weight embeds them as
    ordinary subset fonts and makes the PDF about 40% smaller. Needs fontTools + brotli; returns (doc, note)."""
    try:
        from fontTools.ttLib import TTFont
        from fontTools.varLib import instancer
        import brotli  # noqa: F401  (woff2)
    except ImportError:
        return doc, "variable fonts kept (pip install fonttools brotli makes the PDF about 40% smaller)"
    text = html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<(style|script)[^>]*>.*?</\1>", " ", doc, flags=re.S)))
    text += "0123456789"

    def linked(m):    # a Google Fonts stylesheet link -> the same faces, subset to the text, inline
        q = urllib.parse.parse_qs(urllib.parse.urlparse(html.unescape(m.group(1))).query)
        fams = []
        for f in q.get("family", []):
            fam, _, spec = f.partition(":")
            fams.append((fam, spec, q["text"][0] if q.get("text") else text))
        fcss, failed = embed_fonts(fams)
        if failed or not fcss.strip():
            raise RuntimeError("; ".join(failed[:2]) or "no fonts")
        return "<style>\n" + fcss + "\n</style>"
    n = [0]

    def face(m):
        block = m.group(0)
        w = re.search(r"font-weight:\s*(\d+)", block)
        u = re.search(r"url\(data:font/(?:woff2|woff|ttf);base64,([^)]+)\)\s*format\('[^']+'\)", block)
        if not u:
            return block
        f = TTFont(io.BytesIO(base64.b64decode(u.group(1))))
        if "fvar" not in f:
            return block
        loc = {}
        for ax in f["fvar"].axes:
            v = int(w.group(1)) if (ax.axisTag == "wght" and w) else ax.defaultValue
            loc[ax.axisTag] = max(ax.minValue, min(ax.maxValue, v))
        st = instancer.instantiateVariableFont(f, loc)
        st.flavor = None
        wt = loc.get("wght")
        if wt:    # the default instance's name (NotoSerifSC-ExtraLight) would stay on every weight
            wn = {100: "Thin", 200: "ExtraLight", 300: "Light", 400: "Regular", 500: "Medium", 600: "SemiBold", 700: "Bold", 800: "ExtraBold", 900: "Black"}.get(int(round(wt / 100) * 100), str(int(wt)))
            for rec in st["name"].names:
                if rec.nameID == 6:
                    fam = rec.toUnicode().split("-")[0]
                    fam = re.sub(r"(Thin|ExtraLight|UltraLight|Light|Regular|Medium|SemiBold|DemiBold|Bold|ExtraBold|UltraBold|Black|Heavy)$", "", fam) or fam
                    rec.string = fam + "-" + wn
        b = io.BytesIO()
        st.save(b)
        n[0] += 1
        return block.replace(u.group(0), f"url(data:font/ttf;base64,{base64.b64encode(b.getvalue()).decode()}) format('truetype')")
    try:
        doc = re.sub(r'<link rel="preconnect"[^>]*>', "", doc)
        doc = re.sub(r'<link rel="stylesheet" href="(https://fonts\.googleapis\.com/[^"]+)">', linked, doc)
        doc = re.sub(r"@font-face\s*\{[^}]*\}", face, doc)
    except Exception as e:
        return None, f"static fonts failed ({e}); variable fonts kept"
    return doc, f"{n[0]} variable font faces pinned to static instances"


def page_text(pdf, n):
    try:
        return re.sub(r"\s+", " ", run(["pdftotext", "-f", str(n), "-l", str(n), pdf, "-"], text=True).stdout).strip()
    except Exception:
        return ""


def blank_pages(pdf, dpi=24):
    """Pages that show nothing: no text, no picture, only the page ground (at most a stray border line). A block whose
    content fits a page exactly can push its last bit of padding or border onto the next one, and the next section's page
    break then leaves that page empty. None when pdftoppm / pdftotext are missing."""
    if not (shutil.which("pdftoppm") and shutil.which("pdftotext")):
        return None
    out = []
    with tempfile.TemporaryDirectory() as td:
        run(["pdftoppm", "-r", str(dpi), "-png", pdf, os.path.join(td, "p")])
        for i, p in enumerate(sorted(glob.glob(os.path.join(td, "p*.png"))), 1):
            if page_end(p) < 0.12 and not page_text(pdf, i):
                has_img = shutil.which("pdfimages") and len(run(["pdfimages", "-list", "-f", str(i), "-l", str(i), pdf], text=True).stdout.strip().splitlines()) > 2
                if not has_img:
                    out.append(i)
    return out


def drop_pages(pdf, pages):
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        return False
    r = PdfReader(pdf)
    w = PdfWriter()
    for i, p in enumerate(r.pages, 1):
        if i not in pages:
            w.add_page(p)
    tmp = pdf + ".tmp"
    with open(tmp, "wb") as f:
        w.write(f)
    os.replace(tmp, pdf)
    return True


def pdf_breakdown(pdf):
    """Where the bytes of a PDF are: the document's pictures, page-sized bitmaps (what Chromium makes of a gradient with
    transparent stops), embedded fonts, the rest. Returns (one line, bytes of page-sized bitmaps), or None without pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        r = PdfReader(pdf)
        pb = r.pages[0].mediabox
        pw, ph = float(pb.width), float(pb.height)
        nums = set()
        for d in r.xref.values():
            nums |= set(d)
        t = {"pic": [0, 0], "page": [0, 0], "font": [0, 0]}
        for num in nums:
            try:
                o = r.get_object(num)
            except Exception:
                continue
            if o is None or not hasattr(o, "_data"):
                continue
            n = len(o._data or b"")
            if o.get("/Subtype") == "/Image":
                full = abs(o.get("/Width", 0) - pw) <= 2 and abs(o.get("/Height", 0) - ph) <= 2
                t["page" if full else "pic"][0] += n
                t["page" if full else "pic"][1] += 1
            elif "/Length1" in o or o.get("/Subtype") in ("/Type1C", "/CIDFontType0C", "/OpenType"):
                t["font"][0] += n
                t["font"][1] += 1
        tot = os.path.getsize(pdf)
        rest = tot - sum(v[0] for v in t.values())
        return (f"{tot / 1e6:.1f} MB = document pictures {t['pic'][0] / 1e6:.1f} ({t['pic'][1]}) + page-sized bitmaps {t['page'][0] / 1e6:.1f} ({t['page'][1]})"
                f" + embedded fonts {t['font'][0] / 1e6:.1f} ({t['font'][1]}) + the rest {rest / 1e6:.1f}"), t["page"][0]
    except Exception:
        return None


def cmd_pdf(a):
    src = os.path.abspath(a.html)
    out = os.path.abspath(a.out)
    pw, ph = [int(v) for v in a.page.lower().split("x")]
    doc = open(src, encoding="utf-8").read()
    note = "variable fonts kept (--keep-fonts)"
    if not a.keep_fonts:
        d2, note = static_fonts(doc)
        doc = d2 or doc
    if a.page != "1120x1584":    # override the @page size from amber_ext.css
        doc = doc.replace("</head>", f"<style>@page{{size:{pw}px {ph}px;margin:0}}</style></head>", 1) if "</head>" in doc else f"<style>@page{{size:{pw}px {ph}px;margin:0}}</style>" + doc
    tmp = os.path.join(os.path.dirname(src), f".{os.path.basename(src)}.print.html")    # next to the HTML, so relative paths still work
    open(tmp, "w", encoding="utf-8").write(doc)
    job = {"url": "file://" + tmp, "pdf": {"path": out, "width": pw, "height": ph}, "fontcheck": not a.nofontcheck}
    try:
        how = browser_job(job)
    finally:
        os.remove(tmp)
    blank = blank_pages(out)
    if blank and not a.keep_blank:
        kept = drop_pages(out, blank)
        print(("removed " if kept else "empty page(s) found, not removed (pip install pypdf): ") + ", ".join(f"p{n}" for n in blank)
              + ": nothing on them but the page ground, the padding or border of the block before that spilled over")
    n, size = pdf_pages(out)
    print(f"{out}: {os.path.getsize(out) / 1e6:.2f} MB, {n} pages, page {size} pt  (via {how}; {note})")
    bd = pdf_breakdown(out)
    if bd:
        print("   " + bd[0] + (".  Page-sized bitmaps are what Chromium makes of every gradient with a transparent stop (about 60 KB each): give those"
                                " elements a flat translucent fill in print CSS (@media print), as the template does for its own" if bd[1] > 1.5e6 else ""))
    report_fonts(job.get("fonts"))


def tile(png, out_prefix, tile_h_ratio=1.5, out_w=900):
    im = Image.open(png).convert("RGB")
    w, h = im.size
    th = int(w * tile_h_ratio)
    outs = []
    for i, y in enumerate(range(0, h, th)):
        t = im.crop((0, y, w, min(h, y + th)))
        t = t.resize((out_w, max(1, round(t.height * out_w / w))), Image.LANCZOS) if w > out_w else t
        p = f"{out_prefix}_{i + 1:02d}.jpg"
        t.save(p, quality=80)
        outs.append(p)
    return outs, (w, h)


def strips(tiles, out_prefix, per=4, gap=12):
    """Phone tiles side by side, `per` to an image, so a long page is a handful of images to look at."""
    outs = []
    for i in range(0, len(tiles), per):
        ims = [Image.open(t).convert("RGB") for t in tiles[i:i + per]]
        sh = Image.new("RGB", (sum(m.width for m in ims) + gap * (len(ims) + 1), max(m.height for m in ims) + 2 * gap), (60, 60, 66))
        x = gap
        for m in ims:
            sh.paste(m, (x, gap))
            x += m.width + gap
        p = f"{out_prefix}_{i // per + 1:02d}.jpg"
        sh.save(p, quality=80)
        outs.append(p)
    for t in tiles:
        if t not in outs:
            os.remove(t)
    return outs


def page_end(png, bg=(11, 13, 24)):
    """Fraction of the page height where its content ends (dark page ground counts as empty)."""
    a = np.asarray(Image.open(png).convert("RGB")).astype(np.int16)
    a = a[2:-2, 2:-2]    # the raster's outermost row and column come out white at some resolutions
    rows = np.where((np.abs(a - np.array(bg)).sum(2) > 45).mean(1) > 0.004)[0]
    return 0.0 if not len(rows) else (rows[-1] + 1) / a.shape[0]


def cmd_shots(a):
    src = os.path.abspath(a.html)
    out = os.path.abspath(a.out or os.path.join(os.path.dirname(src), "qa"))
    os.makedirs(out, exist_ok=True)
    widths = [int(x) for x in a.widths.split(",")] if a.widths else []
    only = [x.strip().lstrip("#") for x in a.only.split(",")] if a.only else []
    if only:
        shots = [{"width": w, "sel": f"#{i}", "tag": f"{i}_", "path": os.path.join(out, f"sec_{w}_{i}.png")} for w in widths for i in only]
    else:
        shots = [{"width": w, "tag": "", "path": os.path.join(out, f"full_{w}.png")} for w in widths]
    if shots:
        job = {"url": "file://" + src, "shots": shots}
        browser_job(job)
        if not only:
            for w in widths:
                for f in glob.glob(os.path.join(out, f"w{w}_*.jpg")):
                    os.remove(f)
        for s in shots:
            if not os.path.exists(s["path"]):
                continue
            pref = os.path.join(out, f"w{s['width']}_{s['tag']}".rstrip("_"))
            for f in glob.glob(pref + "_*.jpg"):
                os.remove(f)
            phone = s["width"] <= 600
            tiles, size = tile(s["path"], pref, 2.2 if phone else 1.5, 390 if phone else 900)
            os.remove(s["path"])
            if phone:
                tiles = strips(tiles, pref)
            what = f"#{s['sel'][1:]}" if s.get("sel") else "page"
            print(f"width {s['width']}: {what} {size[0]}x{size[1]} px -> {len(tiles)} images {tiles[0]}" + (f" … {tiles[-1]}" if len(tiles) > 1 else "")
                  + ("  (phone screens side by side, left to right)" if phone else ""))
        for w in widths:
            pr = (job.get("probes") or {}).get(str(w))
            if not pr:
                continue
            if pr["page"]:
                print(f"   width {w}: problem: the page is wider than the screen (scrolls sideways); sticking out: "
                      + ", ".join(f"{x['el']} in #{x['sec']} (to {x['right']} px)" for x in pr["wide"]))
            sc = [x for x in pr["scrollers"] if x["need"] > x["have"] + 2 and (not only or x["sec"] in only)]
            if sc:
                print(f"   width {w}: scroll boxes (content wider than the box; the screenshots show only the left part): "
                      + "; ".join(f"#{x['sec']} {x['el']} {x['need']}>{x['have']} px" for x in sc[:12]))
    if a.pdf:
        if not shutil.which("pdftoppm"):
            print("pdftoppm not found: skip PDF page previews")
            return
        pref = os.path.join(out, "pdfpage")
        for f in glob.glob(pref + "*.png") + glob.glob(os.path.join(out, "pdf_pages_*.jpg")):
            os.remove(f)
        run(["pdftoppm", "-r", str(a.dpi), "-png", os.path.abspath(a.pdf), pref])
        pages = sorted(glob.glob(pref + "*.png"))
        per = 8
        for s in range(0, len(pages), per):
            ims = [Image.open(p).convert("RGB") for p in pages[s:s + per]]
            cw = max(i.width for i in ims)
            chh = max(i.height for i in ims)
            sheet = Image.new("RGB", (4 * (cw + 10) + 10, ((len(ims) + 3) // 4) * (chh + 30) + 10), (60, 60, 66))
            d = ImageDraw.Draw(sheet)
            for i, im in enumerate(ims):
                x, y = 10 + (i % 4) * (cw + 10), 10 + (i // 4) * (chh + 30)
                sheet.paste(im, (x, y))
                d.text((x, y + chh + 4), f"p{s + i + 1}", font=font(16), fill=(240, 240, 240))
            p = os.path.join(out, f"pdf_pages_{s // per + 1:02d}.jpg")
            sheet.save(p, quality=82)
            print("pdf sheet", p)
        ends = [page_end(p) for p in pages]
        pdfp = os.path.abspath(a.pdf)
        _, psize = pdf_pages(pdfp)
        page_px = (psize[1] * 96 / 72) if psize else 1584
        short_ok = []
        for n, e in [(i + 1, e) for i, e in enumerate(ends) if e < 0.25 and i > 0]:
            txt = page_text(pdfp, n) if shutil.which("pdftotext") else ""
            before = ends[n - 2]
            if not txt and e < 0.12:
                print(f"   p{n} is empty (only the page ground): the padding or border of the last block of p{n - 1} spilled onto it, and the next section's page break "
                      "left it empty. `pdf` removes such pages; if you built the PDF with --keep-blank, run it again")
            elif before >= 0.8:    # the page before is full: the section ran over
                px = max(0, e * page_px - 60)
                print(f"   p{n} is nearly empty: it holds only about {px:.0f} px of content (≈ {max(1, round(px / 28))} lines of text), «{txt[:60]}», and the rest of the page is empty;"
                      f" p{n - 1} ends at {before:.0%}. Cut about that much from the end of the section on p{n - 1} (shorten a note, merge rows, drop a sentence)"
                      " or split a big block of it into two smaller ones, so the section fits its pages")
            else:
                short_ok.append(f"p{n} ({e:.0%})")
        if short_ok:
            print("   short pages that follow a page with room left, so they are short sections or a section's tail, not a spill: " + ", ".join(short_ok))
        for f in pages:
            os.remove(f)
        if shutil.which("pdffonts"):
            fonts = sorted(set(re.sub(r"^[A-Z]{6}\+", "", l.split()[0]) for l in run(["pdffonts", os.path.abspath(a.pdf)], text=True).stdout.splitlines()[2:] if l.strip()))
            doc = open(src, encoding="utf-8").read()
            loaded = set(re.findall(r"family=([^&:\"]+)", html.unescape(doc))) | set(re.findall(r"font-family:\s*'([^']+)'", doc))
            known = tuple({"Cinzel", "IBMPlexMono", "NotoSerifSC"} | {re.sub(r"[ +]", "", f) for f in loaded})
            odd = [f for f in fonts if not f.startswith(known)]
            print("   PDF fonts: " + ", ".join(fonts) + (f"   ← not loaded by the document (system fonts of this machine): {', '.join(odd)}."
                  " A few symbols (σ, ≤, arrows) falling back is fine; whole captions in them mean a font stack lacks Chinese,"
                  " or text in .sans/.kai that a `fonts` entry with that class would pin." if odd else ""))
            miss = uncovered_chars(doc) if odd else None
            if miss:
                print("   characters that no embedded font has (the ones drawn in the system font): "
                      + " ".join(f"{c}(U+{ord(c):04X})" for c in miss[:24]) + (" …" if len(miss) > 24 else "")
                      + ". Swap them for ones the fonts have, or add a `fonts` entry that does.")
            if odd:
                fj = {"url": "file://" + src, "fontcheck": True}
                try:
                    browser_job(fj)
                    report_fonts(fj.get("fonts"))
                except SystemExit:
                    pass


# ───────────────────────── CLI ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        p = sp.add_parser(name, help=help_)
        p.set_defaults(fn=fn)
        return p

    p = add("probe", cmd_probe, "inventory inputs")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--work", "-w", required=True)
    p = add("sample", cmd_sample, "sample frames + contact sheets")
    p.add_argument("--work", "-w", required=True)
    p.add_argument("--codes")
    p.add_argument("--fps", type=float, default=2.0, help="analysis rate (default 2 fps, lowered for long videos)")
    p.add_argument("--cap", type=float, default=900, help="max analysed samples per video")
    p.add_argument("--every", type=float, default=0, help="uniform pick spacing in s (default duration/30, 2–10 s)")
    p.add_argument("--max", type=int, default=48, help="max kept frames per video")
    p.add_argument("--cut", type=float, default=0.08, help="shot-cut threshold")
    p.add_argument("--dedup", type=float, default=0.025)
    p = add("overlay", cmd_overlay, "detect static watermarks / overlays")
    p.add_argument("--work", "-w", required=True)
    p.add_argument("--codes")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--edge", type=float, default=18.0)
    p = add("crop", cmd_overlay_set, "set crop boxes by hand: C=x0,y0,x1,y1")
    p.add_argument("--work", "-w", required=True)
    p.add_argument("set", nargs="+")
    p = add("grab", cmd_grab, "full-res frames")
    p.add_argument("--work", "-w", required=True)
    p.add_argument("refs", nargs="+")
    p.add_argument("--force", action="store_true")
    p = add("zoom", cmd_zoom, "view a frame or region with a coordinate grid (nearest-neighbour when enlarging)")
    p.add_argument("--work", "-w", required=True)
    p.add_argument("ref", help="frame: C_140.0, C@2:20, or image code A")
    p.add_argument("box", nargs="*", type=int, help="x0 y0 x1 y1 in full-frame pixels (default: whole frame)")
    p.add_argument("--scale", type=float, default=0, help="default: fit 1600 px, at most 8x")
    p.add_argument("--grid", type=int, default=0, help="grid spacing in frame pixels (default: about 10 lines)")
    p.add_argument("--nogrid", action="store_true")
    p.add_argument("--out")
    p = add("sheet", cmd_sheet, "contact sheet of images")
    p.add_argument("images", nargs="+", help="images (or OUT.jpg first, then the images)")
    p.add_argument("--out", "-o")
    p.add_argument("--cell", type=int, default=0, help="cell width in px (default: from the number of images)")
    p.add_argument("--cols", type=int, default=0, help="columns (default: from the number of images)")
    p.add_argument("--per", type=int, default=0, help="images per sheet before starting the next one (default: what fits in about 1100 px of height)")
    p = add("measure", cmd_measure, "colour / light / lens measurement")
    p.add_argument("--work", "-w", required=True)
    p.add_argument("refs", nargs="+")
    p.add_argument("--box", nargs=4, type=int, help="x0 y0 x1 y1 in full-frame pixels (as on the zoom grid), or put the 4 numbers "
                   "right after the frame; the region is stored under its own name (FRAME@x0,y0,x1,y1) unless --name is given")
    p.add_argument("--nocrop", action="store_true")
    p.add_argument("--name")
    p = add("profile", cmd_profile, "sharpness profile over a time range")
    p.add_argument("--work", "-w", required=True)
    p.add_argument("code", help="one video code, or image codes joined by commas (A,B,C)")
    p.add_argument("t0", nargs="?")
    p.add_argument("t1", nargs="?")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--name")
    p.add_argument("--keep", type=float, default=0.6, help="fraction of peak counted as sharp")
    p.add_argument("--bands", type=int, default=24, help="horizontal bands (default 24: about ±2%% of frame height)")
    p.add_argument("--exclude", action="append", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"),
                   help="leave out a region (full-frame pixels): HUD, dialogue box, bright windows; repeatable")
    p.add_argument("--nocrop", action="store_true")
    p = add("pixel", cmd_pixel, "pixel-art scale")
    p.add_argument("--work", "-w", required=True)
    p.add_argument("ref")
    p.add_argument("box_pos", nargs="*", help="x0 y0 x1 y1 in full-frame pixels (or use --box)")
    p.add_argument("--box", nargs=4, type=int)
    p.add_argument("--name")
    p = add("pick", cmd_pick, "median colour of named patches")
    p.add_argument("--work", "-w", required=True)
    p.add_argument("ref", help="frame: C_140.0, C@2:20, or image code A")
    p.add_argument("patches", nargs="+", help="name=x0,y0,x1,y1, name=x,y (5x5 px around the point) or name=x,y,r (2r+1 px square; r=0: one pixel), full-frame pixels read off a zoom grid")
    p.add_argument("--main", action="store_true", help="for a patch of two colours, take the colour of the bigger cluster instead of the median")
    p.add_argument("--noshow", action="store_true", help="do not draw the boxes (zoom/pick_FRAME.png)")
    p = add("drop", cmd_drop, "remove entries from measure.json")
    p.add_argument("--work", "-w", required=True)
    p.add_argument("kind", help="frame, profile, pick, light, post or pixel")
    p.add_argument("names", nargs="+", help="entry names as `facts` lists them, e.g. G:sign_tea")
    p = add("light", cmd_light, "falloff of a light pool / glow")
    p.add_argument("--work", "-w", required=True)
    p.add_argument("ref")
    p.add_argument("cx", type=int)
    p.add_argument("cy", type=int)
    p.add_argument("--r", type=int, default=0, help="max radius in px (default 30%% of frame height)")
    p.add_argument("--name")
    p = add("post", cmd_post, "chromatic aberration + grain")
    p.add_argument("--work", "-w", required=True)
    p.add_argument("refs", nargs="+")
    p.add_argument("--nocrop", action="store_true")
    p = add("audio", cmd_audio, "audio analysis")
    p.add_argument("--work", "-w", required=True)
    p.add_argument("--codes")
    p.add_argument("--thr", type=float, default=6.0, help="dB jump that starts a new segment")
    p.add_argument("--ranges", help="extra ranges to analyse, e.g. 20-66,1:56~2:37")
    p.add_argument("--spectrum", action="append", help="spectrogram image CODE@START+SECONDS[/FMAX], e.g. B@1:05+12/9000; repeatable. "
                   "Inputs already analysed are not analysed again")
    p.add_argument("--linear", action="store_true", help="linear frequency axis for --spectrum (default: log, 40 Hz up)")
    p = add("facts", cmd_facts, "every measured number on one screen (+ colour check against content.json)")
    p.add_argument("--work", "-w", required=True)
    p.add_argument("content", nargs="?")
    p = add("crops", cmd_crops, "plates + specimens")
    p.add_argument("--work", "-w", required=True)
    p.add_argument("spec")
    p = add("build", cmd_build, "content.json -> HTML")
    p.add_argument("content")
    p.add_argument("--work", "-w", required=True)
    p.add_argument("--out", "-o", required=True)
    p.add_argument("--embed-fonts", action="store_true", help="inline Google-Fonts subsets so the HTML works offline")
    p.add_argument("--fragment", action="store_true", help="no <html>/<body> wrapper (for hosts that add their own)")
    p = add("pdf", cmd_pdf, "HTML -> PDF")
    p.add_argument("html")
    p.add_argument("out")
    p.add_argument("--page", default="1120x1584", help="page size in CSS px (default A4 ratio at desktop layout width)")
    p.add_argument("--keep-fonts", action="store_true", help="keep variable fonts as they are (bigger PDF)")
    p.add_argument("--keep-blank", action="store_true", help="keep pages that show nothing (by default they are removed)")
    p.add_argument("--nofontcheck", action="store_true", help="skip the list of elements drawn with system fonts")
    p = add("shots", cmd_shots, "QA screenshots")
    p.add_argument("html")
    p.add_argument("--pdf")
    p.add_argument("--out")
    p.add_argument("--widths", default="1280,390")
    p.add_argument("--only", help="section ids to shoot (e.g. l3,l7) instead of whole pages, after a fix")
    p.add_argument("--dpi", type=int, default=24)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
