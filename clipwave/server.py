#!/usr/bin/env python3
"""Clipwave: upload a long video, get ranked 9:16 clips.

Needs ffmpeg + numpy. Optional: opencv-python-headless (face-tracking crop),
faster-whisper (burned-in captions). Run: python3 server.py  ->  http://localhost:8000
"""
import json, os, re, shutil, subprocess, threading, time, traceback, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

try: import cv2
except ImportError: cv2 = None
try: from faster_whisper import WhisperModel
except ImportError: WhisperModel = None

ROOT = Path(__file__).parent
JOBS = ROOT / 'jobs'
CLIP_LEN = int(os.environ.get('CLIP_SECONDS', 30))
MAX_CLIPS = int(os.environ.get('MAX_CLIPS', 6))
MAX_MB = int(os.environ.get('MAX_UPLOAD_MB', 4096))
KEEP_SECONDS = 24 * 3600
STYLE = "Alignment=2,FontSize=11,Bold=1,Outline=2,Shadow=0,MarginV=45,PrimaryColour=&H00FFFFFF&,OutlineColour=&H00000000&"

state = {}                      # job id -> status dict (in memory; lost on restart)
GATE = threading.Semaphore(1)   # one ffmpeg job at a time


def run(*cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def probe(p):
    o = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height:format=duration', '-of', 'json', str(p)])
    j = json.loads(o)
    s = j['streams'][0]
    return float(j['format']['duration']), int(s['width']), int(s['height'])


def loudness(p):
    """Loudness in dB for every second of audio (empty array if there is no audio)."""
    pr = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', str(p), '-vn', '-ac', '1', '-ar', '8000',
                           '-f', 's16le', '-'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out, rest = [], b''
    while True:
        b = pr.stdout.read(16000 * 60)
        if not b: break
        b = rest + b
        n = len(b) // 16000
        a = np.frombuffer(b[:n * 16000], dtype='<i2').astype(np.float32).reshape(n, 8000)
        out.append(np.sqrt((a ** 2).mean(1)))
        rest = b[n * 16000:]
    pr.wait()
    if len(rest) >= 2000:
        a = np.frombuffer(rest[:len(rest) // 2 * 2], dtype='<i2').astype(np.float32)
        out.append(np.array([np.sqrt((a ** 2).mean())]))
    if not out: return np.zeros(0)
    return 20 * np.log10(np.maximum(np.concatenate(out), 1) / 32768)


def pick(db, length, k):
    """Best non-overlapping windows: loud on average and lively (high spread). -> [(start, len, score)]"""
    n = len(db)
    L = int(min(length, n))
    if L < 1: return []
    c, c2 = np.concatenate([[0], np.cumsum(db)]), np.concatenate([[0], np.cumsum(db ** 2)])
    m = (c[L:] - c[:-L]) / L
    s = m + 0.5 * np.sqrt(np.maximum((c2[L:] - c2[:-L]) / L - m ** 2, 0))
    left, out = s.copy(), []
    for _ in range(min(k, max(1, n // L))):
        i = int(np.argmax(left))
        if left[i] == -np.inf: break
        out.append((i, L, min(98, 60 + round(38 * float(np.mean(s < s[i]))))))   # score = rank among all windows
        left[max(0, i - L + 1):i + L] = -np.inf
    return out


def face_track(src, start, length, vw, vh):
    """[(seconds into clip, face centre x in px)], twice a second. Empty if OpenCV or its face model is missing."""
    if cv2 is None: return []
    xml = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
    if not xml.exists(): return []
    det, cap, pts = cv2.CascadeClassifier(str(xml)), cv2.VideoCapture(str(src)), []
    for k in range(int(length * 2)):
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + k / 2) * 1000)
        ok, f = cap.read()
        if not ok: break
        g = cv2.cvtColor(cv2.resize(f, (640, int(640 * vh / vw))), cv2.COLOR_BGR2GRAY)
        r = det.detectMultiScale(g, 1.1, 5, minSize=(30, 30))
        if len(r):
            x, y, w, h = max(r, key=lambda b: b[2] * b[3])   # biggest face = main speaker
            pts.append((k / 2, (x + w / 2) / 640 * vw))
    cap.release()
    return pts


def crop_x(pts, length, vw, cw):
    """ffmpeg x-expression that follows the face: smoothed, linear between 1 s keyframes."""
    if not pts: return str((vw - cw) // 2)
    ts = np.arange(int(length) + 1)
    xs = np.interp(ts, [t for t, _ in pts], [x for _, x in pts])
    xs = np.convolve(np.pad(xs, 2, mode='edge'), np.ones(5) / 5, mode='valid')
    xs = np.clip(xs - cw / 2, 0, vw - cw)
    terms = [f'gte(t,{a})*lt(t,{a + 1})*({xs[a]:.1f}+({xs[a + 1] - xs[a]:.1f})*(t-{a}))' for a in range(len(ts) - 1)]
    terms.append(f'gte(t,{ts[-1]})*{xs[-1]:.1f}')
    return '+'.join(terms)


def stamp(t):
    ms = int(round(t * 1000))
    return f'{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}'


def words_to_srt(words, n=3):
    """[(start, end, word)] -> SRT with n words per caption."""
    rows = []
    for i in range(0, len(words), n):
        g = words[i:i + n]
        rows.append(f'{i // n + 1}\n{stamp(g[0][0])} --> {stamp(g[-1][1])}\n{" ".join(w[2] for w in g)}\n')
    return '\n'.join(rows)


_model = None
def captions(src, start, length, d, n):
    """Transcribe the clip with faster-whisper and write captionN.srt. Returns the file name or None."""
    global _model
    _model = _model or WhisperModel(os.environ.get('WHISPER_MODEL', 'base'), compute_type='int8')
    wav = d / f'a{n}.wav'
    run('ffmpeg', '-y', '-v', 'error', '-ss', str(start), '-t', str(length), '-i', src.name,
        '-vn', '-ac', '1', '-ar', '16000', wav.name, cwd=d)
    segs, _ = _model.transcribe(str(wav), word_timestamps=True, vad_filter=True)
    words = [(w.start, w.end, w.word.strip()) for s in segs for w in s.words if w.word.strip()]
    wav.unlink()
    if not words: return None
    (d / f'caption{n}.srt').write_text(words_to_srt(words))
    return f'caption{n}.srt'


def render(d, src, start, length, vw, vh, pts, srt, out):
    """Cut [start, start+length] to 1080x1920 H.264. Landscape: face-following crop. Portrait source: centre crop."""
    if vw / vh <= 9 / 16:
        vf = 'crop=iw:trunc(iw*16/9/2)*2'
    else:
        cw = int(vh * 9 / 16) // 2 * 2
        vf = f"crop=w={cw}:h={vh // 2 * 2}:x='{crop_x(pts, length, vw, cw)}':y=0"
    vf += ',scale=1080:1920:flags=lanczos,setsar=1'
    if srt: vf += f",subtitles={srt}:force_style='{STYLE}'"
    (d / 'vf.txt').write_text(vf)   # filter script file: no shell/filtergraph escaping needed
    run('ffmpeg', '-y', '-v', 'error', '-ss', str(start), '-t', str(length), '-i', src,
        '-filter_script:v', 'vf.txt', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '21', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', out, cwd=d)


def work(jid):
    j, d = state[jid], JOBS / jid
    src = next(d.glob('src.*'))

    def upd(p, s): j['progress'], j['stage'] = round(p), s
    try:
        upd(0, 'Waiting in queue')
        with GATE:
            upd(3, 'Reading video')
            dur, vw, vh = probe(src)
            upd(8, 'Listening for energetic moments')
            db = loudness(src)
            if not len(db): db = np.zeros(max(1, int(dur)))   # no audio: evenly spaced clips
            picks = pick(db, CLIP_LEN, MAX_CLIPS)
            N = len(picks)
            for n, (st, ln, sc) in enumerate(picks, 1):
                base, step = 12 + (n - 1) * 88 / N, 88 / N
                upd(base, f'Tracking faces, clip {n} of {N}')
                pts = face_track(src, st, ln, vw, vh)
                srt = None
                if WhisperModel:
                    upd(base + step * .3, f'Writing captions, clip {n} of {N}')
                    srt = captions(src, st, ln, d, n)
                upd(base + step * .5, f'Rendering clip {n} of {N}')
                render(d, src.name, st, ln, vw, vh, pts, srt, f'clip{n}.mp4')
                run('ffmpeg', '-y', '-v', 'error', '-ss', str(ln / 2), '-i', f'clip{n}.mp4',
                    '-frames:v', '1', '-vf', 'scale=360:-2', f'clip{n}.jpg', cwd=d)
                j['clips'].append({'n': n, 'score': sc, 'start': st, 'end': st + ln, 'video': f'clip{n}.mp4',
                                   'thumb': f'clip{n}.jpg', 'captions': bool(srt), 'tracked': bool(pts)})
            j['status'] = 'done'
            upd(100, 'Done')
    except subprocess.CalledProcessError as e:
        traceback.print_exc()
        j['status'], j['error'] = 'error', 'Could not read this file as a video.'
        print(e.stderr.decode(errors='replace')[-600:] if e.stderr else '')
    except Exception as e:
        traceback.print_exc()
        j['status'], j['error'] = 'error', f'Processing failed: {e}'


def sweep():
    for p in JOBS.iterdir():
        if time.time() - p.stat().st_mtime > KEEP_SECONDS:
            shutil.rmtree(p, ignore_errors=True)
            state.pop(p.name, None)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def file(self, path, ctype, name=None):
        size = path.stat().st_size
        a, b, code = 0, size - 1, 200
        m = re.match(r'bytes=(\d*)-(\d*)$', self.headers.get('Range', ''))
        if m and (m[1] or m[2]):
            if m[1]: a, b = int(m[1]), (int(m[2]) if m[2] else size - 1)
            else: a = max(0, size - int(m[2]))
            b = min(b, size - 1)
            if a > b: return self.json({'error': 'bad range'}, 416)
            code = 206
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(b - a + 1))
        if code == 206: self.send_header('Content-Range', f'bytes {a}-{b}/{size}')
        if name: self.send_header('Content-Disposition', f'attachment; filename="{name}"')
        self.end_headers()
        try:
            with open(path, 'rb') as f:
                f.seek(a)
                left = b - a + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk: break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path, _, query = self.path.partition('?')
        if path == '/':
            return self.file(ROOT / 'static' / 'index.html', 'text/html; charset=utf-8')
        m = re.fullmatch(r'/api/jobs/([0-9a-f]{12})', path)
        if m:
            j = state.get(m[1])
            return self.json(j, 200) if j else self.json({'error': 'Unknown or expired job'}, 404)
        m = re.fullmatch(r'/clips/([0-9a-f]{12})/(clip(\d+)\.(mp4|jpg))', path)
        if m and (JOBS / m[1] / m[2]).exists():
            p = JOBS / m[1] / m[2]
            if m[4] == 'jpg': return self.file(p, 'image/jpeg')
            dl = re.search(r'dl=(tiktok|reels|shorts)', query)
            return self.file(p, 'video/mp4', f'clipwave-{m[1][:6]}-{m[3]}-{dl[1]}.mp4' if dl else None)
        self.json({'error': 'not found'}, 404)

    def do_POST(self):
        if self.path != '/api/jobs': return self.json({'error': 'not found'}, 404)
        n = int(self.headers.get('Content-Length') or 0)
        if not 0 < n <= MAX_MB << 20:
            return self.json({'error': f'Upload must be under {MAX_MB} MB'}, 413)
        ext = re.sub(r'[^a-z0-9]', '', os.path.splitext(self.headers.get('X-Filename', ''))[1].lower())[:5] or 'mp4'
        sweep()
        jid = uuid.uuid4().hex[:12]
        d = JOBS / jid
        d.mkdir(parents=True)
        left = n
        with open(d / f'src.{ext}', 'wb') as f:
            while left > 0:
                b = self.rfile.read(min(1 << 20, left))
                if not b: break
                f.write(b)
                left -= len(b)
        if left:
            shutil.rmtree(d, ignore_errors=True)
            return self.json({'error': 'Upload was interrupted'}, 400)
        state[jid] = {'status': 'running', 'stage': 'Queued', 'progress': 0, 'clips': [], 'error': None}
        threading.Thread(target=work, args=(jid,), daemon=True).start()
        self.json({'id': jid})


if __name__ == '__main__':
    JOBS.mkdir(exist_ok=True)
    if not shutil.which('ffmpeg'): raise SystemExit('ffmpeg not found. Install it first (https://ffmpeg.org/download.html).')
    host, port = os.environ.get('HOST', '127.0.0.1'), int(os.environ.get('PORT', 8000))
    print(f'Clipwave on http://{host}:{port}  | face tracking: {"on" if cv2 else "off (pip install opencv-python-headless)"}'
          f'  | captions: {"on" if WhisperModel else "off (pip install faster-whisper)"}')
    ThreadingHTTPServer((host, port), H).serve_forever()
