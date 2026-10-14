#!/usr/bin/env python3
"""Self-check: python3 test_pipeline.py  (needs ffmpeg; makes a 90 s test video, loud only from 40-70 s)."""
import json, os, subprocess, tempfile, threading, time, urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

os.environ['CLIP_SECONDS'] = '20'
import server as S

tmp = Path(tempfile.mkdtemp())
vid = tmp / 'long.mp4'
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=1280x720:rate=24:duration=90',
                '-f', 'lavfi', '-i', 'sine=frequency=440:duration=90',
                '-af', "volume='if(between(t,40,70),1,0.03)':eval=frame", '-c:v', 'libx264', '-c:a', 'aac', '-shortest', str(vid)], check=True)

# 1. highlight picker lands inside the loud part, scores are ranked
p = S.pick(S.loudness(vid), S.CLIP_LEN, 6)
assert 38 <= p[0][0] <= 52, p
assert [c[2] for c in p] == sorted((c[2] for c in p), reverse=True), p

# 2. captions text -> SRT
srt = S.words_to_srt([(0, .4, 'hello'), (.4, .8, 'there'), (.8, 1.2, 'big'), (1.3, 1.9, 'world')])
assert '00:00:00,000 --> 00:00:01,200\nhello there big' in srt and srt.count('-->') == 2, srt

# 3. moving crop + burned captions render to 1080x1920 (fake face path: left -> right)
(tmp / 'c.srt').write_text(srt)
shutil_src = tmp / 'src.mp4'; shutil_src.write_bytes(vid.read_bytes())
S.render(tmp, 'src.mp4', 40, 6, 1280, 720, [(0, 200), (3, 640), (6, 1100)], 'c.srt', 'out.mp4')
w, h = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                                '-of', 'csv=p=0', str(tmp / 'out.mp4')]).decode().strip().split(',')
assert (w, h) == ('1080', '1920'), (w, h)

# 4. end to end over HTTP: upload, poll, fetch thumb, ranged mp4, download header
S.JOBS.mkdir(exist_ok=True)
srv = ThreadingHTTPServer(('127.0.0.1', 0), S.H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f'http://127.0.0.1:{srv.server_port}'
req = urllib.request.Request(base + '/api/jobs', data=vid.read_bytes(), headers={'X-Filename': 'long.mp4'}, method='POST')
jid = json.load(urllib.request.urlopen(req))['id']
for _ in range(180):
    j = json.load(urllib.request.urlopen(f'{base}/api/jobs/{jid}'))
    if j['status'] != 'running': break
    time.sleep(1)
assert j['status'] == 'done' and j['clips'], j
c = j['clips'][0]
assert urllib.request.urlopen(f"{base}/clips/{jid}/{c['thumb']}").read(2) == b'\xff\xd8'
r = urllib.request.urlopen(urllib.request.Request(f"{base}/clips/{jid}/{c['video']}?dl=tiktok", headers={'Range': 'bytes=0-99'}))
assert r.status == 206 and len(r.read()) == 100 and 'tiktok' in r.headers['Content-Disposition']
bad = urllib.request.Request(base + '/api/jobs', data=b'not a video', headers={'X-Filename': 'x.mp4'}, method='POST')
bj = json.load(urllib.request.urlopen(bad))['id']
for _ in range(30):
    b = json.load(urllib.request.urlopen(f'{base}/api/jobs/{bj}'))
    if b['status'] != 'running': break
    time.sleep(1)
assert b['status'] == 'error' and 'video' in b['error'], b
print('ALL OK', [(c['n'], c['score'], c['start']) for c in j['clips']], 'tracked:', c['tracked'])
