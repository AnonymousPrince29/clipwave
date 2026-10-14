# Clipwave

Upload a long video, get ranked 9:16 clips you can download. No watermarks.

## Run it

1. Install [ffmpeg](https://ffmpeg.org/download.html) and Python 3.9+.
2. `pip install numpy opencv-python-headless`
3. `python3 server.py`, then open http://localhost:8000

Optional captions: `pip install faster-whisper`. Restart the server and clips get burned-in captions (first run downloads a speech model, set `WHISPER_MODEL=small` for better accuracy).

Check everything works: `python3 test_pipeline.py`

## What each feature does

- **Highlight detection:** loudness per second from the audio track. The best non-overlapping 30 s windows (loud and lively) become clips. The viral score is the window's rank among all windows in your video (60 to 98). It is a heuristic, not a prediction.
- **Face-tracking crop:** OpenCV finds the largest face twice a second and the 9:16 crop follows it smoothly. No face found, or no OpenCV: centre crop.
- **Auto-captioning:** faster-whisper, 3 words at a time, burned into the video. Off unless installed.
- **Export buttons:** TikTok, Reels and Shorts all download the same 1080x1920 H.264 file, which all three accept. Only the file name differs.

## Settings (environment variables)

`CLIP_SECONDS` (30), `MAX_CLIPS` (6), `MAX_UPLOAD_MB` (4096), `HOST` (127.0.0.1), `PORT` (8000), `WHISPER_MODEL` (base).

## Hosting for other people

It binds to localhost by default. To put it online, run it on a VPS with `HOST=0.0.0.0` behind HTTPS (Caddy or nginx), and add login or rate limiting first. Jobs run one at a time and files are deleted after 24 hours.
