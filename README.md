# ClipForge

Paste a YouTube link, get a batch of ready-to-post short videos.

ClipForge downloads the video, transcribes it, finds the strongest moments, then cuts each one into a
vertical short with a hook, face-following framing, jump cuts, punch-in zooms and animated captions.
Everything runs on your own computer and is controlled from a page in your browser.

## Install (Windows)

1. Download this repository (green **Code** button, then **Download ZIP**) and unzip it, or `git clone` it.
2. Double-click **`install.bat`**. It sets up Python packages and installs FFmpeg and Deno if they are missing.
3. Double-click **`start.bat`**. The app opens in your browser at http://127.0.0.1:8765.

Keep the black window open while you use the app. Close it to stop.

Run **`update.bat`** now and then. YouTube changes often, and the updater pulls the latest downloader.

macOS and Linux: run `./install.sh`, then `./start.sh` (install FFmpeg and Deno yourself first).

## Using it

1. Paste a YouTube link, or upload a video file.
2. Choose the output: aspect ratio (9:16, 4:5, 1:1, 16:9), resolution (720p to 4K), frame rate, number of
   clips and clip length.
3. Choose the look: framing mode, caption style (or a different style per clip), hook style, and which
   edits to apply.
4. Press **Generate shorts**. Each finished clip shows a score, a title, a post caption with hashtags, and
   a download button.

**Re-run with current settings** renders the same video again with new options, without downloading or
transcribing a second time. Useful for trying another caption style or aspect ratio.

Files are saved in `workspace/jobs/<run>/`, together with `post_text.txt` (titles and captions for posting).

## Turning on Claude (recommended)

Without an API key the app picks moments with simple offline scoring. With one, Claude reads the whole
transcript, chooses the moments, and writes the hooks, titles and captions, which is a large quality step up.

1. Create a key at https://console.anthropic.com.
2. In the app, press **AI settings**, paste the key, and save. It is stored only in `workspace/settings.json`
   on your computer.

Cost is typically a few cents per video and depends on video length and the model you choose.

## What the edits do

| Setting | Effect |
| --- | --- |
| Follow faces | Detects the speaker and keeps them in frame, cutting or gliding when they move |
| Fit + blur | Shows the whole frame over a blurred background (for gameplay, screen recordings) |
| Cut out pauses | Removes silences longer than the threshold, which tightens the pacing |
| Punch-in zooms | Alternates a slight zoom on new sentences to keep the picture moving |
| Hook | Short text at the top for the first seconds, written to stop the scroll |
| Progress bar | Thin bar along the bottom that shows how much of the clip is left |
| Even out loudness | Normalizes volume to the level short-form platforms expect |
| Background music | Mixes in an audio file you provide, at the volume you set |

## Troubleshooting

- **Download fails**: run `update.bat` and try again. If YouTube asks for a sign-in, download the video
  yourself and use the upload option.
- **"YouTube: install Deno" in the header**: run `install.bat` again, then reopen `start.bat`.
- **"No caption support in FFmpeg"**: install the full build with `winget install Gyan.FFmpeg`, then restart.
- **Slow transcription**: pick a smaller speech model under Analysis. An NVIDIA GPU is used automatically
  when the CUDA 12 and cuDNN 9 libraries are installed; otherwise the app runs on the CPU.
- **Port already in use**: set the `CLIPFORGE_PORT` environment variable to another number before starting.

## Notes

- Only English speech is supported.
- Use it on videos you own or have permission to reuse.
- The score is an estimate of short-form potential, not a prediction of views.
- Bundled fonts and the face detection model are open source; see `assets/THIRD_PARTY.md`.
