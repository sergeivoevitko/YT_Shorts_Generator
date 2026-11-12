# ClipForge

Two tools in one app:

- **YouTube to shorts**: paste a YouTube link, get a batch of ready-to-post short videos.
- **My footage + story**: upload your own clips (drone shots, landscapes), get a narrated video with a
  voiceover, captions and music.

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
4. Press **Generate shorts**. The app proposes more moments than you asked for and shows them in a list:
   tick the ones you want, and choose a title and an on-screen hook for each (the AI offers three of each).
   Only what you tick is rendered. Untick "Let me choose which moments to render" under Analysis to skip
   this step and render automatically.
5. Each finished clip shows a score, a title, a post caption with hashtags, and a download button.

**Edit** on any clip opens the clip editor: change the hook, title and caption, move the start or end of
the clip by clicking a word, fix a misheard word, or hide a word from the captions. Only that clip is
rendered again.

**Re-run with current settings** renders the same video again with new options, without downloading or
transcribing a second time. Useful for trying another caption style or aspect ratio.

Files are saved in `workspace/jobs/<run>/`, together with `post_text.txt` (titles and captions for posting).

## My footage + story

Switch to **My footage + story** at the top left.

1. Choose several of your video clips at once.
2. Pick a theme and tone and let the app supply the story, or paste your own text.
3. Choose the narrator voice, length (20 to 60 seconds), and how many videos to make.
4. Press **Generate narrated video**.

The app looks through each clip for its sharpest, steadiest stretches, records the voiceover with a voice
model that runs on your computer, changes shot on sentence boundaries, and adds captions timed to the voice.
Upload a music file under Edits and it is lowered automatically whenever the narrator speaks; without one,
a soft ambient sound can be added instead.

The first run downloads the voice model (about 350 MB, one time). No account or key is needed for the voice.
Stories are written by the AI chosen under AI settings; without one, a built-in story for the chosen theme is used.

## Results tracker

After you post a clip, press **Track** on it and paste the link of the posted video. The app reads its public
view count (YouTube and usually TikTok; for Instagram, type the number in by hand) and the **Results** window
compares your clips by hook type, caption style, length, cold open and platform. Press **Update view counts**
now and then. Once six or more clips have view counts, clear patterns are passed to the AI when it picks
moments for your next video. With few clips the averages can mislead, so treat them as hints.

## Compilation

Switch to **Compilation** at the top left to turn several videos into one "best moments" video.

1. Paste video links (one per line) and/or choose video files, up to 12 in total.
2. Pick a theme (best, funniest, most surprising, most useful, most emotional, or describe your own), a style
   (numbered countdown or plain reel) and a target length from 30 seconds to 5 minutes.
3. Press **Find moments for the compilation**. The app transcribes every video, marks loud reactions
   (laughter, cheering) as hints, and asks the AI for moments that fit the theme.
4. In the list, tick the moments you want, move them up or down to set the order, edit the title, and press
   **Build compilation**.

The result opens with a short teaser of the best moment under the title, then plays the moments in order with
a number and label on each. `post_text.txt` includes chapter timestamps to paste into the description. Choose
16:9 under Output for a normal YouTube video or 9:16 for a vertical one.

Use videos you own or have permission to reuse. Compilations of other people's videos are a common target for
copyright claims and for YouTube's reused-content rule.

## Turning on the AI

Without an AI the app picks moments with simple offline scoring and uses built-in stories. With one, the AI
reads the whole transcript, chooses the moments, and writes the hooks, titles, captions and stories, which is
a large quality step up. Press **AI settings** and choose a provider:

| Provider | Cost | Notes |
| --- | --- | --- |
| Google Gemini, Groq, Cerebras, OpenRouter, Mistral, SambaNova, NVIDIA NIM, Z.ai, GitHub Models, Cohere, Hugging Face | Free within limits | Create a free account, press "Get a key here" in the settings window, paste the key |
| Pollinations | Free, no account or key | Quality and speed vary |
| Ollama, LM Studio | Free, no limits | Runs on your own PC. Slow without a good graphics card |
| Claude | Paid by usage (a few cents per video) | Best quality. Key from https://console.anthropic.com |
| Other | Depends | Any service with an OpenAI-compatible API |

**Find a working model** tries the provider's models one by one and selects the first that answers, so you do
not need to know model names. **Ollama is detected automatically**: the settings window shows its version and
your installed models and picks the one best suited to this job (general chat models of about 7 to 14 billion
parameters; coding and embedding models are skipped). If Ollama is installed but not running, the app can start
it. On a PC where no AI was ever chosen, a running Ollama with a suitable model is used without any setup.

Use **Show available models** to see what the provider currently offers (model names change over time), and
**Save and test** to check that the key works. Keys are stored only in `workspace/settings.json` on your computer.

Free tiers accept less text per request, so long videos are analyzed in several parts. With the cloud providers
your transcripts are sent to that provider; with Ollama or LM Studio nothing leaves your PC.

## What the edits do

| Setting | Effect |
| --- | --- |
| Follow faces | Detects the speaker and keeps them in frame, cutting or gliding when they move |
| Fit + blur | Shows the whole frame over a blurred background (for gameplay, screen recordings) |
| Cut out pauses | Removes silences longer than the threshold, which tightens the pacing |
| Punch-in zooms | Alternates a slight zoom on new sentences to keep the picture moving |
| Hook | Short text at the top for the first seconds, written to stop the scroll |
| Cold open | Plays the clip's strongest line for 2 to 3 seconds first, then starts from the top. Can be switched off per clip in the review list |
| Topic label | A small label at the top for the whole clip, so someone who lands mid-clip knows what it is about |
| A/B versions | Renders each chosen moment twice, with a different hook, caption style and opening, so you can post both and compare |
| Keep text clear of buttons | Moves captions, hook and label out of the areas TikTok, Reels or Shorts cover with their own buttons |
| Cut filler words | Removes "um", "uh", "er" from both the audio and the captions |
| Emojis on key words | Pops an emoji above the captions when a matching word is spoken |
| Sound effects | Whoosh on zooms, pop on the hook and emojis |
| Brand | Your logo in a corner, your highlight color and font, your handle at the bottom |
| Progress bar | Thin bar along the bottom that shows how much of the clip is left |
| Even out loudness | Normalizes volume to the level short-form platforms expect |
| Background music | Mixes in an audio file you provide, at the volume you set |

## Graphics card (GPU)

The app uses your graphics card automatically when it can, and falls back to the processor when it cannot:

- **Rendering**: NVIDIA, Intel Quick Sync and AMD video encoders are tried at start-up and the first one that
  works is used. Choose "Processor only" under Output to turn this off.
- **Speech recognition**: runs on NVIDIA cards. `install.bat` installs the needed libraries when it finds one.

The header shows "GPU: ..." when a card is in use, and the Log says which device each step ran on.
Cropping, captions and other effects still run on the processor, so the speed-up is real but partial.

## Troubleshooting

- **Speech recognition fails on one PC but works on another** ("stopped unexpectedly", exit code 3221225477):
  the usual cause is an outdated Microsoft Visual C++ runtime on that PC. Run `install.bat` again, which
  updates it, then restart the app. Until then, under Analysis set **Speech recognition** to **Groq cloud**:
  it is free, needs only a Groq key saved in AI settings, and is much faster than a laptop processor. If a
  Groq key is saved, the app also switches to it by itself when local speech recognition fails.
- **Something went wrong and you want to see why**: press **Log** in the top right. It shows everything the
  app did, including every AI request and the exact error text. Press Copy to share it. Keys are never logged.
  The same text is in `workspace/clipforge.log`.
- **A run says "picked by offline scoring" although an AI is set up**: a yellow box above the clips gives the
  reason. Open AI settings and press **Save and test**. With Gemini, `gemini-3.5-flash` is a model that works
  on the free tier; the newest models often have no free quota or do not answer.
- **Download fails**: run `update.bat` and try again. If YouTube asks for a sign-in, download the video
  yourself and use the upload option.
- **"YouTube: install Deno" in the header**: run `install.bat` again, then reopen `start.bat`.
- **"No caption support in FFmpeg"**: install the full build with `winget install Gyan.FFmpeg`, then restart.
- **Slow transcription**: pick a smaller speech model under Analysis. With an NVIDIA card, run `install.bat`
  again so the GPU libraries are installed.
- **The window closes or drops to "Press any key" during a run**: update to the latest version. Speech
  recognition now runs in a separate helper, so a problem there is reported in the Log instead of closing the app.
- **Port already in use**: set the `CLIPFORGE_PORT` environment variable to another number before starting.

## Notes

- Only English speech is supported.
- Use it on videos you own or have permission to reuse.
- The score is an estimate of short-form potential, not a prediction of views.
  With Claude, each clip also shows separate hook, flow, value and trend scores.
- Bundled fonts and the face detection model are open source; see `assets/THIRD_PARTY.md`.
- The narrator voice is Kokoro (Apache 2.0), downloaded on first use into `workspace/models/`.
- Footage shot in a flat or log color profile is not color graded by the app; grade it first if needed.
