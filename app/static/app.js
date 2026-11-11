"use strict";
const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, kids = []) => {
  const node = Object.assign(document.createElement(tag), props);
  for (const kid of [].concat(kids)) node.append(kid);
  return node;
};
const api = async (path, opts) => {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
};

let cfg = null;
let settings = {};
let currentJob = null;
let pollTimer = null;

const SHAPES = { "9:16": [9, 16], "4:5": [12, 15], "1:1": [13, 13], "16:9": [18, 10] };
const ASPECT_NOTE = { "9:16": "TikTok, Reels, Shorts", "4:5": "Instagram feed", "1:1": "Square feed posts", "16:9": "Landscape" };
const REFRAME = {
  smart: ["Follow faces", "Crops to the speaker and follows them when they move or the shot changes."],
  center: ["Center crop", "Crops the middle of the frame. Best when the subject never leaves the center."],
  blur: ["Fit + blur", "Keeps the whole frame visible over a blurred copy of itself. Best for screen recordings and gameplay."],
};
const STAGES = [["download", "Download"], ["transcribe", "Transcribe"], ["analyze", "Find moments"], ["render", "Edit + render"]];
const STORY_STAGES = [["footage", "Review footage"], ["story", "Write story"], ["voice", "Voiceover"], ["render", "Edit + render"]];
const MODES = [["clips", "YouTube to shorts"], ["story", "My footage + story"]];
const RES_SHORT = { "720p": 720, "1080p": 1080, "1440p": 1440, "4k": 2160 };

function save() { localStorage.setItem("clipforge.settings", JSON.stringify(settings)); }

function segmented(id, options, key, onChange) {
  const box = $(id);
  box.replaceChildren();
  for (const [value, label, shape] of options) {
    const btn = el("button", { type: "button", className: settings[key] === value ? "on" : "" });
    if (shape) btn.append(el("span", { className: "shape", style: `width:${shape[0]}px;height:${shape[1]}px` }));
    btn.append(label);
    btn.onclick = () => { settings[key] = value; save(); segmented(id, options, key, onChange); onChange && onChange(); };
    box.append(btn);
  }
}

function bind(id, key, kind, labelId, fmt) {
  const input = $(id);
  if (kind === "check") input.checked = !!settings[key]; else input.value = settings[key];
  const show = () => { if (labelId) $(labelId).textContent = fmt ? fmt(settings[key]) : settings[key]; };
  show();
  input.addEventListener(kind === "check" || input.tagName === "SELECT" ? "change" : "input", () => {
    settings[key] = kind === "check" ? input.checked : kind === "num" ? Number(input.value) : input.value;
    save(); show(); refreshHints();
  });
}

function captionStyles() {
  const box = $("caption_style");
  box.replaceChildren();
  const entries = [["mix", { name: "Mix it up", desc: "A different style on every clip", mix: true }],
    ...Object.entries(cfg.caption_styles), ["none", { name: "No captions", desc: "Leave the video clean", none: true }]];
  for (const [key, st] of entries) {
    const demo = el("div", { className: "demo" });
    if (st.mix) {
      demo.append(el("span", { textContent: "MIX ", style: "color:#ffe100" }), el("span", { textContent: "OF ", style: "color:#39ff14" }),
        el("span", { textContent: "STYLES", style: "color:#00f0ff" }));
      demo.style.textShadow = "0 2px 0 #000";
    } else if (st.none) {
      demo.textContent = "—"; demo.style.color = "#8d96a8";
    } else {
      const words = st.upper ? ["THIS", "IS", "IT"] : ["this", "is", "it"];
      words.forEach((w, i) => demo.append(el("span", { textContent: w + (i < 2 ? " " : ""), style: `color:${i === 1 ? st.active : st.primary}` })));
      const o = st.outline;
      demo.style.textShadow = `-1.5px -1.5px 0 ${o}, 1.5px -1.5px 0 ${o}, -1.5px 1.5px 0 ${o}, 1.5px 1.5px 0 ${o}, 0 3px 6px rgba(0,0,0,.5)`;
      demo.style.fontFamily = `"${st.font}", Impact, "Arial Black", sans-serif`;
    }
    const btn = el("button", { type: "button", className: "style" + (settings.caption_style === key ? " on" : "") },
      [demo, el("div", { className: "name", textContent: st.name }), el("div", { className: "desc", textContent: st.desc })]);
    btn.onclick = () => { settings.caption_style = key; save(); captionStyles(); };
    box.append(btn);
  }
}

function refreshHints() {
  const [aw, ah] = settings.aspect.split(":").map(Number);
  const short = RES_SHORT[settings.resolution];
  const w = aw <= ah ? short : Math.round(short * aw / ah / 2) * 2;
  const h = aw <= ah ? Math.round(short * ah / aw / 2) * 2 : short;
  $("sizeHint").textContent = `${w} x ${h} pixels. ${ASPECT_NOTE[settings.aspect]}.` +
    (short > 1080 ? " Sizes above 1080p are upscaled unless the source video is that sharp, and render much slower." : "");
  const story = settings.mode === "story";
  const cloud = settings.speech_engine === "groq";
  $("whisperRow").hidden = cloud;
  $("speechHint").textContent = cloud
    ? (cfg.status.groq_key ? "The audio is sent to Groq for transcription. A 20 minute video takes well under a minute."
       : "No Groq key is saved yet. Open AI settings, choose Groq, paste a free key and press Save. You can switch the AI provider back afterwards.")
    : (cfg.status.msvc_runtime_old ? "Warning: the Microsoft Visual C++ runtime on this PC is old or missing, which can crash local speech recognition. Run install.bat again to update it, or use Groq cloud. "
       : "") + (cfg.status.groq_key ? "If it fails on this PC, the app switches to Groq cloud automatically." : "");
  const g = cfg.status.gpu;
  $("gpuHint").textContent = settings.encoder === "cpu" ? "Rendering uses the processor only."
    : [g.encoder ? `Rendering uses your ${g.encoder_name} graphics card.` : "No supported graphics card encoder was found, so rendering uses the processor.",
       g.speech ? "Speech recognition also runs on the graphics card." : ""].join(" ");
  document.body.dataset.mode = settings.mode;
  $("reframeHint").textContent = story
    ? (settings.reframe === "blur" ? REFRAME.blur[1] : "Crops the middle of each clip to fill the frame. Use Fit + blur to keep whole landscape shots visible.")
    : REFRAME[settings.reframe][1];
  $("useAiLabel").textContent = story ? "Let the AI write the story" : "Let the AI pick the moments and write the hooks";
  if (!$("go").disabled) $("go").textContent = goLabel();
  $("storyHint").textContent = settings.story_text.trim()
    ? `Your text: about ${Math.round(settings.story_text.trim().split(/\s+/).length / 2.5)} seconds when spoken. Length and theme are ignored.`
    : cfg.status.ai.ready && settings.use_ai ? `${cfg.status.ai.name} will write an original story for each video.`
    : "Without an AI set up, one of the built-in stories for this theme is used.";
  const keyed = cfg.status.ai.ready;
  const aiLabel = `${cfg.status.ai.name} (${cfg.status.ai.model})`;
  $("aiHint").textContent = story ? (keyed && settings.use_ai ? `Stories are written by ${aiLabel}. The speech model below is used to time the captions to the voice.`
      : "Built-in stories are used unless you paste your own text or set up an AI under AI settings.")
    : !settings.use_ai ? "Moments are picked by offline scoring (keywords, pacing, clean endings)."
    : keyed ? `Using ${aiLabel}.`
    : "No AI is set up yet, so offline scoring is used. Open AI settings to connect a free, local or paid AI for much better picks and hooks.";
}

function pills() {
  const s = cfg.status;
  const items = [
    [s.ffmpeg.found ? "FFmpeg ready" : "FFmpeg missing", s.ffmpeg.found ? "ok" : "bad"],
    [s.ffmpeg.libass ? "Captions ready" : "No caption support in FFmpeg", s.ffmpeg.libass ? "ok" : "bad"],
    [s.js_runtime.length ? "YouTube ready" : "YouTube: install Deno", s.js_runtime.length ? "ok" : "warn"],
    [s.gpu.encoder || s.gpu.speech ? `GPU: ${s.gpu.encoder_name || "NVIDIA"}` : "No GPU found (using CPU)", s.gpu.encoder || s.gpu.speech ? "ok" : ""],
    [s.ai.ready ? `AI: ${s.ai.name}` : "Offline mode (no AI set up)", s.ai.ready ? "ok" : "warn"],
  ];
  $("pills").replaceChildren(...items.map(([text, cls]) => el("span", { className: "pill " + cls, textContent: text })));
}

async function init() {
  cfg = await api("/api/config");
  let stored = {};
  try { stored = JSON.parse(localStorage.getItem("clipforge.settings") || "{}"); } catch (e) { /* ignore */ }
  if ((stored._v || 1) < 2) { delete stored.fps; stored._v = 2; }  // older versions forced 30 fps, which can stutter
  settings = Object.assign({ mode: "clips" }, cfg.defaults, stored);
  pills();
  segmented("mode", MODES, "mode", refreshHints);
  const options = (id, map) => $(id).replaceChildren(...Object.entries(map).map(([k, v]) => el("option", { value: k, textContent: v })));
  options("story_theme", cfg.story_themes); options("story_tone", cfg.story_tones); options("voice", cfg.voices);
  for (const id of ["story_theme", "story_tone", "story_text", "voice", "transition", "footage_order"]) bind(id, id);
  bind("story_seconds", "story_seconds", "num", "storySecVal", (v) => `about ${v} seconds`);
  bind("num_videos", "num_videos", "num", "numVideosVal");
  bind("voice_speed", "voice_speed", "num", "voiceSpeedVal", (v) => Math.round(v * 100) + "%");
  bind("ambient_pad", "ambient_pad", "check");
  $("footage").onchange = () => {
    const files = [...$("footage").files];
    const mb = files.reduce((a, f) => a + f.size, 0) / 1e6;
    $("footageName").textContent = files.length ? `${files.length} clip${files.length > 1 ? "s" : ""} selected (${mb.toFixed(0)} MB)` : "Select several files at once: drone shots, landscapes, b-roll";
  };

  segmented("aspect", cfg.aspects.map((a) => [a, a, SHAPES[a]]), "aspect", refreshHints);
  segmented("reframe", Object.entries(REFRAME).map(([k, v]) => [k, v[0]]), "reframe", refreshHints);
  captionStyles();
  $("hook_style").replaceChildren(...Object.entries(cfg.hook_styles).map(([k, v]) => el("option", { value: k, textContent: v })),
    el("option", { value: "none", textContent: "No hook text" }));
  $("whisper_model").replaceChildren(...Object.entries(cfg.whisper_models).map(([k, v]) => el("option", { value: k, textContent: v })));
  $("model").replaceChildren(...Object.entries(cfg.claude_models).map(([k, v]) => el("option", { value: k, textContent: v })));

  for (const id of ["speech_engine", "encoder", "resolution", "fps", "quality", "caption_position", "hook_style", "whisper_model", "instructions"]) bind(id, id);
  bind("num_clips", "num_clips", "num", "numClipsVal");
  bind("min_len", "min_len", "num"); bind("max_len", "max_len", "num"); bind("hook_seconds", "hook_seconds", "num");
  bind("caption_scale", "caption_scale", "num", "capScaleVal", (v) => Math.round(v * 100) + "%");
  bind("silence_gap", "silence_gap", "num", "gapVal", (v) => Number(v).toFixed(2) + " s");
  bind("zoom_amount", "zoom_amount", "num", "zoomVal", (v) => "+" + Math.round((v - 1) * 100) + "%");
  bind("music_volume", "music_volume", "num", "musicVal", (v) => Math.round(v * 100) + "%");
  for (const id of ["remove_silence", "zooms", "progress_bar", "normalize_audio", "use_ai", "remove_fillers", "emojis", "sfx", "logo_enabled"]) bind(id, id, "check");
  bind("sfx_volume", "sfx_volume", "num", "sfxVal", (v) => Math.round(v * 100) + "%");
  bind("logo_scale", "logo_scale", "num", "logoScaleVal", (v) => Math.round(v * 100) + "% of width");
  bind("logo_opacity", "logo_opacity", "num", "logoOpVal", (v) => Math.round(v * 100) + "%");
  $("brand_font").replaceChildren(el("option", { value: "", textContent: "Style default" }),
    ...cfg.fonts.map((f) => el("option", { value: f, textContent: f })));
  for (const id of ["logo_position", "brand_font", "handle_text"]) bind(id, id);
  brandControls();
  editorControls();
  refreshHints();

  fileInput("video", "videoName", "clearVideo", "or upload a video file");
  fileInput("music", "musicName", "clearMusic", "choose an audio file");
  $("go").onclick = () => submit(false);
  $("remix").onclick = () => submit(true);
  $("cancel").onclick = () => currentJob && api(`/api/jobs/${currentJob}/cancel`, { method: "POST" });
  $("openFolder").onclick = async () => {
    const r = await api(`/api/jobs/${currentJob}/open`, { method: "POST" });
    if (!r.ok) alert("Your clips are in:\n" + r.path);
  };
  aiSettings();
  const loadLog = async () => {
    const r = await api("/api/log?lines=400");
    $("logText").textContent = r.text || "The log is empty.";
    $("logPath").textContent = "File: " + r.path;
    $("logText").scrollTop = $("logText").scrollHeight;
  };
  $("btnLog").onclick = async () => { $("logDlg").showModal(); await loadLog(); };
  $("logRefresh").onclick = loadLog;
  $("logClose").onclick = () => $("logDlg").close();
  $("logCopy").onclick = async () => {
    try { await navigator.clipboard.writeText($("logText").textContent); $("logCopy").textContent = "Copied"; }
    catch (e) { const sel = getSelection(); sel.selectAllChildren($("logText")); }
    setTimeout(() => ($("logCopy").textContent = "Copy"), 1500);
  };

  const jobs = await loadHistory();
  const active = jobs.find((j) => j.status === "running" || j.status === "queued");
  if (active) watch(active.id);
}

// ---- AI provider settings ----------------------------------------------------
let providers = {};

function aiSettings() {
  const prov = () => $("provider").value;
  const show = () => {
    const key = prov(), p = providers[key];
    const claude = key === "claude", custom = key === "custom";
    $("model").hidden = !claude;
    $("modelText").hidden = claude;
    $("urlRow").hidden = claude || !(custom || p.local);
    $("keyRow").hidden = p.local;
    $("loadModels").hidden = claude;
    if (claude) $("model").value = p.model; else $("modelText").value = p.model;
    $("baseUrl").value = p.base_url || "";
    $("apiKey").value = "";
    $("apiKey").placeholder = claude ? "sk-ant-..." : "paste the key here";
    $("keyState").textContent = p.has_key ? "A key is saved. Leave the field empty to keep it." : "No key saved yet.";
    $("modelList").replaceChildren();
    $("testResult").textContent = "";
    const hint = $("provHint");
    hint.replaceChildren(claude ? "Best quality. Paid by usage, a few cents per video. "
      : p.local ? "Runs on this PC: free, private, no limits. Needs the program installed and a model downloaded. Slower without a good graphics card. "
      : custom ? "Any service with an OpenAI-compatible API. Enter its address, key and model name. "
      : "Free within daily limits, no credit card. Your transcripts are sent to this provider. ");
    if (p.key_url) hint.append(el("a", { href: p.key_url, target: "_blank", rel: "noopener", textContent: p.local ? "Download it here" : "Get a key here", style: "color:var(--accent)" }));
  };
  const collect = () => {
    const key = prov();
    const body = { provider: key, model: key === "claude" ? $("model").value : $("modelText").value.trim() };
    if ($("apiKey").value.trim()) body.api_key = $("apiKey").value.trim();
    if (key !== "claude") body.base_url = $("baseUrl").value.trim();
    return body;
  };
  const saveNow = async () => {
    cfg.status = await api("/api/settings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(collect()) });
    providers = await api("/api/ai/providers");
    pills(); refreshHints();
  };
  $("provider").onchange = show;
  $("btnSettings").onclick = async () => {
    providers = await api("/api/ai/providers");
    $("provider").replaceChildren(...Object.entries(providers).map(([k, v]) => el("option", { value: k, textContent: v.name })));
    $("provider").value = cfg.status.ai.provider;
    show();
    $("settingsDlg").showModal();
  };
  $("closeSettings").onclick = () => $("settingsDlg").close();
  $("saveSettings").onclick = async () => {
    try { await saveNow(); $("settingsDlg").close(); } catch (err) { $("testResult").textContent = err.message; }
  };
  $("testAi").onclick = async () => {
    $("testResult").style.color = "";
    $("testResult").textContent = "Testing, this takes up to 30 seconds...";
    try {
      await saveNow();
      const r = await api("/api/ai/test", { method: "POST" });
      $("testResult").textContent = (r.ok ? "Works. " : "Did not work. ") + r.message;
      $("testResult").style.color = r.ok ? "var(--good)" : "var(--bad)";
      $("keyState").textContent = providers[prov()].has_key ? "A key is saved. Leave the field empty to keep it." : "No key saved yet.";
      $("apiKey").value = "";
    } catch (err) { $("testResult").textContent = err.message; $("testResult").style.color = "var(--bad)"; }
  };
  $("loadModels").onclick = async () => {
    $("testResult").style.color = "";
    $("testResult").textContent = "Loading the model list...";
    try {
      await saveNow();
      const r = await api("/api/ai/models");
      $("modelList").replaceChildren(...r.models.map((m) => el("option", { value: m })));
      $("testResult").textContent = r.error ? "Could not load models: " + r.error
        : `${r.models.length} models found. Click the model field and start typing to pick one.`;
    } catch (err) { $("testResult").textContent = err.message; }
  };
}

function brandControls() {
  const pick = $("brandColorPick"), on = $("brandColorOn");
  pick.value = settings.brand_color_pick || "#ffe100";
  on.checked = !!settings.brand_color;
  const sync = () => { settings.brand_color_pick = pick.value; settings.brand_color = on.checked ? pick.value : ""; save(); };
  pick.oninput = sync; on.onchange = sync;

  const show = (has) => {
    $("logoPreview").hidden = $("clearLogo").hidden = !has;
    if (has) $("logoPreview").src = "/api/brand/logo?t=" + Date.now();
    $("logoName").textContent = has ? "replace the logo" : "upload a logo or watermark";
  };
  show(cfg.logo);
  $("logo").onchange = async () => {
    const f = $("logo").files[0];
    if (!f) return;
    const form = new FormData();
    form.append("logo", f);
    try {
      await api("/api/brand/logo", { method: "POST", body: form });
      show(true);
      if (!settings.logo_enabled) { settings.logo_enabled = true; $("logo_enabled").checked = true; save(); }
    } catch (err) { alert(err.message); }
    $("logo").value = "";
  };
  $("clearLogo").onclick = async () => {
    await api("/api/brand/logo", { method: "DELETE" });
    settings.logo_enabled = false; $("logo_enabled").checked = false; save();
    show(false);
  };
}

// ---- clip editor -----------------------------------------------------------
let ed = null;

function editorControls() {
  const picked = () => ed && ed.words.find((w) => w.i === ed.sel);
  $("edStart").onclick = () => { if (picked() && ed.sel <= ed.w1) { ed.w0 = ed.sel; drawEditor(); } };
  $("edEnd").onclick = () => { if (picked() && ed.sel >= ed.w0) { ed.w1 = ed.sel; drawEditor(); } };
  $("edFix").onclick = () => {
    const w = picked();
    if (!w) return;
    const text = prompt("Correct spelling for this word:", w.w || w.orig);
    if (text !== null && text.trim()) { w.w = text.trim(); drawEditor(); }
  };
  $("edHide").onclick = () => { const w = picked(); if (w) { w.w = w.w ? "" : w.orig; drawEditor(); } };
  $("edCancel").onclick = () => $("editDlg").close();
  $("edSave").onclick = async () => {
    const edits = {};
    for (const w of ed.words) if (w.w !== w.orig || w.was !== w.w) edits[w.i] = w.w;
    const body = { w0: ed.w0, w1: ed.w1, hook: $("edHook").value, title: $("edTitle").value, description: $("edDesc").value, word_edits: edits };
    try {
      await api(`/api/jobs/${ed.job}/clips/${ed.index}/rerender`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      $("editDlg").close();
      watch(ed.job);
    } catch (err) { $("edErr").textContent = err.message; }
  };
}

async function openEditor(jobId, index) {
  let data;
  try { data = await api(`/api/jobs/${jobId}/clips/${index}/words`); } catch (err) { alert(err.message); return; }
  ed = { job: jobId, index, w0: data.w0, w1: data.w1, sel: null, words: data.words.map((w) => Object.assign(w, { was: w.w })) };
  $("edHook").value = data.hook; $("edTitle").value = data.title; $("edDesc").value = data.description;
  $("edErr").textContent = "";
  drawEditor();
  $("editDlg").showModal();
  const first = $("edWords").querySelector(".edge");
  if (first) first.scrollIntoView({ block: "center" });
}

function drawEditor() {
  const byI = Object.fromEntries(ed.words.map((w) => [w.i, w]));
  const dur = byI[ed.w1].e - byI[ed.w0].s;
  $("edInfo").textContent = `Clip length before pause cutting: ${dur.toFixed(1)} s`;
  $("edWords").replaceChildren(...ed.words.flatMap((w) => {
    const inside = w.i >= ed.w0 && w.i <= ed.w1;
    const cls = ["word", inside ? "" : "out", w.i === ed.sel ? "sel" : "", w.i === ed.w0 || w.i === ed.w1 ? "edge" : "",
      !w.w ? "hidden" : w.w !== w.orig ? "changed" : ""].join(" ");
    const span = el("span", { className: cls, textContent: w.w || w.orig, title: `${w.s.toFixed(1)} s` });
    span.onclick = () => { ed.sel = w.i; drawEditor(); };
    return [span, " "];
  }));
}

function fileInput(id, nameId, clearId, placeholder) {
  $(id).onchange = () => {
    const f = $(id).files[0];
    $(nameId).textContent = f ? f.name : placeholder;
    $(clearId).hidden = !f;
  };
  $(clearId).onclick = () => { $(id).value = ""; $(id).onchange(); };
}

function goLabel() { return settings.mode === "story" ? "Generate narrated video" : "Generate shorts"; }

async function submit(reuse) {
  $("formError").textContent = "";
  const form = new FormData();
  form.append("settings", JSON.stringify(settings));
  if (reuse) {
    form.append("reuse", currentJob);
  } else if (settings.mode === "story") {
    const files = [...$("footage").files];
    if (!files.length) { $("formError").textContent = "Choose at least one video clip of your footage first."; return; }
    form.append("mode", "story");
    for (const f of files) form.append("footage", f);
    if (!cfg.voice_model) $("formError").textContent = "";
  } else if ($("video").files[0]) {
    form.append("video", $("video").files[0]);
  } else if ($("url").value.trim()) {
    form.append("url", $("url").value.trim());
  } else {
    $("formError").textContent = "Paste a YouTube link or choose a video file first.";
    return;
  }
  if ($("music").files[0]) form.append("music", $("music").files[0]);
  $("go").disabled = true;
  $("go").textContent = !reuse && (settings.mode === "story" || $("video").files[0]) ? "Uploading..." : "Starting...";
  try {
    const job = await api("/api/jobs", { method: "POST", body: form });
    watch(job.id);
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (err) {
    $("formError").textContent = err.message;
    $("go").disabled = false;
    $("go").textContent = goLabel();
  }
}

function watch(id) {
  currentJob = id;
  clearInterval(pollTimer);
  const tick = async () => {
    let job;
    try { job = await api(`/api/jobs/${id}`); } catch (e) { return; }
    if (currentJob !== id) return;
    render(job);
    if (!["queued", "running"].includes(job.status)) { clearInterval(pollTimer); loadHistory(); }
  };
  tick();
  pollTimer = setInterval(tick, 1200);
}

function render(job) {
  const busy = ["queued", "running"].includes(job.status);
  $("empty").hidden = true;
  $("jobCard").hidden = job.status === "done" && !job.message.startsWith("Re-render failed");
  $("go").disabled = busy;
  $("go").textContent = busy ? "Working..." : goLabel();
  $("cancel").hidden = !busy;
  $("jobTitle").textContent = job.title || (job.status === "queued" ? "Queued" : "Working on your video");
  const stages = job.kind === "story" ? STORY_STAGES : STAGES;
  const at = stages.findIndex(([k]) => k === job.stage);
  $("steps").replaceChildren(...stages.map(([k, label], i) =>
    el("div", { className: "step" + (busy && i === at ? " active" : (i < at || job.status === "done") ? " done" : ""), textContent: label })));
  $("barFill").style.width = Math.round(job.progress * 100) + "%";
  $("jobMsg").textContent = job.status === "error" ? "" : `${job.message}${busy ? ` (${Math.round(job.progress * 100)}%)` : ""}`;
  $("jobErr").textContent = job.status === "error" ? job.error : "";
  $("jobLog").textContent = (job.log || []).join("\n");

  // Say clearly when the AI was meant to be used but the run fell back.
  const fell = (job.log || []).filter((l) => /Using offline scoring instead|Using a built-in story instead/.test(l)).pop();
  $("aiWarn").hidden = !fell;
  if (fell) $("aiWarn").textContent = "The AI was not used for this run. " + fell.replace(/^\[[\d:]+\]\s*/, "") + " Open AI settings and press Save and test.";
  const clips = job.clips || [];
  $("results").hidden = clips.length === 0;
  $("remix").hidden = $("openFolder").hidden = busy;
  if (clips.length) {
    if (job.kind === "story") {
      const how = { pasted: "your own text", builtin: "built-in story" }[job.engine] || (job.engine ? "story by " + job.engine : "");
      $("resultsTitle").textContent = `${clips.length} narrated video${clips.length > 1 ? "s" : ""}${how ? ", " + how : ""}`;
    } else {
      const how = !job.engine || job.engine === "offline" ? "picked by offline scoring" : "picked by " + job.engine;
      $("resultsTitle").textContent = `${job.title || "Your video"}: ${clips.length} clip${clips.length > 1 ? "s" : ""}, ${how}`;
    }
    const box = $("clips");
    const sig = job.id + "|" + clips.map((c) => c.file).join("|");
    if (box.dataset.sig !== sig) {
      box.dataset.sig = sig;
      box.replaceChildren(...clips.map((c) => clipCard(job.id, c)));
    }
  }
}

function clipCard(jobId, c) {
  const isStory = c.kind === "story";
  const base = `/media/${jobId}/`;
  const tags = c.hashtags.map((h) => "#" + h).join(" ");
  const post = `${c.title}\n\n${c.description}\n\n${tags}`.trim();
  const copy = el("button", { type: "button", textContent: "Copy text" });
  copy.onclick = async () => {
    try { await navigator.clipboard.writeText(post); copy.textContent = "Copied"; } catch (e) { prompt("Copy the text:", post); }
    setTimeout(() => (copy.textContent = "Copy text"), 1500);
  };
  const cls = c.score >= 75 ? "" : c.score >= 55 ? " mid" : " low";
  const styleName = c.style ? (cfg.caption_styles[c.style] || {}).name : "No captions";
  const edit = el("button", { type: "button", textContent: "Edit" });
  edit.onclick = () => openEditor(jobId, c.index);
  const subs = el("div", { className: "subs" }, Object.entries(c.scores || {}).filter(([, v]) => v > 0).map(([k, v]) =>
    el("div", { className: "sub" }, [`${k[0].toUpperCase()}${k.slice(1)} ${v}`,
      el("div", { className: "track" }, el("div", { className: "fill" + (v >= 75 ? "" : v >= 55 ? " mid" : " low"), style: `width:${v}%` }))])));
  const mins = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  return el("div", { className: "clip" }, [
    el("video", { src: base + c.file, poster: base + c.thumb, controls: true, preload: "none", playsInline: true }),
    el("div", { className: "body" }, [
      el("div", { className: "meta" }, [
        ...(isStory ? [] : [el("span", { className: "score" + cls, textContent: `Score ${c.score}`, title: "Estimated short-form potential, 0 to 100" })]),
        el("span", { textContent: `${Math.round(c.duration)}s` }),
        el("span", { textContent: `${c.width}x${c.height}` }),
        el("span", { textContent: styleName }),
        ...(isStory ? [] : [el("span", { textContent: `from ${mins(c.source_start)}` })]),
      ]),
      subs,
      el("div", { className: "title", textContent: c.title }),
      el("div", { className: "text", textContent: `${c.description} ${tags}` }),
      el("div", { className: "why", textContent: isStory ? c.transcript : c.reason }),
      el("div", { className: "actions" }, [
        el("a", { className: "dl", href: base + c.file + "?download=1", textContent: "Download" }), ...(isStory ? [] : [edit]), copy]),
    ]),
  ]);
}

async function loadHistory() {
  const jobs = await api("/api/jobs");
  $("historyCard").hidden = jobs.length === 0;
  $("history").replaceChildren(...jobs.slice(0, 30).map((j) => {
    const title = el("span", { className: "t", textContent: j.title || j.source_url || j.id, title: "Open this run" });
    title.onclick = () => watch(j.id);
    const when = new Date(j.created * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
    const state = j.status === "done" ? `${j.clip_count} clips` : j.status;
    const del = el("button", { className: "link", textContent: "delete", type: "button" });
    del.onclick = async () => {
      if (!confirm("Delete this run and its video files?")) return;
      await api(`/api/jobs/${j.id}`, { method: "DELETE" });
      if (currentJob === j.id) location.reload(); else loadHistory();
    };
    return el("div", { className: "hist" }, [title, el("small", { textContent: `${state} · ${when}` }), del]);
  }));
  return jobs;
}

init().catch((err) => { document.body.prepend(el("p", { className: "error", textContent: "Could not start: " + err.message, style: "padding:20px" })); });
