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
  $("reframeHint").textContent = REFRAME[settings.reframe][1];
  const keyed = cfg.status.api_key_set;
  $("aiHint").textContent = !settings.use_ai ? "Moments are picked by offline scoring (keywords, pacing, clean endings)."
    : keyed ? `Using ${cfg.claude_models[cfg.status.model] || cfg.status.model}.`
    : "No API key saved yet, so offline scoring is used. Add a key under AI settings for much better picks and hooks.";
}

function pills() {
  const s = cfg.status;
  const items = [
    [s.ffmpeg.found ? "FFmpeg ready" : "FFmpeg missing", s.ffmpeg.found ? "ok" : "bad"],
    [s.ffmpeg.libass ? "Captions ready" : "No caption support in FFmpeg", s.ffmpeg.libass ? "ok" : "bad"],
    [s.js_runtime.length ? "YouTube ready" : "YouTube: install Deno", s.js_runtime.length ? "ok" : "warn"],
    [s.api_key_set ? "Claude connected" : "Offline mode (no API key)", s.api_key_set ? "ok" : "warn"],
  ];
  $("pills").replaceChildren(...items.map(([text, cls]) => el("span", { className: "pill " + cls, textContent: text })));
}

async function init() {
  cfg = await api("/api/config");
  let stored = {};
  try { stored = JSON.parse(localStorage.getItem("clipforge.settings") || "{}"); } catch (e) { /* ignore */ }
  settings = Object.assign({}, cfg.defaults, stored);
  pills();

  segmented("aspect", cfg.aspects.map((a) => [a, a, SHAPES[a]]), "aspect", refreshHints);
  segmented("reframe", Object.entries(REFRAME).map(([k, v]) => [k, v[0]]), "reframe", refreshHints);
  captionStyles();
  $("hook_style").replaceChildren(...Object.entries(cfg.hook_styles).map(([k, v]) => el("option", { value: k, textContent: v })),
    el("option", { value: "none", textContent: "No hook text" }));
  $("whisper_model").replaceChildren(...Object.entries(cfg.whisper_models).map(([k, v]) => el("option", { value: k, textContent: v })));
  $("model").replaceChildren(...Object.entries(cfg.claude_models).map(([k, v]) => el("option", { value: k, textContent: v })));
  $("model").value = cfg.status.model;

  for (const id of ["resolution", "fps", "quality", "caption_position", "hook_style", "whisper_model", "instructions"]) bind(id, id);
  bind("num_clips", "num_clips", "num", "numClipsVal");
  bind("min_len", "min_len", "num"); bind("max_len", "max_len", "num"); bind("hook_seconds", "hook_seconds", "num");
  bind("caption_scale", "caption_scale", "num", "capScaleVal", (v) => Math.round(v * 100) + "%");
  bind("silence_gap", "silence_gap", "num", "gapVal", (v) => Number(v).toFixed(2) + " s");
  bind("zoom_amount", "zoom_amount", "num", "zoomVal", (v) => "+" + Math.round((v - 1) * 100) + "%");
  bind("music_volume", "music_volume", "num", "musicVal", (v) => Math.round(v * 100) + "%");
  for (const id of ["remove_silence", "zooms", "progress_bar", "normalize_audio", "use_ai"]) bind(id, id, "check");
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
  $("btnSettings").onclick = () => {
    $("apiKey").value = "";
    $("keyState").textContent = cfg.status.api_key_set ? "A key is saved. Leave the field empty to keep it." : "No key saved.";
    $("settingsDlg").showModal();
  };
  $("settingsDlg").addEventListener("close", async () => {
    if ($("settingsDlg").returnValue !== "save") return;
    const body = { model: $("model").value };
    if ($("apiKey").value.trim()) body.api_key = $("apiKey").value.trim();
    cfg.status = await api("/api/settings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    pills(); refreshHints();
  });

  const jobs = await loadHistory();
  const active = jobs.find((j) => j.status === "running" || j.status === "queued");
  if (active) watch(active.id);
}

function fileInput(id, nameId, clearId, placeholder) {
  $(id).onchange = () => {
    const f = $(id).files[0];
    $(nameId).textContent = f ? f.name : placeholder;
    $(clearId).hidden = !f;
  };
  $(clearId).onclick = () => { $(id).value = ""; $(id).onchange(); };
}

async function submit(reuse) {
  $("formError").textContent = "";
  const form = new FormData();
  form.append("settings", JSON.stringify(settings));
  if (reuse) {
    form.append("reuse", currentJob);
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
  $("go").textContent = $("video").files[0] && !reuse ? "Uploading..." : "Starting...";
  try {
    const job = await api("/api/jobs", { method: "POST", body: form });
    watch(job.id);
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (err) {
    $("formError").textContent = err.message;
    $("go").disabled = false;
    $("go").textContent = "Generate shorts";
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
  $("jobCard").hidden = job.status === "done";
  $("go").disabled = busy;
  $("go").textContent = busy ? "Working..." : "Generate shorts";
  $("cancel").hidden = !busy;
  $("jobTitle").textContent = job.title || (job.status === "queued" ? "Queued" : "Working on your video");
  const at = STAGES.findIndex(([k]) => k === job.stage);
  $("steps").replaceChildren(...STAGES.map(([k, label], i) =>
    el("div", { className: "step" + (busy && i === at ? " active" : (i < at || job.status === "done") ? " done" : ""), textContent: label })));
  $("barFill").style.width = Math.round(job.progress * 100) + "%";
  $("jobMsg").textContent = job.status === "error" ? "" : `${job.message}${busy ? ` (${Math.round(job.progress * 100)}%)` : ""}`;
  $("jobErr").textContent = job.status === "error" ? job.error : "";
  $("jobLog").textContent = (job.log || []).join("\n");

  const clips = job.clips || [];
  $("results").hidden = clips.length === 0;
  $("remix").hidden = $("openFolder").hidden = busy;
  if (clips.length) {
    const how = job.engine === "claude" ? "picked by Claude" : "picked by offline scoring";
    $("resultsTitle").textContent = `${job.title || "Your video"}: ${clips.length} clip${clips.length > 1 ? "s" : ""}, ${how}`;
    const box = $("clips");
    if (box.dataset.job !== job.id || box.childElementCount !== clips.length) {
      box.dataset.job = job.id;
      box.replaceChildren(...clips.map((c) => clipCard(job.id, c)));
    }
  }
}

function clipCard(jobId, c) {
  const base = `/media/${jobId}/`;
  const tags = c.hashtags.map((h) => "#" + h).join(" ");
  const post = `${c.title}\n\n${c.description}\n\n${tags}`.trim();
  const copy = el("button", { type: "button", textContent: "Copy post text" });
  copy.onclick = async () => {
    try { await navigator.clipboard.writeText(post); copy.textContent = "Copied"; } catch (e) { prompt("Copy the text:", post); }
    setTimeout(() => (copy.textContent = "Copy post text"), 1500);
  };
  const cls = c.score >= 75 ? "" : c.score >= 55 ? " mid" : " low";
  const styleName = c.style ? (cfg.caption_styles[c.style] || {}).name : "No captions";
  const mins = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  return el("div", { className: "clip" }, [
    el("video", { src: base + c.file, poster: base + c.thumb, controls: true, preload: "none", playsInline: true }),
    el("div", { className: "body" }, [
      el("div", { className: "meta" }, [
        el("span", { className: "score" + cls, textContent: `Score ${c.score}`, title: "Estimated short-form potential, 0 to 100" }),
        el("span", { textContent: `${Math.round(c.duration)}s` }),
        el("span", { textContent: `${c.width}x${c.height}` }),
        el("span", { textContent: styleName }),
        el("span", { textContent: `from ${mins(c.source_start)}` }),
      ]),
      el("div", { className: "title", textContent: c.title }),
      el("div", { className: "text", textContent: `${c.description} ${tags}` }),
      el("div", { className: "why", textContent: c.reason }),
      el("div", { className: "actions" }, [
        el("a", { className: "dl", href: base + c.file + "?download=1", textContent: "Download" }), copy]),
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
