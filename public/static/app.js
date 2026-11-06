/* NeuroScan AI - client.
 * Vanilla JS, no build step. All server data is rendered through `el()` /
 * textContent (never innerHTML) so file names or metadata cannot inject markup.
 */
(() => {
  "use strict";

  const CLASSES = ["glioma", "meningioma", "notumor", "pituitary"];
  const css = getComputedStyle(document.documentElement);
  const COLORS = Object.fromEntries(CLASSES.map((c) => [c, css.getPropertyValue(`--${c}`).trim()]));
  const LABELS = { glioma: "Glioma", meningioma: "Meningioma", notumor: "No tumor", pituitary: "Pituitary" };
  const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
  const ACCEPTED_EXTENSIONS = /\.(jpe?g|png|webp|bmp)$/i;
  // Platform errors (e.g. Vercel's edge) arrive as HTML/text, not JSON.
  const STATUS_MESSAGES = {
    413: () => `The file is too large (limit ${state.maxUploadMb} MB).`,
    429: () => "Too many requests. Please wait a moment and try again.",
    502: () => "The analysis service is unavailable. Please try again in a moment.",
    503: () => "The analysis service is starting up or unavailable. Please try again in a moment.",
    504: () => "The analysis took too long and timed out. Please try again.",
  };

  const state = {
    status: null,
    metrics: null,
    histories: {},
    meta: {},
    file: null,
    sampleId: null,
    previewUrl: null,
    charts: {},
    rendered: { benchmark: false, errors: false, training: false, guide: false },
    maxUploadMb: 10,
  };

  /* ------------------------------------------------------------ helpers */
  const $ = (sel, root = document) => root.querySelector(sel);

  function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props || {})) {
      if (value === undefined || value === null || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key === "style") Object.assign(node.style, value);
      else if (key === "vars") for (const [k, v] of Object.entries(value)) node.style.setProperty(k, v);
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : value);
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  const pct = (v, digits = 1) => (Number.isFinite(v) ? `${v.toFixed(digits)}%` : "–");
  const num = (v, digits = 3) => (Number.isFinite(v) ? v.toFixed(digits) : "–");
  const classLabel = (c) => LABELS[c] || c;
  const shortModel = (name) => (name || "").replace(/\s*\(.*\)$/, "");

  async function fetchJSON(url, options) {
    let res;
    try {
      res = await fetch(url, options);
    } catch (err) {
      if (err && err.name === "AbortError") throw err;
      throw new Error("Cannot reach the NeuroScan server. Is it running?");
    }
    let data = null;
    try { data = await res.json(); } catch { /* non-JSON error page */ }
    if (!res.ok) {
      const fallback = STATUS_MESSAGES[res.status] ? STATUS_MESSAGES[res.status]() : `Request failed (${res.status})`;
      const error = new Error((data && typeof data.error === "string" && data.error) || fallback);
      error.status = res.status;
      throw error;
    }
    if (data === null) throw new Error("The server sent an unexpected response.");
    return data;
  }

  let toastTimer = null;
  function toast(message, isError = false) {
    const node = $("#toast");
    node.textContent = message;
    node.classList.toggle("is-error", isError);
    node.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { node.hidden = true; }, 4500);
  }

  /* ------------------------------------------------------------ tabs */
  const tabs = Array.from(document.querySelectorAll('[role="tab"]'));

  function selectTab(tab, { focus = false, updateHash = true } = {}) {
    for (const t of tabs) {
      const selected = t === tab;
      t.setAttribute("aria-selected", String(selected));
      t.tabIndex = selected ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !selected;
    }
    if (focus) tab.focus();
    const name = tab.id.replace("tab-", "");
    if (updateHash) history.replaceState(null, "", name === "diagnose" ? location.pathname : `#${name}`);
    renderTab(name);
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectTab(tab));
    tab.addEventListener("keydown", (event) => {
      const moves = { ArrowRight: 1, ArrowLeft: -1, Home: -index, End: tabs.length - 1 - index };
      if (!(event.key in moves)) return;
      event.preventDefault();
      selectTab(tabs[(index + moves[event.key] + tabs.length) % tabs.length], { focus: true });
    });
  });

  function renderTab(name) {
    if (name === "benchmark") renderBenchmark();
    else if (name === "errors") renderErrors();
    else if (name === "training") renderTraining();
    else if (name === "guide") renderGuide();
  }

  /* ------------------------------------------------------------ status */
  async function loadStatus(attempt = 1) {
    const chip = $("#chip-engine");
    try {
      state.status = await fetchJSON("/api/status");
      const s = state.status;
      chip.classList.remove("is-error");
      chip.classList.add("is-ok");
      showInputError("");
      const engine = [s.backend && s.backend !== "none" ? s.backend.toUpperCase() : null, String(s.device || "cpu").toUpperCase()];
      $("#status-engine").textContent = `Engine ${engine.filter(Boolean).join(" · ")}${s.tta ? " · TTA" : ""}`;
      $("#status-models").textContent = `${s.loaded_models}/${s.total_models} models`;
      if (s.disclaimer) $("#disclaimer-text").textContent = s.disclaimer;
      if (s.max_upload_mb) {
        state.maxUploadMb = s.max_upload_mb;
        $("#max-upload").textContent = String(s.max_upload_mb);
      }
      if (s.loaded_models === 0) {
        showInputError("No trained models are installed on the server. Train models, then export them with scripts/export_onnx.py.");
      }
      renderGuide(true);
    } catch (err) {
      if (attempt < 2) {  // serverless cold starts can fail the very first call
        setTimeout(() => loadStatus(attempt + 1), 2500);
        return;
      }
      chip.classList.add("is-error");
      $("#status-engine").textContent = "Engine offline";
      showInputError(err.message);
    }
  }

  /* ------------------------------------------------------------ samples */
  async function loadSamples() {
    const box = $("#samples");
    try {
      // Static file served by the CDN: no serverless call needed.
      const { samples } = await fetchJSON("/samples/index.json");
      box.replaceChildren();
      if (!Array.isArray(samples) || !samples.length) {
        box.append(el("p", { class: "hint", text: "No sample scans found." }));
        return;
      }
      for (const sample of samples) {
        const button = el(
          "button",
          {
            class: "sample", type: "button", "aria-pressed": "false",
            "aria-label": `Sample ${classLabel(sample.class)} scan ${sample.filename}`,
            onclick: () => chooseSample(sample, button),
          },
          el("img", { src: sample.path, alt: "", loading: "lazy" }),
          el("span", { class: "sample__label" }, el("i", { style: { background: COLORS[sample.class] } }), classLabel(sample.class)),
        );
        box.append(button);
      }
    } catch (err) {
      box.replaceChildren(el("p", { class: "hint", text: `Samples unavailable: ${err.message}` }));
    } finally {
      box.setAttribute("aria-busy", "false");
    }
  }

  /* ------------------------------------------------------------ input */
  const dropzone = $("#dropzone");
  const fileInput = $("#file-input");
  const analyzeBtn = $("#analyze-btn");

  function showInputError(message) {
    const node = $("#input-error");
    node.textContent = message || "";
    node.hidden = !message;
  }

  function setPreview(src, caption) {
    $("#dropzone-empty").hidden = true;
    $("#dropzone-preview").hidden = false;
    $("#preview-img").src = src;
    $("#preview-name").textContent = caption;
    analyzeBtn.disabled = false;
    $("#clear-btn").hidden = false;
  }

  function clearSelection() {
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    Object.assign(state, { file: null, sampleId: null, previewUrl: null });
    fileInput.value = "";
    $("#dropzone-empty").hidden = false;
    $("#dropzone-preview").hidden = true;
    $("#preview-img").removeAttribute("src");
    analyzeBtn.disabled = true;
    $("#clear-btn").hidden = true;
    document.querySelectorAll(".sample[aria-pressed='true']").forEach((b) => b.setAttribute("aria-pressed", "false"));
    $("#result-view").hidden = true;
    $("#empty-state").hidden = false;
    showInputError("");
  }

  function chooseFile(file) {
    showInputError("");
    if (!file) return;
    const typeOk = file.type ? ACCEPTED_TYPES.includes(file.type) : ACCEPTED_EXTENSIONS.test(file.name || "");
    if (!typeOk) {
      showInputError(`"${file.name}" is not a JPEG, PNG, WebP or BMP image.`);
      return;
    }
    if (file.size === 0) {
      showInputError(`"${file.name}" is empty.`);
      return;
    }
    if (file.size > state.maxUploadMb * 1024 * 1024) {
      showInputError(`"${file.name}" is ${(file.size / 1048576).toFixed(1)} MB; the limit is ${state.maxUploadMb} MB.`);
      return;
    }
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.file = file;
    state.sampleId = null;
    state.previewUrl = URL.createObjectURL(file);
    document.querySelectorAll(".sample[aria-pressed='true']").forEach((b) => b.setAttribute("aria-pressed", "false"));
    setPreview(state.previewUrl, `${file.name} · ${(file.size / 1024).toFixed(0)} KB`);
  }

  async function chooseSample(sample, button) {
    showInputError("");
    document.querySelectorAll(".sample").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
    let blob;
    try {
      const res = await fetch(sample.path);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      blob = await res.blob();
    } catch (err) {
      showInputError(`Could not load the sample scan (${err.message}).`);
      return;
    }
    // Uploaded exactly like a user's file, so samples and uploads share one path.
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    const file = new File([blob], sample.filename, { type: blob.type || "image/jpeg" });
    Object.assign(state, { file, sampleId: sample.id, previewUrl: URL.createObjectURL(blob) });
    setPreview(state.previewUrl, `sample · ${sample.id}`);
    analyze();
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => chooseFile(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((type) => dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragover");
  }));
  ["dragleave", "drop"].forEach((type) => dropzone.addEventListener(type, () => dropzone.classList.remove("is-dragover")));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    chooseFile(event.dataTransfer.files && event.dataTransfer.files[0]);
  });
  $("#clear-btn").addEventListener("click", clearSelection);
  analyzeBtn.addEventListener("click", analyze);

  /* ------------------------------------------------------------ analyze */
  let inFlight = null;

  async function analyze() {
    if (!state.file && !state.sampleId) return;
    if (inFlight) inFlight.abort();
    const controller = new AbortController();
    inFlight = controller;

    const form = new FormData();
    if (state.file) form.append("file", state.file, state.file.name || "scan");
    else form.append("sample_id", state.sampleId);

    showInputError("");
    setBusy(true);
    try {
      const data = await fetchJSON("/api/predict", { method: "POST", body: form, signal: controller.signal });
      renderResult(data);
    } catch (err) {
      if (err.name === "AbortError") return;
      showInputError(err.message);
      toast(err.message, true);
    } finally {
      if (inFlight === controller) { inFlight = null; setBusy(false); }
    }
  }

  function setBusy(busy) {
    analyzeBtn.disabled = busy || (!state.file && !state.sampleId);
    analyzeBtn.classList.toggle("is-busy", busy);
    analyzeBtn.querySelector("span").textContent = busy ? "Analyzing…" : "Analyze scan";
    $("#results").setAttribute("aria-busy", String(busy));
    $("#lightbox").classList.toggle("is-busy", busy);
  }

  /* ------------------------------------------------------------ result */
  function renderResult(data) {
    $("#empty-state").hidden = true;
    const view = $("#result-view");
    view.hidden = false;
    view.classList.remove("is-entering");
    void view.offsetWidth; // restart the entrance animation
    view.classList.add("is-entering");

    renderVerdict(data);
    renderViewer(data);
    renderModels(data);
    renderCategories($("#categories"), data.categories_guide, true);
    if (window.matchMedia("(max-width: 900px)").matches) $("#verdict").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderVerdict(data) {
    const verdict = $("#verdict");
    const top = data.consensus_prediction;
    verdict.classList.toggle("is-inconclusive", data.inconclusive);
    verdict.style.setProperty("--verdict-color", data.inconclusive ? "var(--text)" : COLORS[top]);

    $("#verdict-title").textContent = data.consensus_display_name;
    const level = $("#verdict-level");
    const levelText = { high: "High confidence", moderate: "Moderate confidence", low: "Low confidence" };
    level.className = `level level--${data.confidence_level}`;
    level.textContent = levelText[data.confidence_level] || data.confidence_level;

    $("#verdict-agreement").textContent = `${data.agreement_text} · top class ${pct(data.consensus_confidence)}`;
    $("#verdict-time").textContent = `${Math.round(data.inference_ms)} ms${data.tta ? " · flip TTA" : ""}`;

    const bar = $("#spectrum-bar");
    const legend = $("#spectrum-legend");
    bar.replaceChildren();
    legend.replaceChildren();
    const probs = data.ensemble_probabilities;
    const ordered = [...CLASSES].sort((a, b) => probs[b] - probs[a]);
    for (const c of ordered) {
      const seg = el("span", { class: "spectrum__seg", vars: { "--seg": COLORS[c] }, title: `${classLabel(c)} ${pct(probs[c])}` });
      bar.append(seg);
      requestAnimationFrame(() => { seg.style.width = `${probs[c]}%`; });
    }
    for (const c of CLASSES) {
      legend.append(el("li", { class: c === top ? "is-top" : "", vars: { "--seg": COLORS[c] } },
        el("i"), classLabel(c), el("b", { text: pct(probs[c]) })));
    }
    $("#spectrum").setAttribute("aria-label",
      `Ensemble probabilities: ${CLASSES.map((c) => `${classLabel(c)} ${pct(probs[c])}`).join(", ")}. Confidence thresholds at 60 and 85 percent.`);

    $("#verdict-summary").textContent = data.plain_summary;
    const warnings = $("#verdict-warnings");
    warnings.replaceChildren(...(data.warnings || []).map((w) =>
      el("li", {}, el("i", { class: "fa-solid fa-circle-exclamation", "aria-hidden": "true" }), w)));
    warnings.hidden = !(data.warnings || []).length;
  }

  function renderViewer(data) {
    const cam = data.gradcam;
    $("#viewer-base").src = cam ? cam.input_b64 : data.original_image_b64;
    const heat = $("#viewer-heat");
    if (cam) {
      heat.src = cam.heatmap_b64;
      heat.hidden = false;
      $("#viewer-label").textContent = `${shortModel(cam.display_name)} → ${classLabel(cam.target_class)}`;
    } else {
      heat.hidden = true;
      heat.removeAttribute("src");
      $("#viewer-label").textContent = "no Grad-CAM model available";
    }
    $("#viewer-controls").hidden = !cam;
    updateOpacity();
  }

  function updateOpacity() {
    const value = Number($("#opacity").value);
    $("#viewer-heat").style.opacity = String(value / 100);
    $("#opacity-out").textContent = `${value}%`;
  }
  $("#opacity").addEventListener("input", updateOpacity);

  function renderModels(data) {
    const list = $("#model-cards");
    list.replaceChildren();
    for (const m of data.model_results) {
      const bars = el("div", { class: "bars" });
      for (const c of CLASSES) {
        const fill = el("span", { class: "bar__fill", vars: { "--seg": COLORS[c] } });
        bars.append(el("div", { class: "bar" },
          el("span", { text: classLabel(c) }),
          el("span", { class: "bar__track" }, fill),
          el("span", { class: "bar__val", text: pct(m.confidences[c]) })));
        requestAnimationFrame(() => { fill.style.width = `${m.confidences[c]}%`; });
      }
      const badges = el("div", { class: "badges" },
        el("span", { class: "badge", text: m.calibrated ? `calibrated T=${m.temperature}` : "uncalibrated" }),
        m.legacy_checkpoint ? el("span", { class: "badge", text: "legacy checkpoint" }) : null);
      list.append(el("section", { class: "model", "aria-label": m.display_name },
        el("div", { class: "model__top" },
          el("span", { class: "model__name", text: m.display_name }),
          el("span", { class: "model__weight", text: `weight ${(m.weight * 100).toFixed(0)}%` })),
        el("div", { class: "model__call" },
          el("span", { class: "model__class", vars: { "--seg": COLORS[m.predicted_class] } }, el("i"), classLabel(m.predicted_class)),
          el("span", { class: "model__conf", text: pct(m.confidence_pct) })),
        bars,
        badges));
    }
    const note = $("#unavailable-note");
    const missing = data.unavailable_models || [];
    note.hidden = !missing.length;
    note.textContent = missing.length
      ? `Not included: ${missing.map((m) => `${m.display_name} (${m.reason})`).join("; ")}.`
      : "";
  }

  function renderCategories(container, guide, withScores) {
    container.replaceChildren();
    for (const cat of guide) {
      const detected = withScores && cat.is_detected;
      container.append(el("article", { class: `category${detected ? " is-detected" : ""}`, vars: { "--seg": cat.color } },
        el("div", { class: "category__head" },
          el("h3", { class: "category__name", text: cat.display_name }),
          withScores ? el("span", { class: "category__pct", text: pct(cat.average_confidence_pct) }) : null),
        el("span", { class: "category__tag", text: detected ? "Best match" : cat.severity }),
        el("p", {}, el("strong", { text: "What it is: " }), cat.simple_summary),
        el("p", {}, el("strong", { text: "Details: " }), cat.plain_details),
        withScores && detected ? el("p", {}, el("strong", { text: "What it means: " }), cat.what_it_means) : null));
    }
  }

  /* ------------------------------------------------------------ guide */
  function renderGuide(force = false) {
    if (!state.status || (state.rendered.guide && !force)) return;
    const guide = CLASSES.map((c) => ({ class_key: c, ...(state.status.class_metadata[c] || {}) }));
    renderCategories($("#guide-categories"), guide, false);
    state.rendered.guide = true;
  }

  /* ------------------------------------------------------------ metrics */
  async function loadMetrics() {
    try {
      const data = await fetchJSON("/api/metrics");
      state.metrics = data.status === "available" ? data.metrics : null;
      state.histories = data.histories || {};
      state.meta = data.meta || {};
      const notice = $("#metrics-notice");
      notice.hidden = data.status === "available";
      notice.textContent = data.message || "";
    } catch (err) {
      state.metrics = null;
      $("#metrics-notice").hidden = false;
      $("#metrics-notice").textContent = `Metrics unavailable: ${err.message}`;
    }
    state.rendered.benchmark = state.rendered.errors = state.rendered.training = false;
    const active = tabs.find((t) => t.getAttribute("aria-selected") === "true");
    if (active) renderTab(active.id.replace("tab-", ""));
  }
  $("#refresh-metrics").addEventListener("click", async () => {
    await loadMetrics();
    toast("Benchmark data refreshed");
  });

  function chartsAvailable(canvasIds) {
    if (window.Chart) return true;
    for (const id of canvasIds) {
      const box = document.getElementById(id).parentElement;
      box.replaceChildren(el("p", { class: "chart-fallback", text: "Charts could not be loaded." }));
    }
    return false;
  }

  function setChartDefaults() {
    if (!window.Chart || setChartDefaults.done) return;
    const C = window.Chart;
    C.defaults.color = css.getPropertyValue("--text-muted").trim();
    C.defaults.borderColor = "rgba(236,232,222,0.08)";
    C.defaults.font.family = "'IBM Plex Mono', ui-monospace, monospace";
    C.defaults.font.size = 11;
    C.defaults.plugins.legend.labels.boxWidth = 10;
    C.defaults.plugins.legend.labels.boxHeight = 10;
    C.defaults.maintainAspectRatio = false;
    C.defaults.animation.duration = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 600;
    setChartDefaults.done = true;
  }

  function drawChart(id, config) {
    if (state.charts[id]) state.charts[id].destroy();
    state.charts[id] = new window.Chart(document.getElementById(id), config);
  }

  /* Tumor-vs-normal view derived from the confusion matrix, so older metric
   * files (without these fields) still get real sensitivity/specificity. */
  function withScreening(m) {
    if (m.tumor_sensitivity != null || !Array.isArray(m.confusion_matrix)) return m;
    const cm = m.confusion_matrix;
    const n = CLASSES.indexOf("notumor");
    let tumors = 0, missed = 0;
    cm.forEach((row, i) => { if (i !== n) { tumors += row.reduce((a, b) => a + b, 0); missed += row[n]; } });
    const normals = cm[n].reduce((a, b) => a + b, 0);
    const falseAlarms = normals - cm[n][n];
    return {
      ...m,
      tumor_sensitivity: tumors ? 1 - missed / tumors : null,
      tumor_specificity: normals ? 1 - falseAlarms / normals : null,
      missed_tumors: missed,
      false_alarms: falseAlarms,
    };
  }

  const modelEntries = () => Object.entries(state.metrics || {}).map(([k, m]) => [k, withScreening(m)]);
  const displayNameFor = (key) => {
    const fromStatus = state.status && state.status.models.find((m) => m.key === key);
    return (fromStatus && fromStatus.display_name) || (state.metrics && state.metrics[key] && state.metrics[key].display_name) || key;
  };

  function renderBenchmark() {
    if (state.rendered.benchmark) return;
    const thead = $("#benchmark-table thead");
    const tbody = $("#benchmark-table tbody");
    thead.replaceChildren();
    tbody.replaceChildren();
    $("#benchmark-stats").replaceChildren();
    if (!state.metrics) return;

    const entries = modelEntries();
    const singles = entries.filter(([k]) => k !== "ensemble");
    const testN = state.meta.num_test_images || (entries[0] && entries[0][1].total_samples);
    const leak = state.meta.test_leakage && state.meta.test_leakage.overall;
    $("#benchmark-lede").textContent =
      `${testN ? `${testN} held-out test images` : "Held-out test set"}${state.meta.tta ? " · flip TTA" : ""}. ` +
      "Each model is evaluated with its own preprocessing and calibration.";

    // headline stats
    const best = (key) => {
      const have = singles.filter(([, m]) => Number.isFinite(m[key]));
      return have.length ? have.reduce((a, b) => (b[1][key] > a[1][key] ? b : a)) : null;
    };
    const card = (entry, key, format) => entry ? [format(entry[1][key]), shortModel(entry[1].display_name)] : ["–", "not in this metrics file"];
    if (singles.length) {
      const stats = [
        ["Best accuracy", ...card(best("test_accuracy_pct"), "test_accuracy_pct", (v) => pct(v))],
        ["Best macro F1", ...card(best("macro_f1"), "macro_f1", (v) => num(v))],
        ["Best tumor sensitivity", ...card(best("tumor_sensitivity"), "tumor_sensitivity", (v) => pct(v * 100))],
        ["Train→test duplicates", leak != null ? pct(leak * 100) : "not audited",
          leak != null ? "test images with a near-copy in Training/" : "run evaluate.py with Training/ present"],
      ];
      for (const [label, value, note] of stats) {
        $("#benchmark-stats").append(el("div", { class: "stat" },
          el("div", { class: "stat__label", text: label }), el("div", { class: "stat__value", text: value }),
          el("div", { class: "stat__note", text: note })));
      }
    }

    const columns = [
      ["Model", null],
      ["Accuracy", "test_accuracy_pct", 1, (v) => pct(v, 2)],
      ["Balanced acc.", "balanced_accuracy_pct", 1, (v) => pct(v, 2)],
      ["Macro F1", "macro_f1", 1, (v) => num(v)],
      ["ROC-AUC", "auc_macro", 1, (v) => num(v)],
      ["Log loss", "cross_entropy_loss", -1, (v) => num(v)],
      ["ECE", "ece", -1, (v) => num(v)],
      ["Tumor sens.", "tumor_sensitivity", 1, (v) => pct(v * 100)],
      ["Tumor spec.", "tumor_specificity", 1, (v) => pct(v * 100)],
      ["Latency", "latency_ms_per_image", -1, (v) => `${v.toFixed(1)} ms`],
      ["Params", "parameters_total", 0, (v) => `${(v / 1e6).toFixed(2)}M`],
    ];
    thead.append(el("tr", {}, columns.map(([label]) => el("th", { scope: "col", text: label }))));
    const bestValue = {};
    for (const [, key, dir] of columns) {
      if (!key || !dir) continue;
      const values = singles.map(([, m]) => m[key]).filter(Number.isFinite);
      if (values.length) bestValue[key] = dir > 0 ? Math.max(...values) : Math.min(...values);
    }
    for (const [key, m] of entries) {
      const row = el("tr", { class: key === "ensemble" ? "is-ensemble" : "" });
      for (const [label, field, dir, format] of columns) {
        if (!field) { row.append(el("td", { text: m.display_name })); continue; }
        const value = m[field];
        const isBest = dir && key !== "ensemble" && Number.isFinite(value) && value === bestValue[field];
        row.append(el("td", { class: `${isBest ? "is-best" : ""}${Number.isFinite(value) ? "" : " muted-cell"}`,
          text: Number.isFinite(value) ? format(value) : "–", title: isBest ? `Best ${label}` : null }));
      }
      tbody.append(row);
    }

    if (!chartsAvailable(["chart-accuracy", "chart-f1", "chart-perclass", "chart-screening"])) return;
    setChartDefaults();
    const labels = entries.map(([, m]) => shortModel(m.display_name));
    const ivory = "rgba(236,232,222,0.85)";
    const muted = "rgba(236,232,222,0.28)";
    const barOpts = (max) => ({
      scales: { y: { beginAtZero: true, max, grid: { color: "rgba(236,232,222,0.06)" } }, x: { grid: { display: false } } },
      plugins: { legend: { position: "bottom" } },
      datasets: { bar: { borderRadius: 3, maxBarThickness: 28 } },
    });
    drawChart("chart-accuracy", { type: "bar", data: { labels, datasets: [
      { label: "Accuracy %", data: entries.map(([, m]) => m.test_accuracy_pct), backgroundColor: ivory },
      { label: "Error rate %", data: entries.map(([, m]) => m.error_rate_pct), backgroundColor: COLORS.glioma },
    ] }, options: barOpts(100) });
    drawChart("chart-f1", { type: "bar", data: { labels, datasets: [
      { label: "Macro F1", data: entries.map(([, m]) => m.macro_f1), backgroundColor: ivory },
      { label: "ROC-AUC", data: entries.map(([, m]) => m.auc_macro), backgroundColor: muted },
    ] }, options: barOpts(1) });
    drawChart("chart-perclass", { type: "bar", data: { labels, datasets: CLASSES.map((c) => ({
      label: classLabel(c), data: entries.map(([, m]) => m[`${c}_f1`]), backgroundColor: COLORS[c],
    })) }, options: barOpts(1) });
    drawChart("chart-screening", { type: "bar", data: { labels, datasets: [
      { label: "Tumor sensitivity", data: entries.map(([, m]) => (m.tumor_sensitivity ?? NaN) * 100), backgroundColor: ivory },
      { label: "Specificity", data: entries.map(([, m]) => (m.tumor_specificity ?? NaN) * 100), backgroundColor: COLORS.notumor },
    ] }, options: barOpts(100) });
    state.rendered.benchmark = true;
  }

  /* ------------------------------------------------------------ errors */
  const cmSelect = $("#cm-model");
  cmSelect.addEventListener("change", () => { state.rendered.errors = false; renderErrors(); });

  function renderErrors() {
    if (state.rendered.errors) return;
    const box = $("#confusion-matrix");
    const insights = $("#error-insights");
    box.replaceChildren();
    insights.replaceChildren();
    const entries = modelEntries();
    if (!entries.length) {
      box.append(el("p", { class: "hint", text: "No evaluation results yet. Run python evaluate.py." }));
      return;
    }
    if (cmSelect.options.length !== entries.length) {
      const previous = cmSelect.value;
      cmSelect.replaceChildren(...entries.map(([k, m]) => el("option", { value: k, text: m.display_name })));
      const preferred = entries.find(([k]) => k === previous) || entries.find(([k]) => k === "ensemble") || entries[0];
      cmSelect.value = preferred[0];
    }
    const m = withScreening(state.metrics[cmSelect.value]);
    const cm = m.confusion_matrix;

    box.append(el("div", { class: "cm__axis", text: "true ↓ / predicted →" }));
    for (const c of CLASSES) box.append(el("div", { class: "cm__colhead", vars: { "--seg": COLORS[c] } }, el("i"), classLabel(c)));
    cm.forEach((row, i) => {
      const total = row.reduce((a, b) => a + b, 0) || 1;
      box.append(el("div", { class: "cm__rowhead", vars: { "--seg": COLORS[CLASSES[i]] } }, el("i"), classLabel(CLASSES[i])));
      row.forEach((count, j) => {
        const share = count / total;
        const diag = i === j;
        const intensity = diag ? share : Math.min(1, share * 2.5); // make small error rates visible
        box.append(el("div", {
          class: `cm__cell ${diag ? "is-diag" : "is-off"}${intensity > 0.45 ? " is-strong" : ""}`,
          vars: { "--v": intensity.toFixed(3) }, role: "cell",
          "aria-label": `True ${classLabel(CLASSES[i])}, predicted ${classLabel(CLASSES[j])}: ${count} of ${total} (${pct(share * 100)})`,
        }, el("span", { class: "cm__pct", text: pct(share * 100, share < 0.1 && share > 0 ? 1 : 0) }), el("span", { class: "cm__n", text: `${count}/${total}` })));
      });
    });

    // data-driven insights for the selected model
    let worst = null;
    cm.forEach((row, i) => row.forEach((count, j) => {
      if (i !== j && (!worst || count > worst.count)) worst = { i, j, count, total: row.reduce((a, b) => a + b, 0) };
    }));
    const missRates = CLASSES.map((c, i) => ({ c, miss: 1 - cm[i][i] / (cm[i].reduce((a, b) => a + b, 0) || 1) }));
    const hardest = missRates.reduce((a, b) => (b.miss > a.miss ? b : a));
    const items = [
      ["Most common mistake", worst && worst.count ? `${classLabel(CLASSES[worst.i])} → ${classLabel(CLASSES[worst.j])}` : "None",
        worst && worst.count ? `${worst.count} of ${worst.total} ${classLabel(CLASSES[worst.i]).toLowerCase()} scans (${pct(100 * worst.count / worst.total)})` : "No off-diagonal errors."],
      ["Hardest class", classLabel(hardest.c), `${pct(hardest.miss * 100)} of these scans are missed (recall ${pct((1 - hardest.miss) * 100)}).`],
    ];
    if (m.missed_tumors != null) {
      items.push(["Missed tumors", String(m.missed_tumors), `tumor scans labelled "no tumor" · sensitivity ${pct(m.tumor_sensitivity * 100)}`]);
      items.push(["False alarms", String(m.false_alarms), `normal scans labelled as a tumor · specificity ${pct(m.tumor_specificity * 100)}`]);
    }
    if (Number.isFinite(m.ece)) {
      items.push(["Calibration error", num(m.ece), m.ece_uncalibrated != null
        ? `${num(m.ece_uncalibrated)} before temperature scaling`
        : "gap between displayed confidence and accuracy"]);
    }
    for (const [title, value, note] of items) insights.append(el("li", {}, el("span", { text: title }), el("b", { text: value }), el("span", { text: note })));
    state.rendered.errors = true;
  }

  /* ------------------------------------------------------------ training */
  const lastOf = (values) => (Array.isArray(values) && values.length ? values[values.length - 1] : null);
  const maxOf = (values) => (Array.isArray(values) && values.length ? Math.max(...values) : null);

  function renderTraining() {
    if (state.rendered.training) return;
    const grid = $("#training-grid");
    grid.replaceChildren();
    const entries = Object.entries(state.histories || {}).filter(([, h]) => h && Array.isArray(h.train_loss) && h.train_loss.length);
    const notice = $("#training-notice");
    notice.hidden = entries.length > 0;
    notice.textContent = "No training histories found in results/. Train a model with python train.py.";
    if (!entries.length) return;
    const canDraw = !!window.Chart;
    setChartDefaults();

    for (const [key, h] of entries) {
      const test = state.metrics && state.metrics[key];
      const name = h.display_name && h.display_name !== key ? h.display_name : displayNameFor(key);
      const canvasId = `train-${key}`;
      const fact = (label, value) => el("div", { class: "fact" }, el("span", { text: label }), el("b", { text: value }));
      grid.append(el("article", { class: "panel train-card" },
        el("div", { class: "train-card__head" }, el("h2", { text: name })),
        el("div", { class: "train-card__facts" },
          fact("epochs", String(h.train_loss.length)),
          fact("train acc", pct(lastOf(h.train_acc))),
          fact("best val acc", pct(maxOf(h.val_acc))),
          fact("test acc", pct(test ? test.test_accuracy_pct : NaN))),
        el("div", { class: "chart__canvas" }, canDraw
          ? el("canvas", { id: canvasId, role: "img", "aria-label": `Training and validation loss and accuracy for ${name}` })
          : el("p", { class: "chart-fallback", text: "Charts could not be loaded." }))));
      if (!canDraw) continue;
      const epochs = h.train_loss.map((_, i) => i + 1);
      drawChart(canvasId, {
        type: "line",
        data: { labels: epochs, datasets: [
          { label: "train loss", data: h.train_loss, borderColor: "rgba(236,232,222,0.9)", backgroundColor: "rgba(236,232,222,0.9)", tension: 0.25, pointRadius: 2, yAxisID: "y" },
          { label: "val loss", data: h.val_loss, borderColor: COLORS.glioma, backgroundColor: COLORS.glioma, tension: 0.25, pointRadius: 2, yAxisID: "y" },
          { label: "train acc %", data: h.train_acc, borderColor: "rgba(236,232,222,0.45)", borderDash: [4, 4], pointRadius: 0, tension: 0.25, yAxisID: "y1" },
          { label: "val acc %", data: h.val_acc, borderColor: COLORS.pituitary, borderDash: [4, 4], pointRadius: 0, tension: 0.25, yAxisID: "y1" },
        ] },
        options: {
          interaction: { mode: "index", intersect: false },
          plugins: { legend: { position: "bottom" } },
          scales: {
            x: { title: { display: true, text: "epoch" }, grid: { display: false } },
            y: { position: "left", title: { display: true, text: "loss" }, beginAtZero: true, grid: { color: "rgba(236,232,222,0.06)" } },
            y1: { position: "right", title: { display: true, text: "accuracy %" }, min: 0, max: 100, grid: { display: false } },
          },
        },
      });
    }
    state.rendered.training = true;
  }

  /* ------------------------------------------------------------ boot */
  const initial = tabs.find((t) => `#${t.id.replace("tab-", "")}` === location.hash);
  if (initial) selectTab(initial, { updateHash: false });
  loadStatus();
  loadSamples();
  loadMetrics();
})();
