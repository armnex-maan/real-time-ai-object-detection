(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const fmt = (n) => (Number.isFinite(n) ? n : 0).toLocaleString();

  const spark = $("fpsSpark");
  const sctx = spark.getContext("2d");

  function drawSpark(values) {
    const w = spark.clientWidth;
    const h = spark.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    if (spark.width !== w * dpr || spark.height !== h * dpr) {
      spark.width = w * dpr;
      spark.height = h * dpr;
      sctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    sctx.clearRect(0, 0, w, h);
    if (!values.length) return;
    const max = Math.max(1, ...values);
    const step = w / Math.max(1, values.length - 1);
    sctx.beginPath();
    values.forEach((v, i) => {
      const x = i * step;
      const y = h - (v / max) * (h - 6) - 3;
      if (i === 0) sctx.moveTo(x, y);
      else sctx.lineTo(x, y);
    });
    sctx.strokeStyle = "#5eead4";
    sctx.lineWidth = 2;
    sctx.stroke();
    sctx.lineTo(w, h);
    sctx.lineTo(0, h);
    sctx.closePath();
    sctx.fillStyle = "rgba(94, 234, 212, 0.12)";
    sctx.fill();
  }

  function renderTopClasses(top) {
    const box = $("topClasses");
    if (!top.length) {
      box.innerHTML = '<p class="muted">Waiting for detections…</p>';
      return;
    }
    const max = top[0][1] || 1;
    box.innerHTML = top
      .map(
        ([name, n]) => `
      <div class="bar-row">
        <span class="name">${name}</span>
        <span class="bar-track"><span class="bar-fill" style="width:${(n / max) * 100}%"></span></span>
        <span class="count">${fmt(n)}</span>
      </div>`
      )
      .join("");
  }

  function renderCurrent(current) {
    const box = $("currentObjects");
    const entries = Object.entries(current).sort((a, b) => b[1] - a[1]);
    if (!entries.length) {
      box.innerHTML = '<p class="muted">Nothing detected yet.</p>';
      return;
    }
    box.innerHTML = entries
      .map(
        ([name, n]) => `
      <span class="chip">${name} <span class="n">${n}</span></span>`
      )
      .join("");
  }

  function fmtUptime(s) {
    s = Math.floor(s);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    if (h) return `${h}h ${m}m ${sec}s`;
    if (m) return `${m}m ${sec}s`;
    return `${sec}s`;
  }

  async function tick() {
    try {
      const r = await fetch("/api/stats", { cache: "no-store" });
      if (!r.ok) throw new Error(r.statusText);
      const s = await r.json();

      $("statFps").textContent = s.fps.toFixed(1);
      $("fpsBadge").textContent = `${s.fps.toFixed(1)} FPS`;
      $("statFrames").textContent = fmt(s.frames_processed);
      $("uptime").textContent = `Uptime: ${fmtUptime(s.uptime_seconds)}`;

      const onScreen = Object.values(s.current_counts).reduce((a, b) => a + b, 0);
      $("statOnScreen").textContent = onScreen;
      const breakdown = Object.entries(s.current_counts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([k, v]) => `${k}: ${v}`)
        .join(" · ");
      $("statOnScreenBreak").textContent = breakdown || "—";

      const total = Object.values(s.total_counts).reduce((a, b) => a + b, 0);
      $("statTotal").textContent = fmt(total);
      $("statUnique").textContent = Object.keys(s.total_counts).length;

      renderTopClasses(s.top_classes);
      renderCurrent(s.current_counts);
      drawSpark(s.recent_fps);
    } catch (err) {
      /* transient — the page will try again next tick */
    }
  }

  $("resetBtn").addEventListener("click", async () => {
    await fetch("/api/reset", { method: "POST" });
    tick();
  });

  tick();
  setInterval(tick, 500);
})();
