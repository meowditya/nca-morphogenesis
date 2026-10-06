// NCA sandbox client: draws streamed grid frames, sends brush strokes and controls.
(() => {
  "use strict";
  const $ = (sel) => document.querySelector(sel);

  const grid = $("#grid");
  const gctx = grid.getContext("2d");
  const overlay = $("#overlay");
  const octx = overlay.getContext("2d");

  const DISH = { light: [251, 241, 199], dark: [29, 32, 33] };
  const WALL = [[214, 93, 14], [189, 78, 10]]; // two close oranges in a diagonal stripe
  const S = {
    ws: null, retry: 500, H: 0, W: 0,
    tool: "erase", radius: 6, strength: 0.3, view: "rgba", dish: "light", paused: false,
    frame: null, dirty: false, image: null,
    pointer: null,     // last pointer position in cell units (for the brush ring)
    stroke: null,      // {last: [x, y]} while a drag is in progress
    queue: [],         // interpolated points waiting to be sent this animation frame
  };
  window.__nca = S; // handy for debugging in the console

  // ---- connection ---------------------------------------------------------------------
  function setConn(state, text) {
    const el = $("#conn");
    el.dataset.state = state;
    el.textContent = text;
  }

  function send(msg) {
    if (S.ws && S.ws.readyState === WebSocket.OPEN) S.ws.send(JSON.stringify(msg));
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    S.ws = ws;
    setConn("connecting", "connecting");
    ws.onopen = () => { S.retry = 500; setConn("open", "live"); };
    ws.onclose = () => {
      setConn("closed", "disconnected - retrying");
      setTimeout(connect, S.retry);
      S.retry = Math.min(S.retry * 2, 5000);
    };
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") onJSON(JSON.parse(ev.data));
      else onFrame(ev.data);
    };
  }

  function onJSON(msg) {
    if (msg.type === "hello") {
      S.H = msg.height; S.W = msg.width;
      grid.width = S.W; grid.height = S.H;
      S.image = gctx.createImageData(S.W, S.H);
      const m = msg.meta || {};
      const tags = [
        ["ckpt", msg.checkpoint],
        ["trained", `${m.trained_by ? m.trained_by.split("/").pop() : "?"} · ${m.iterations ?? "?"} it`],
        ["backend", msg.backend],
        ["grid", `${S.W}×${S.H}`],
      ];
      const box = $("#tags");
      box.querySelectorAll(".tag.meta").forEach((t) => t.remove());
      for (const [k, v] of tags) {
        const t = document.createElement("span");
        t.className = "tag meta";
        t.innerHTML = `${k} · <b></b>`;
        t.querySelector("b").textContent = v;
        box.appendChild(t);
      }
      // re-send client-side settings so a reconnect restores them
      send({ type: "view", mode: S.view });
      send({ type: "speed", steps_per_frame: +$("#speed").value });
      if (S.paused) send({ type: "pause", value: true });
      resizeOverlay();
    } else if (msg.type === "stats") {
      $("#sStep").textContent = msg.step.toLocaleString();
      $("#sFps").textContent = msg.fps.toFixed(1);
      $("#sSps").textContent = msg.steps_per_sec.toFixed(0);
      $("#sMs").textContent = msg.ms_per_step.toFixed(2);
      $("#sLive").textContent = msg.mature_cells.toLocaleString();
    } else if (msg.type === "error") {
      console.warn("server:", msg.message);
    }
  }

  // ---- frames -----------------------------------------------------------------------
  function onFrame(buf) {
    const dv = new DataView(buf);
    const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
    if (magic !== "NCA1") return;
    const H = dv.getUint16(8, true), W = dv.getUint16(10, true);
    if (H !== S.H || W !== S.W) return;
    S.frame = {
      step: dv.getUint32(4, true),
      rgba: new Uint8Array(buf, 16, H * W * 4),
      walls: new Uint8Array(buf, 16 + H * W * 4),
    };
    S.dirty = true;
  }

  function draw() {
    const f = S.frame;
    if (!f || !S.image) return;
    const out = S.image.data, src = f.rgba, walls = f.walls, W = S.W, n = S.H * S.W;
    const [br, bg, bb] = DISH[S.dish];
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      if (walls[i >> 3] & (0x80 >> (i & 7))) {
        const c = WALL[((i % W) + ((i / W) | 0)) & 2 ? 1 : 0];
        out[p] = c[0]; out[p + 1] = c[1]; out[p + 2] = c[2];
      } else {
        const t = 255 - src[p + 3]; // premultiplied colour over the dish
        out[p] = src[p] + ((t * br) / 255 | 0);
        out[p + 1] = src[p + 1] + ((t * bg) / 255 | 0);
        out[p + 2] = src[p + 2] + ((t * bb) / 255 | 0);
      }
      out[p + 3] = 255;
    }
    gctx.putImageData(S.image, 0, 0);
  }

  // ---- brush overlay ----------------------------------------------------------------
  function resizeOverlay() {
    const r = overlay.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    overlay.width = Math.round(r.width * dpr);
    overlay.height = Math.round(r.height * dpr);
  }

  function drawOverlay() {
    octx.clearRect(0, 0, overlay.width, overlay.height);
    if (!S.pointer || !S.W) return;
    const sx = overlay.width / S.W, sy = overlay.height / S.H;
    const [x, y] = S.pointer;
    const r = S.tool === "seed" ? 1.5 : S.radius;
    const color = { erase: "#fb4934", noise: "#d3869b", wall: "#fe8019", unwall: "#8ec07c", seed: "#b8bb26" }[S.tool];
    octx.lineWidth = Math.max(1.5, (window.devicePixelRatio || 1) * 1.5);
    octx.strokeStyle = "rgba(29,32,33,0.55)";
    octx.beginPath(); octx.ellipse(x * sx, y * sy, r * sx + 1.5, r * sy + 1.5, 0, 0, Math.PI * 2); octx.stroke();
    octx.strokeStyle = color;
    octx.beginPath(); octx.ellipse(x * sx, y * sy, r * sx, r * sy, 0, 0, Math.PI * 2); octx.stroke();
  }

  // ---- pointer input ----------------------------------------------------------------
  function toCell(ev) {
    const r = overlay.getBoundingClientRect();
    return [((ev.clientX - r.left) / r.width) * S.W, ((ev.clientY - r.top) / r.height) * S.H];
  }

  function addPoint(pt) {
    const last = S.stroke.last;
    const spacing = Math.max(0.5, S.radius / 2);
    const dx = pt[0] - last[0], dy = pt[1] - last[1];
    const steps = Math.ceil(Math.hypot(dx, dy) / spacing);
    for (let i = 1; i <= steps; i++) S.queue.push([last[0] + (dx * i) / steps, last[1] + (dy * i) / steps]);
    S.stroke.last = pt;
  }

  overlay.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    overlay.setPointerCapture(ev.pointerId);
    const pt = toCell(ev);
    S.pointer = pt;
    if (S.tool === "seed") {
      send({ type: "stroke", tool: "seed", points: [pt] });
      return;
    }
    S.stroke = { last: pt };
    S.queue.push(pt);
  });
  overlay.addEventListener("pointermove", (ev) => {
    S.pointer = toCell(ev);
    if (S.stroke) addPoint(S.pointer);
  });
  const endStroke = () => { S.stroke = null; };
  overlay.addEventListener("pointerup", endStroke);
  overlay.addEventListener("pointercancel", endStroke);
  overlay.addEventListener("pointerleave", () => { if (!S.stroke) S.pointer = null; });

  function flushStroke() {
    if (!S.queue.length) return;
    const points = S.queue.splice(0, 512).map(([x, y]) => [+x.toFixed(2), +y.toFixed(2)]);
    send({ type: "stroke", tool: S.tool, points, radius: S.radius, strength: S.strength });
  }

  // ---- controls ---------------------------------------------------------------------
  function setTool(tool) {
    S.tool = tool;
    document.querySelectorAll(".tool").forEach((b) => {
      const on = b.dataset.tool === tool;
      b.classList.toggle("active", on);
      b.setAttribute("aria-checked", on);
    });
  }
  document.querySelectorAll(".tool").forEach((b) => b.addEventListener("click", () => setTool(b.dataset.tool)));

  function setRadius(v) {
    S.radius = Math.min(24, Math.max(1, v));
    $("#radius").value = S.radius;
    $("#radiusOut").textContent = S.radius;
  }
  $("#radius").addEventListener("input", (e) => setRadius(+e.target.value));
  $("#strength").addEventListener("input", (e) => {
    S.strength = +e.target.value;
    $("#strengthOut").textContent = S.strength.toFixed(2);
  });
  $("#speed").addEventListener("input", (e) => {
    $("#speedOut").textContent = e.target.value;
    send({ type: "speed", steps_per_frame: +e.target.value });
  });

  function setPaused(p) {
    S.paused = p;
    const b = $("#pause");
    b.classList.toggle("paused", p);
    b.querySelector("span").textContent = p ? "Resume" : "Pause";
    send({ type: "pause", value: p });
  }
  function setView(v) {
    S.view = v;
    $("#view").value = v;
    send({ type: "view", mode: v });
  }
  $("#pause").addEventListener("click", () => setPaused(!S.paused));
  $("#step").addEventListener("click", () => send({ type: "step", n: 1 }));
  $("#reset").addEventListener("click", () => send({ type: "reset" }));
  $("#damage").addEventListener("click", () => send({ type: "damage" }));
  $("#clearWalls").addEventListener("click", () => send({ type: "clear_walls" }));
  $("#clear").addEventListener("click", () => send({ type: "clear" }));
  $("#view").addEventListener("change", (e) => setView(e.target.value));
  $("#dish").addEventListener("change", (e) => { S.dish = e.target.value; S.dirty = true; });

  document.addEventListener("keydown", (ev) => {
    if (ev.target.closest("input, select, textarea") || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    // a focused button already reacts to Space/Enter by itself; handling it here too would toggle twice
    if (ev.target.closest("button, summary") && (ev.key === " " || ev.key === "Enter")) return;
    const k = ev.key.toLowerCase();
    const tools = { e: "erase", n: "noise", w: "wall", u: "unwall", s: "seed" };
    if (tools[k]) setTool(tools[k]);
    else if (k === " ") { ev.preventDefault(); setPaused(!S.paused); }
    else if (k === ".") send({ type: "step", n: 1 });
    else if (k === "r") send({ type: "reset" });
    else if (k === "d") send({ type: "damage" });
    else if (k === "[") setRadius(S.radius - 1);
    else if (k === "]") setRadius(S.radius + 1);
    else if (k === "1" || k === "2" || k === "3") setView(["rgba", "hidden", "alive"][+k - 1]);
  });

  // keep the brush ring's backing store matched to the dish size (layout, fonts, window, zoom)
  if (window.ResizeObserver) new ResizeObserver(resizeOverlay).observe(overlay);
  else window.addEventListener("resize", resizeOverlay);

  // ---- main loop --------------------------------------------------------------------
  function frame() {
    flushStroke();
    if (S.dirty) { draw(); S.dirty = false; }
    drawOverlay();
    requestAnimationFrame(frame);
  }

  resizeOverlay();
  connect();
  requestAnimationFrame(frame);
})();
