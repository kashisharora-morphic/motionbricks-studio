// Canvas timeline.
//   Ruler (top strip) ... click / drag to scrub. Scrubbing only starts here.
//   Track (below) ....... click a keyframe to select it (Shift toggles), drag selected
//                         keyframes to retime them together, drag on empty space to
//                         box-select (Shift adds), click empty space to clear.
//                         Clicking a gap between two keyframes selects that span (its settings
//                         show in the viewport, and it gets a green outline here).
//   Right-click ......... context menu (copy / cut / paste / delete), via opts.onContext.
const C = {
  ruler: '#6a717d', rulerText: '#7d8490', rulerBg: 'rgba(255,255,255,0.035)', track: '#20232a',
  key: '#ff7a1a', keyHot: '#ffc08a', keySel: '#ffffff', gen: '#2f6fe0', stale: '#f2c230', missing: '#3a3f48', head: '#ffffff',
  box: 'rgba(242,194,48,0.12)', boxLine: 'rgba(242,194,48,0.8)', bd: '#35c48d', jump: 'rgba(90,209,106,0.75)',
};
const PAD = 16;
const RULER_H = 22;   // the scrub strip
const KEY_HIT = 9;    // px either side of a diamond
const ROW_HIT = 16;   // px above/below the key row that still counts as on a key

export class Timeline {
  constructor(canvas, opts) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.opts = opts;
    this.drag = null;
    this.hoverKey = null;
    canvas.addEventListener('pointerdown', (e) => this.down(e));
    canvas.addEventListener('pointermove', (e) => this.move(e));
    canvas.addEventListener('pointerup', (e) => this.up(e));
    canvas.addEventListener('pointerleave', () => { this.hoverKey = null; this.setHoverGap(null); });
    canvas.addEventListener('contextmenu', (e) => this.context(e));
  }

  resize() {
    const r = this.canvas.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio, 2);
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    this.w = r.width; this.h = r.height; this.dpr = dpr;
  }

  rowY() { return Math.round(RULER_H + (this.h - RULER_H) * 0.5); }
  x(f, len) { return PAD + (f / Math.max(1, len - 1)) * (this.w - 2 * PAD); }
  frameAt(px, len) { return Math.max(0, Math.min(len - 1, Math.round(((px - PAD) / (this.w - 2 * PAD)) * (len - 1)))); }
  local(e) { const r = this.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }

  keyNear(px, py, st) {
    if (py < RULER_H || Math.abs(py - this.rowY()) > ROW_HIT) return null;
    let best = null, bd = KEY_HIT;
    for (const k of st.keys) { const d = Math.abs(this.x(k.frame, st.length) - px); if (d < bd) { bd = d; best = k; } }
    return best;
  }

  /** The span (its first keyframe) whose gap on the key row is under the cursor, or null. */
  gapAt(px, py, st) {
    if (py < RULER_H || Math.abs(py - this.rowY()) > ROW_HIT) return null;
    const f = ((px - PAD) / (this.w - 2 * PAD)) * (st.length - 1);
    for (let i = 0; i + 1 < st.keys.length; i++) if (st.keys[i].frame < f && f < st.keys[i + 1].frame) return st.keys[i];
    return null;
  }
  setHoverGap(k) {
    const id = k ? k.id : null;
    if (id === this.hoverGap) return;
    this.hoverGap = id;
    this.opts.onHoverSpan?.(id);
  }

  down(e) {
    if (e.button !== 0) return;
    const st = this.opts.getState();
    const [px, py] = this.local(e);
    try { this.canvas.setPointerCapture(e.pointerId); } catch {} // keep the drag if the pointer leaves
    this.opts.onPointerDown?.();
    if (py < RULER_H) { // scrubbing starts only on the ruler
      this.drag = { type: 'scrub' };
      this.opts.onScrub(this.frameAt(px, st.length));
      return;
    }
    const key = this.keyNear(px, py, st);
    if (key) {
      let sel = new Set(st.selected);
      if (e.shiftKey) { // toggle in/out of the selection
        if (sel.has(key.id)) sel.delete(key.id); else sel.add(key.id);
        this.opts.onSelect(sel);
        this.drag = { type: 'none' };
        return;
      }
      if (!sel.has(key.id)) { sel = new Set([key.id]); this.opts.onSelect(sel); }
      this.drag = { type: 'keys', key, startFrame: this.frameAt(px, st.length), delta: 0, moved: false };
      this.opts.onMoveStart?.(sel);
      return;
    }
    this.drag = { type: 'box', x0: px, y0: py, x1: px, y1: py, base: e.shiftKey ? new Set(st.selected) : new Set(), moved: false };
  }

  move(e) {
    const st = this.opts.getState();
    const [px, py] = this.local(e);
    if (!this.drag) {
      this.hoverKey = this.keyNear(px, py, st);
      this.setHoverGap(this.hoverKey ? null : this.gapAt(px, py, st));
      this.canvas.style.cursor = py < RULER_H ? 'col-resize' : this.hoverKey ? 'grab' : this.hoverGap != null ? 'pointer' : 'crosshair';
      return;
    }
    this.setHoverGap(null);
    const d = this.drag;
    if (d.type === 'scrub') this.opts.onScrub(this.frameAt(px, st.length));
    else if (d.type === 'keys') {
      const delta = this.frameAt(px, st.length) - d.startFrame;
      if (delta !== d.delta) { d.delta = delta; d.moved = true; this.opts.onMoveKeys?.(delta); }
      this.canvas.style.cursor = 'grabbing';
    } else if (d.type === 'box') {
      d.x1 = px; d.y1 = py;
      if (Math.abs(d.x1 - d.x0) > 3 || Math.abs(d.y1 - d.y0) > 3) d.moved = true;
      if (d.moved) {
        const lo = Math.min(d.x0, d.x1), hi = Math.max(d.x0, d.x1);
        const sel = new Set(d.base);
        for (const k of st.keys) { const kx = this.x(k.frame, st.length); if (kx >= lo - 2 && kx <= hi + 2) sel.add(k.id); }
        this.opts.onSelect(sel);
      }
    }
  }

  up() {
    const d = this.drag;
    this.drag = null;
    if (!d) return;
    if (d.type === 'scrub') this.opts.onScrubEnd?.();
    else if (d.type === 'keys') this.opts.onMoveEnd?.(d.moved);
    else if (d.type === 'box' && !d.moved) { // a plain click on the track: clear keys, pick the gap's span
      this.opts.onSelect(new Set());
      this.opts.onSpanClick?.(this.frameAt(d.x0, this.opts.getState().length), d.x0);
    }
  }

  context(e) {
    e.preventDefault();
    const st = this.opts.getState();
    const [px, py] = this.local(e);
    const key = this.keyNear(px, py, st);
    if (key && !st.selected.has(key.id)) this.opts.onSelect(new Set([key.id]));
    this.opts.onContext?.(e.clientX, e.clientY, this.frameAt(px, st.length));
  }

  draw() {
    const st = this.opts.getState();
    const { ctx, w, h, dpr } = this;
    if (!w) return;
    const len = st.length;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // Ruler strip (scrub here): pick a label step that keeps labels at least ~48px apart.
    ctx.fillStyle = C.rulerBg; ctx.fillRect(0, 0, w, RULER_H);
    const pxPerFrame = (w - 2 * PAD) / Math.max(1, len - 1);
    const step = [1, 2, 5, 10, 15, 30, 60, 150, 300, 600].find((s) => s * pxPerFrame >= 48) || 600;
    const minor = [1, 5, 10, 30, 60].find((s) => s * pxPerFrame >= 6) || 60;
    ctx.font = '10px Jost, Futura, "Avenir Next", ui-sans-serif, sans-serif';
    ctx.textBaseline = 'top';
    for (let f = 0; f < len; f += minor) {
      const x = Math.round(this.x(f, len)) + 0.5;
      const major = f % step === 0;
      ctx.strokeStyle = C.ruler; ctx.globalAlpha = major ? 0.9 : 0.35;
      ctx.beginPath(); ctx.moveTo(x, RULER_H - (major ? 9 : 5)); ctx.lineTo(x, RULER_H); ctx.stroke();
      if (major) { ctx.fillStyle = C.rulerText; ctx.globalAlpha = 1; ctx.fillText(String(f), x + 3, 3); }
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.beginPath(); ctx.moveTo(0, RULER_H + 0.5); ctx.lineTo(w, RULER_H + 0.5); ctx.stroke();

    const midY = this.rowY();
    ctx.fillStyle = C.track;
    ctx.fillRect(PAD - 8, midY - 10, w - 2 * PAD + 16, 20);

    // Jump spans: an arc over the track between their keyframes.
    st.keys.forEach((k, i) => {
      const nx = st.keys[i + 1];
      if (!nx || st.spanOf(k).kind !== 'jump') return;
      const x0 = this.x(k.frame, len), x1 = this.x(nx.frame, len);
      ctx.strokeStyle = C.jump; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x0, midY - 6);
      ctx.quadraticCurveTo((x0 + x1) / 2, midY - 6 - Math.min(26, 10 + st.spanOf(k).height * 10) * 2, x1, midY - 6);
      ctx.stroke(); ctx.lineWidth = 1;
    });

    // Segments between poses (keyframes and breakdowns): what the AI generates, piece by piece.
    st.poses.forEach((k, i) => {
      if (i + 1 >= st.poses.length) return;
      const x0 = this.x(k.frame, len), x1 = this.x(st.poses[i + 1].frame, len);
      const seg = st.ok[i] ? st.cache.get(st.sigs[i]) : null;
      let color = C.missing;
      if (seg) color = C.gen;
      else if (i < st.lastRunLength) color = C.stale; // was generated, keys changed since
      if (st.generating && st.firstBad >= 0 && i >= st.firstBad) {
        const pulse = 0.35 + 0.25 * Math.sin(performance.now() / 150);
        ctx.globalAlpha = pulse; color = C.gen;
      }
      ctx.fillStyle = color;
      ctx.fillRect(x0, midY - 4, x1 - x0, 8);
      ctx.globalAlpha = 1;
      if (seg && seg.heldFrames > 0) { // model finished early: it holds the target pose
        const xh = this.x(st.poses[i + 1].frame - seg.heldFrames, len);
        ctx.fillStyle = 'rgba(0,0,0,.45)';
        ctx.fillRect(xh, midY - 4, x1 - xh, 8);
      }
    });

    // Selected span: its stretch of track filled green.
    const selI = st.spanSel == null ? -1 : st.keys.findIndex((k) => k.id === st.spanSel);
    const selRange = selI >= 0 && selI + 1 < st.keys.length ? [st.keys[selI].frame, st.keys[selI + 1].frame] : null;
    if (selRange) {
      const x0 = this.x(selRange[0], len), x1 = this.x(selRange[1], len);
      ctx.fillStyle = C.bd;
      ctx.fillRect(x0, midY - 4, x1 - x0, 8);
    }
    // Hovered gap (clickable): a fainter green
    const hovI = this.hoverGap == null || this.hoverGap === st.spanSel ? -1 : st.keys.findIndex((k) => k.id === this.hoverGap);
    if (hovI >= 0 && hovI + 1 < st.keys.length) {
      const x0 = this.x(st.keys[hovI].frame, len), x1 = this.x(st.keys[hovI + 1].frame, len);
      ctx.fillStyle = 'rgba(53,196,141,0.45)';
      ctx.fillRect(x0, midY - 4, x1 - x0, 8);
    }

    // Breakdowns: small green diamonds on the track (computed, not editable here).
    ctx.strokeStyle = 'rgba(13,15,19,0.9)'; ctx.lineWidth = 1.5;
    for (const b of st.poses) {
      if (!b.bd) continue;
      const x = this.x(b.frame, len), s = 4.5;
      ctx.fillStyle = selRange && b.frame > selRange[0] && b.frame < selRange[1] ? '#e9fff6' : C.bd; // light on the green fill
      ctx.beginPath(); ctx.moveTo(x, midY - s); ctx.lineTo(x + s, midY); ctx.lineTo(x, midY + s); ctx.lineTo(x - s, midY); ctx.closePath();
      ctx.fill(); ctx.stroke();
    }
    ctx.lineWidth = 1;

    // Box selection
    const d = this.drag;
    if (d && d.type === 'box' && d.moved) {
      const bx = Math.min(d.x0, d.x1), by = Math.max(RULER_H + 1, Math.min(d.y0, d.y1));
      const bw = Math.abs(d.x1 - d.x0), bh = Math.min(h - 1, Math.max(d.y0, d.y1)) - by;
      ctx.fillStyle = C.box; ctx.fillRect(bx, by, bw, bh);
      ctx.strokeStyle = C.boxLine; ctx.lineWidth = 1; ctx.strokeRect(bx + 0.5, by + 0.5, bw, bh);
    }

    // Keys (selected ones get a white rim)
    for (const k of st.keys) {
      const x = this.x(k.frame, len);
      const sel = st.selected.has(k.id);
      const hot = this.hoverKey === k || (d && d.type === 'keys' && sel);
      const s = hot || sel ? 8 : 7;
      ctx.fillStyle = hot ? C.keyHot : C.key;
      ctx.strokeStyle = sel ? C.keySel : 'rgba(13,15,19,0.9)'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x, midY - s); ctx.lineTo(x + s, midY); ctx.lineTo(x, midY + s); ctx.lineTo(x - s, midY); ctx.closePath();
      ctx.fill(); ctx.stroke();
      if (this.hoverKey === k && !d) {
        const label = `${this.opts.label(k)} · ${k.frame}`;
        ctx.font = '11px Jost, Futura, "Avenir Next", ui-sans-serif, sans-serif';
        const tw = ctx.measureText(label).width;
        const lx = Math.min(Math.max(x - tw / 2 - 6, 2), w - tw - 14);
        ctx.fillStyle = 'rgba(13,15,19,.96)'; ctx.fillRect(lx, midY - 33, tw + 12, 18);
        ctx.fillStyle = '#ffd2ac'; ctx.textBaseline = 'middle'; ctx.fillText(label, lx + 6, midY - 24); ctx.textBaseline = 'top';
      }
    }

    // Playhead (the handle lives on the ruler)
    const px = Math.round(this.x(st.frame, len)) + 0.5;
    ctx.strokeStyle = C.head; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(px, 4); ctx.lineTo(px, h); ctx.stroke();
    ctx.fillStyle = C.head;
    ctx.beginPath(); ctx.moveTo(px - 6, 2); ctx.lineTo(px + 6, 2); ctx.lineTo(px + 6, 10); ctx.lineTo(px, 16); ctx.lineTo(px - 6, 10); ctx.closePath(); ctx.fill();
    ctx.lineWidth = 1;
  }
}
