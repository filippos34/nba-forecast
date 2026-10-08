// Market-comparison rendering shared by the public Today page and the private dashboard.
// Inputs: model.json (model side), polymarket.json (midpoints), consensus.json (median of fresh books).

export const STALE_MIN = 90;

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
export const pct = (x, d = 1) => (x === null || x === undefined ? "—" : `${(x * 100).toFixed(d)}%`);
export const pp = (x) => (x === null || x === undefined ? "—" : `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x * 100).toFixed(1)} pp`);
export const etTime = (iso) =>
  new Date(iso).toLocaleTimeString("en-US", { timeZone: "America/New_York", hour: "numeric", minute: "2-digit" }) + " ET";

export function age(iso, now = Date.now()) {
  if (!iso) return null;
  return (now - new Date(iso).getTime()) / 60000;
}

function asOf(iso, now) {
  if (!iso) return { text: "", stale: false };
  const m = age(iso, now);
  const rel = m < 1 ? "just now" : m < 90 ? `${Math.round(m)} min ago` : m < 48 * 60 ? `${Math.round(m / 60)} h ago` : `${Math.round(m / 1440)} d ago`;
  return { text: `as of ${etTime(iso)} · ${rel}`, stale: m > STALE_MIN };
}

function cell(label, value, sub, iso, now, extraClass = "") {
  const a = asOf(iso, now);
  return `<div class="cmp-cell ${a.stale ? "stale" : ""} ${extraClass}">
    <div class="cmp-label">${label}</div>
    <div class="cmp-value">${value}</div>
    ${sub ? `<div class="cmp-sub">${sub}</div>` : ""}
    <div class="cmp-asof">${a.text}${a.stale ? " · stale" : ""}</div></div>`;
}

export function gameCard(g, poly, cons, now = Date.now(), extra = "") {
  const pm = poly?.games?.[String(g.game_id)] ?? null;
  const cs = cons?.games?.[String(g.game_id)] ?? null;
  const gap = cs ? g.model_p_home - cs.p_home : null;
  const tags = [
    g.early_estimate ? `<span class="tag" title="Until both teams have played ${esc(g.early_label_games ?? 10)} games">early-season estimate</span>` : "",
    g.pre_report ? `<span class="tag" title="No official injury report covers this game yet">pre-injury-report</span>` : "",
  ].join(" ");
  const inj = (g.injuries || []).map((x) => `<tr><td>${esc(x.team)} ${esc(x.player)}</td><td>${esc(x.status)}</td><td>${pct(x.p_plays, 0)}</td><td>${esc(x.proj_min)}</td></tr>`).join("");
  return `<article class="card game">
    <div class="game-head"><span class="when">${etTime(g.tip_utc)}${g.neutral ? " · neutral site" : ""}</span>
      <span class="matchup"><b>${esc(g.away)}</b> at <b>${esc(g.home)}</b></span><span class="tags">${tags}</span></div>
    <div class="cmp-grid">
      ${cell(`Model · ${esc(g.home)}`, pct(g.model_p_home), `margin ${g.model_spread_home <= 0 ? esc(g.home) + " by " + Math.abs(g.model_spread_home).toFixed(1) : esc(g.away) + " by " + g.model_spread_home.toFixed(1)}`, g.model_as_of, now, "model")}
      ${cell("Polymarket", pm ? pct(pm.p_home) : "not listed", pm ? `midpoint · bid ${pm.bid.toFixed(2)} / ask ${pm.ask.toFixed(2)} (${pm.spread_pp.toFixed(1)} pp)` : "", pm?.as_of, now)}
      ${cell("Market consensus", cs ? pct(cs.p_home) : "—", cs ? `median of ${cs.n_books} books` : `needs ≥ ${cons?.min_books ?? 2} fresh books`, cs?.as_of, now)}
      ${cell("Gap", cs ? pp(gap) : "—", "model − consensus", cs?.as_of, now, "gap")}
    </div>
    ${extra}
    <details><summary>Injury statuses used (${(g.injuries || []).length})</summary>
      ${inj ? `<table class="small"><thead><tr><th>Player</th><th>Status</th><th>P(plays)</th><th>Proj. min</th></tr></thead><tbody>${inj}</tbody></table>`
            : `<p class="small muted">${g.pre_report ? "No official injury report yet. Every rostered player is weighted by the average pre-report chance of playing; the forecast updates when the report is out." : "No listed players with projected minutes."}</p>`}
    </details></article>`;
}

export async function loadLive(base, names = ["model", "polymarket", "consensus"]) {
  const out = {};
  await Promise.all(names.map(async (n) => {
    try {
      const r = await fetch(`${base}/data/live/${n}.json?t=${Date.now()}`, { cache: "no-store" });
      out[n] = r.ok ? await r.json() : null;
    } catch { out[n] = null; }
  }));
  return out;
}

export function sparkline(series, colors, w = 220, h = 44) {
  // series: {venue: [[iso, p], ...]} → small multi-line SVG on one 0–1 axis band (auto-zoomed)
  const pts = Object.values(series).flat();
  if (!pts.length) return `<span class="muted small">no history</span>`;
  const ts = pts.map((p) => new Date(p[0]).getTime()), ps = pts.map((p) => p[1]);
  const t0 = Math.min(...ts), t1 = Math.max(...ts) || t0 + 1, lo = Math.min(...ps) - 0.01, hi = Math.max(...ps) + 0.01;
  const X = (t) => 2 + ((t - t0) / Math.max(t1 - t0, 1)) * (w - 4), Y = (p) => h - 2 - ((p - lo) / (hi - lo)) * (h - 4);
  const lines = Object.entries(series).map(([v, s]) =>
    `<path d="${s.map((p, i) => `${i ? "L" : "M"}${X(new Date(p[0]).getTime()).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("")}" fill="none" stroke="${colors[v] || "var(--ink-3)"}" stroke-width="2"><title>${esc(v)}</title></path>`).join("");
  return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="price moves">${lines}</svg>
    <span class="small muted">${pct(lo + 0.01, 0)}–${pct(hi - 0.01, 0)}</span>`;
}
