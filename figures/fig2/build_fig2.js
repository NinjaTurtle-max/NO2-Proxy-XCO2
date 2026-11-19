// Fig. 2 — model architecture and comparison design, 33.6 × 25.2 cm (4:3), PptxGenJS. Built-in overlap check (text × line, text × text, canvas): the build stops on any hit.
// Image inputs: figures/out/fig2/*.png (thumbs.py · thumbs2.py · thumb_target_train.py · tiles_final.py).
// Run: cd figures/out/fig2 && node ../../fig2/build_fig2.js → fig2.pptx (export to PNG from PowerPoint for the poster)
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
const W = 13.228, H = 9.921;   // 33.6 × 25.2 cm
pres.defineLayout({ name: "FIG2_43", width: W, height: H });
pres.layout = "FIG2_43";
const F = process.env.FIG_FONT || "Pretendard";
const Z = { L: 24, M: 19, S: 16 };
const C = { ink: "1F2A37", sub: "4B5563", line: "6B7280", mod: "EEF1F5", no2: "C2410C", wind: "1D5F99", windL: "E3EFF9", windT: "D6E6F5", xco2: "5A2A8F", red: "C0261A", rule: "D1D5DB", frame: "9AA3AD" };
const s = pres.addSlide(); s.background = { color: "FFFFFF" };
const TXT = [], SEG = [];

const T = (t, x, y, w, h, o = {}) => { TXT.push({ x, y, w, h, t: (Array.isArray(t) ? t.map(r => r.text).join("") : t) }); return s.addText(t, { x, y, w, h, fontFace: F, fontSize: o.size || Z.S, color: o.color || C.ink, bold: !!o.bold, align: o.align || "center", valign: o.valign || "top", margin: 0, isTextBox: true }); };
const IMG = (path, x, y, w, h, frame = true) => s.addImage({ path, x, y, w, h, ...(frame ? { line: { color: C.frame, width: 0.75 } } : {}) });
const BOX = (x, y, w, h, fill, line, lw = 0.75) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { color: line, width: lw } });
const L = (x1, y1, x2, y2, o = {}) => { SEG.push([x1, y1, x2, y2]); return s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(Math.abs(x2 - x1), 0.001), h: Math.max(Math.abs(y2 - y1), 0.001), flipH: x2 < x1, flipV: y2 < y1, line: { color: o.color || C.line, width: o.w || 2, dashType: o.dash || "solid", ...(o.arrow === false ? {} : { endArrowType: "triangle" }) } }); };
const PL = (pts, o = {}) => { for (let i = 0; i < pts.length - 1; i++) L(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], { ...o, arrow: o.arrow !== false && i === pts.length - 2 }); };
const LETTER = (l, t, x, y) => T([{ text: l + "  ", options: { bold: true, fontSize: Z.L } }, { text: t, options: { bold: true, fontSize: Z.L, color: C.sub } }], x, y, 5.5, 0.42, { align: "left" });
const MOD = (name, sub, x, y, w, h) => { BOX(x, y, w, h, C.mod, C.mod); T(name, x, y + 0.08, w, 0.32, { size: Z.M, bold: true }); T(sub, x, y + 0.44, w, 0.28, { color: C.sub }); };
const yc = 1.25;   // top-row centre line

// ====================== a  Model architecture ======================
LETTER("a", "Model architecture", 0.15, 0.06);
IMG("th_no2_day.png", 0.25, yc - 0.6, 2.0, 1.2);
T([{ text: "TROPOMI NO₂", options: { bold: true } }, { text: "  daily", options: { color: C.sub, fontSize: Z.S } }], 0.02, yc + 0.66, 2.46, 0.3, { size: Z.M, color: C.no2 });
L(2.27, yc, 2.53, yc, { color: C.no2, w: 2.5 });
MOD("Encoder", "pixels → nodes", 2.55, yc - 0.4, 1.7, 0.8);
L(4.27, yc, 4.5, yc);
const cw = 2.8, ch = cw * 491 / 988;
IMG("latent_cube.png", 4.5, yc - ch / 2, cw, ch, false);
T("Latent state hₜ", 4.35, yc + ch / 2 + 0.05, 3.1, 0.3, { size: Z.M, bold: true });
T("121 × 201 × 8", 4.35, yc + ch / 2 + 0.35, 3.1, 0.27, { color: C.sub });
L(7.32, yc, 7.48, yc);
MOD("Decoder", "at soundings", 7.5, yc - 0.4, 1.7, 0.8);
const ux = 9.6;   // + node / g column centre
s.addShape(pres.shapes.OVAL, { x: ux - 0.18, y: yc - 0.18, w: 0.36, h: 0.36, fill: { color: "FFFFFF" }, line: { color: C.line, width: 2 } });
T("+", ux - 0.18, yc - 0.175, 0.36, 0.34, { size: Z.M, bold: true, valign: "middle" });
L(9.22, yc, ux - 0.2, yc);

// physical transport under the latent state; hourly loop; wind from the left
const ly = yc + ch / 2 + 0.66;   // loop arrow top (below the cube labels)
const px = 4.3, pw = 3.2, py = ly + 0.36, ph = 4.0;
BOX(px, py, pw, ph, C.wind, C.wind);
T("Physical transport", px, py + 0.1, pw, 0.32, { size: Z.M, bold: true, color: "FFFFFF" });
T("repeated every hour", px, py + 0.45, pw, 0.28, { color: C.windT });
[["opF_1.png", "Advection", "along wind"], ["opF_2.png", "Diffusion", "learned D"], ["opF_3.png", "Decay", "learned k"]].forEach(([f, n, d], i) => {
  const y = py + 0.86 + i * 1.02;
  IMG(f, px + 0.25, y, 0.92, 0.92, false);
  T(n, px + 1.35, y + 0.12, 1.8, 0.32, { size: Z.M, bold: true, color: "FFFFFF", align: "left" });
  T(d, px + 1.35, y + 0.48, 1.8, 0.28, { color: C.windT, align: "left" });
});
L(5.4, ly, 5.4, py - 0.02);
L(6.4, py - 0.02, 6.4, ly);
const wy = py + ph / 2;   // wind enters at the box mid-height
IMG("th_wind.png", 0.25, wy - 0.6, 2.0, 1.2);
T([{ text: "ERA5 wind", options: { bold: true } }, { text: "  hourly", options: { color: C.sub, fontSize: Z.S } }], 0.02, wy + 0.66, 2.46, 0.3, { size: Z.M, color: C.wind });
L(2.27, wy, px - 0.02, wy, { color: C.wind, w: 2.5 });

// background → g(background) → +
const gy = py - 0.05;
BOX(ux - 0.87, gy, 1.74, 0.42, C.mod, C.mod);
T("g(background)", ux - 0.87, gy + 0.05, 1.74, 0.3, { bold: true });
L(ux, gy - 0.02, ux, yc + 0.2);
const by = gy + 0.95;
IMG("th_bg_sp.png", ux - 0.63, by, 1.5, 0.9); IMG("th_bg_blh.png", ux - 0.75, by + 0.1, 1.5, 0.9); IMG("th_bg_t2m.png", ux - 0.87, by + 0.2, 1.5, 0.9);
L(ux, by - 0.02, ux, gy + 0.44);
T("Background", ux - 1.0, by + 1.15, 2.0, 0.3, { size: Z.M, bold: true });

// outputs (two-line labels so they never wrap into other elements)
const mx = 10.85, mw = 2.3, mh = mw * 0.6;
IMG("th_pred2020.png", mx, yc - mh / 2, mw, mh);
L(ux + 0.2, yc, mx - 0.02, yc);
T("Prediction", mx, yc + mh / 2 + 0.05, mw, 0.3, { size: Z.M, bold: true, color: C.xco2, align: "left" });
T("test region", mx, yc + mh / 2 + 0.35, mw, 0.27, { color: C.sub, align: "left" });
const ty = py;
IMG("th_target2020_train.png", mx, ty, mw, mh);
T("OCO-2/3", mx, ty + mh + 0.05, mw, 0.3, { size: Z.M, bold: true, color: C.xco2, align: "left" });
T("training rows", mx, ty + mh + 0.35, mw, 0.27, { color: C.sub, align: "left" });
const lx = 10.62;
PL([[mx - 0.02, ty + mh / 2], [lx, ty + mh / 2], [lx, yc + 0.2]], { color: C.xco2, w: 2, dash: "dash" });
T("Loss", lx - 0.62, (yc + gy) / 2 - 0.14, 0.52, 0.28, { bold: true, color: C.xco2, align: "right" });
const cbx = mx + 0.15, cbw = mw - 0.3, cby = ty + mh + 0.85;
IMG("cb_bar.png", cbx, cby, cbw, 0.16, false);
[[405, cbx + cbw * (405 - 404.1) / 14.4], [418, cbx + cbw * (418 - 404.1) / 14.4]].forEach(([v, x]) => { L(x, cby + 0.16, x, cby + 0.23, { w: 1, arrow: false, color: C.ink }); T(String(v), x - 0.28, cby + 0.25, 0.56, 0.28); });
T("XCO₂ (ppm)", cbx, cby + 0.55, cbw, 0.28);

// ====================== b  Comparison design ======================
const bt = 7.72;
s.addShape(pres.shapes.LINE, { x: 0.15, y: bt - 0.08, w: W - 0.3, h: 0, line: { color: C.rule, width: 1 } });
LETTER("b", "Comparison design", 0.15, bt);
IMG("th_split.png", 0.3, bt + 0.5, 1.95, 1.17);
T("Test region", 2.4, bt + 0.68, 1.9, 0.3, { size: Z.M, bold: true, color: C.red, align: "left" });
T("12 cells (5°)", 2.4, bt + 1.02, 1.9, 0.28, { color: C.sub, align: "left" });
// 비교 = 60개월 4개 모델(수송 켬 · 끔 · 크리깅 · 같은 입력 LightGBM)
const bw = 2.1, bg = (W - 0.15 - 4.35 - 4 * bw) / 3;
[["Transport on", "this model", true], ["Transport off", "no transport", false], ["Kriging", "spatiotemporal", false], ["LightGBM", "same inputs", false]].forEach(([n, d, hi], i) => {
  const x = 4.35 + i * (bw + bg);
  BOX(x, bt + 0.5, bw, 0.8, hi ? C.windL : C.mod, hi ? C.wind : C.mod, hi ? 1.75 : 0.75);
  T(n, x + 0.06, bt + 0.56, bw - 0.12, 0.3, { size: Z.M, bold: true, color: hi ? C.wind : C.ink });
  T(d, x + 0.06, bt + 0.92, bw - 0.12, 0.28, { color: C.sub });
});
T("→ 60-month training · RMSE on the same test soundings", 4.35, bt + 1.4, 8.7, 0.28, { color: C.sub });

// ---------------- overlap check ----------------
const pad = 0.015;
const segHitsRect = ([x1, y1, x2, y2], r) => {
  const rx0 = r.x - pad, ry0 = r.y - pad, rx1 = r.x + r.w + pad, ry1 = r.y + r.h + pad; let t0 = 0, t1 = 1; const dx = x2 - x1, dy = y2 - y1;
  for (const [p, q] of [[-dx, x1 - rx0], [dx, rx1 - x1], [-dy, y1 - ry0], [dy, ry1 - y1]]) {
    if (p === 0) { if (q < 0) return false; } else { const t = q / p; if (p < 0) { if (t > t1) return false; if (t > t0) t0 = t; } else { if (t < t0) return false; if (t < t1) t1 = t; } }
  }
  return true;
};
const rectsHit = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
const bad = [];
TXT.forEach(r => SEG.forEach(g => { if (segHitsRect(g, r)) bad.push(`글자 '${r.t}' ↔ 선 (${g.map(v => v.toFixed(2)).join(",")})`); }));
TXT.forEach((a, i) => TXT.slice(i + 1).forEach(b => { if (rectsHit(a, b)) bad.push(`글자 '${a.t}' ↔ 글자 '${b.t}'`); }));
TXT.forEach(r => { if (r.x < 0 || r.y < 0 || r.x + r.w > W || r.y + r.h > H) bad.push(`캔버스 밖: '${r.t}'`); });
console.log(`겹침 검사: 글상자 ${TXT.length} · 선분 ${SEG.length} · 위반 ${bad.length}`); bad.forEach(b => console.log("  " + b));

s.addNotes("Caption draft: Fig. 2. Model architecture and comparison design. (a) TROPOMI NO2 pixels are injected into a 0.25° latent grid (121 × 201 nodes, 8 channels), transported every hour by ERA5 boundary-layer wind (advection, diffusion D∇²h with learned D, decay h·exp(−kΔt) with learned k), and read out at OCO-2/3 sounding locations together with 11 background variables: XCO2 = μ + g(background) + β·NO2,latent + r(z). The top layer of the latent-state cube is illustrated with the observed TROPOMI NO2 composite of 2020-01-04; it is not a model output. Prediction and target maps come from the 12-month (2020) spatial-split run (266,148 test soundings; 1,251,332 training rows). Operator tiles are a schematic (same operators on a synthetic field). Solid grey arrows: data flow; orange and blue arrows: NO2 and wind inputs; paired arrows: hourly update; dashed purple arrow: training loss. (b) GNN transport on and off, spatiotemporal kriging and LightGBM with the same inputs are trained on 60 months and compared by RMSE on the same test soundings in the 12 held-out 5° cells (day-block bootstrap 95 % intervals).");
if (bad.length) { console.log("빌드 중단: 겹침 있음"); process.exit(1); }
pres.writeFile({ fileName: "fig2.pptx" }).then(() => console.log("saved"));
