#!/usr/bin/env python
"""Build a static HTML review page for per-image real-degradation labels.

The page is prefilled with the assistant draft.  Reviewers accept or correct
each image; the export button writes a JSON file that is merged only after all
images are confirmed.  Labels are never derived automatically at merge time.
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maritime_robust.common import ROOT as PROJECT  # noqa: E402

TYPES = [("cloud", "厚云"), ("haze", "雾/霾"), ("shadow", "阴影"),
         ("glare", "强反光/过曝"), ("blur", "模糊/低分辨率")]

TEMPLATE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>真实退化标注 · __DATASET__</title>
<style>
:root { --bg:#111; --card:#1c1c1c; --line:#333; --text:#eee; --muted:#999; --ok:#2e7d32; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui, sans-serif; }
header { position:sticky; top:0; z-index:5; background:#181818; border-bottom:1px solid var(--line); padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
header h1 { font-size:15px; margin:0 12px 0 0; font-weight:600; }
header .count { color:var(--muted); }
button, select, input[type=text] { background:#262626; color:var(--text); border:1px solid #444; border-radius:5px; padding:4px 8px; font:inherit; }
button:hover { border-color:#777; }
button.primary { background:#1e5aa8; border-color:#2b6fc4; }
button.confirmed { background:var(--ok); border-color:#3a9a40; }
main { padding:16px; display:grid; grid-template-columns:repeat(auto-fill, minmax(300px, 1fr)); gap:14px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
.card .top { display:flex; justify-content:space-between; gap:8px; padding:8px 10px; font-size:13px; }
.card .top .name { color:var(--muted); }
.card img { width:100%; aspect-ratio:1/1; object-fit:contain; background:#0a0a0a; display:block; cursor:zoom-in; }
.card .body { padding:8px 10px 10px; display:grid; gap:8px; }
.types { display:flex; flex-wrap:wrap; gap:6px 10px; }
.types label { display:flex; align-items:center; gap:3px; font-size:13px; }
.row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.row label { color:var(--muted); font-size:12px; }
.note { width:100%; }
.card .actions { display:flex; justify-content:space-between; align-items:center; gap:8px; }
.badge { font-size:11px; padding:1px 6px; border-radius:10px; background:#333; color:#ddd; }
.badge.sev-medium { background:#8a5a00; }
.badge.sev-mild { background:#4a4a20; }
.badge.uncertain { background:#6a3b00; }
.hidden { display:none !important; }
#exportBox { position:fixed; inset:5% 5%; background:#151515; border:1px solid #555; border-radius:8px; padding:12px; z-index:20; display:none; flex-direction:column; gap:8px; }
#exportBox textarea { flex:1; width:100%; background:#0b0b0b; color:#ddd; border:1px solid #444; font:12px/1.4 monospace; }
</style>
</head>
<body>
<header>
  <h1>真实退化标注 · __DATASET__</h1>
  <span class="count" id="counts"></span>
  <select id="filter">
    <option value="all">全部</option>
    <option value="unconfirmed">未确认</option>
    <option value="confirmed">已确认</option>
    <option value="degraded">有退化</option>
    <option value="clear">清晰且已判定</option>
    <option value="uncertain">不确定</option>
  </select>
  <button id="confirmVisible">确认当前筛选全部</button>
  <button class="primary" id="export">导出 JSON</button>
</header>
<main id="grid"></main>
<div id="exportBox">
  <div class="row"><strong>导出结果</strong><span class="count">保存为 labels___DATASET__.json，合并前会校验全部确认</span></div>
  <textarea id="exportText" spellcheck="false"></textarea>
  <div class="row"><button id="download">下载文件</button><button id="closeExport">关闭</button></div>
</div>
<script>
const DATASET = "__DATASET__";
const DATA = __DATA__;
const TYPES = __TYPES__;
const KEY = "degradation-review-" + DATASET + "-v1";
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { saved = {}; }
const state = new Map();
for (const d of DATA) {
  const s = saved[d.review_index] || {};
  state.set(d.review_index, Object.assign({}, d, {
    degradation_types: s.degradation_types !== undefined ? s.degradation_types : d.degradation_types,
    severity: s.severity !== undefined ? s.severity : d.severity,
    clutter: s.clutter !== undefined ? s.clutter : d.clutter,
    uncertain: s.uncertain !== undefined ? s.uncertain : d.uncertain,
    note: s.note !== undefined ? s.note : d.note,
    confirmed: s.confirmed !== undefined ? s.confirmed : false
  }));
}
const grid = document.getElementById("grid");
const filter = document.getElementById("filter");
const counts = document.getElementById("counts");

function persist() { localStorage.setItem(KEY, JSON.stringify(Object.fromEntries(state))); }
function visible(d) {
  const f = filter.value;
  if (f === "unconfirmed") return !d.confirmed;
  if (f === "confirmed") return d.confirmed;
  if (f === "degraded") return d.degradation_types.length > 0;
  if (f === "clear") return d.degradation_types.length === 0 && !d.uncertain;
  if (f === "uncertain") return d.uncertain;
  return true;
}
function summary() {
  const all = [...state.values()];
  const degraded = all.filter(d => d.degradation_types.length > 0).length;
  counts.textContent = `共 ${all.length} · 已确认 ${all.filter(d => d.confirmed).length} · 有退化 ${degraded} · 不确定 ${all.filter(d => d.uncertain).length}`;
}
function badge(text, cls) { const s = document.createElement("span"); s.className = "badge " + (cls || ""); s.textContent = text; return s; }
function makeCard(d) {
  const card = document.createElement("section"); card.className = "card";
  const top = document.createElement("div"); top.className = "top";
  const left = document.createElement("span"); left.textContent = String(d.review_index).padStart(3, "0");
  const name = document.createElement("span"); name.className = "name"; name.textContent = d.label_name + " · " + d.width + "×" + d.height;
  top.append(left, name);
  if (d.uncertain) top.append(badge("AI 不确定", "uncertain"));
  const img = document.createElement("img"); img.loading = "lazy"; img.src = d.image; img.alt = d.sample_id;
  img.onclick = () => window.open(d.image, "_blank");
  const body = document.createElement("div"); body.className = "body";
  const types = document.createElement("div"); types.className = "types";
  for (const [key, label] of TYPES) {
    const wrap = document.createElement("label");
    const box = document.createElement("input"); box.type = "checkbox"; box.checked = d.degradation_types.includes(key);
    box.onchange = () => {
      const set = new Set(d.degradation_types);
      if (box.checked) set.add(key); else set.delete(key);
      d.degradation_types = [...set]; d.confirmed = false; persist(); render();
    };
    wrap.append(box, document.createTextNode(label)); types.append(wrap);
  }
  const row = document.createElement("div"); row.className = "row";
  const sevLabel = document.createElement("label"); sevLabel.textContent = "强度";
  const sev = document.createElement("select");
  for (const [v, t] of [["none","无"],["mild","轻"],["medium","中"],["severe","重"],["unable","无法判断"]]) {
    const o = document.createElement("option"); o.value = v; o.textContent = t; if (d.severity === v) o.selected = true; sev.append(o);
  }
  sev.onchange = () => { d.severity = sev.value; d.confirmed = false; persist(); render(); };
  const clutterWrap = document.createElement("label");
  const clutter = document.createElement("input"); clutter.type = "checkbox"; clutter.checked = d.clutter;
  clutter.onchange = () => { d.clutter = clutter.checked; d.confirmed = false; persist(); render(); };
  clutterWrap.append(clutter, document.createTextNode("场景杂乱"));
  const uncWrap = document.createElement("label");
  const unc = document.createElement("input"); unc.type = "checkbox"; unc.checked = d.uncertain;
  unc.onchange = () => { d.uncertain = unc.checked; persist(); render(); };
  uncWrap.append(unc, document.createTextNode("不确定"));
  row.append(sevLabel, sev, clutterWrap, uncWrap);
  const note = document.createElement("input"); note.type = "text"; note.className = "note"; note.placeholder = "备注"; note.value = d.note || "";
  note.onchange = () => { d.note = note.value; d.confirmed = false; persist(); };
  const actions = document.createElement("div"); actions.className = "actions";
  const hint = document.createElement("span"); hint.className = "name"; hint.textContent = d.sample_id;
  const ok = document.createElement("button"); ok.textContent = d.confirmed ? "已确认" : "确认"; ok.className = d.confirmed ? "confirmed" : "";
  ok.onclick = () => { d.confirmed = !d.confirmed; persist(); render(); };
  actions.append(hint, ok);
  body.append(types, row, note, actions);
  card.append(top, img, body);
  return card;
}
function render() {
  grid.innerHTML = "";
  const frag = document.createDocumentFragment();
  let shown = 0;
  for (const d of state.values()) if (visible(d)) { frag.append(makeCard(d)); shown++; }
  if (!shown) { const p = document.createElement("p"); p.textContent = "当前筛选为空"; frag.append(p); }
  grid.append(frag); summary();
}
filter.onchange = render;
document.getElementById("confirmVisible").onclick = () => {
  for (const d of state.values()) if (visible(d)) d.confirmed = true;
  persist(); render();
};
document.getElementById("export").onclick = () => {
  const out = [...state.values()].map(d => ({
    review_index: d.review_index, sample_id: d.sample_id, label_name: d.label_name,
    degradation_types: d.degradation_types, severity: d.severity, clutter: d.clutter,
    uncertain: d.uncertain, note: d.note, confirmed: d.confirmed,
    draft_types: d.draft_types, draft_severity: d.draft_severity
  }));
  const text = JSON.stringify(out, null, 2);
  document.getElementById("exportText").value = text;
  document.getElementById("exportBox").style.display = "flex";
  document.getElementById("download").onclick = () => {
    const blob = new Blob([text], {type: "application/json"});
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
    a.download = "labels_" + DATASET + ".json"; a.click(); URL.revokeObjectURL(a.href);
  };
};
document.getElementById("closeExport").onclick = () => { document.getElementById("exportBox").style.display = "none"; };
render();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("fgsc23_val", "fgsrcs_dev"), required=True)
    args = parser.parse_args()
    root = PROJECT / f"reviews/real_degradation/{args.dataset}"
    index = json.loads((root / "index.json").read_text())
    draft = {row["review_index"]: row for row in
             json.loads((root / "draft_labels.json").read_text())}
    images = root / "images"
    images.mkdir(exist_ok=True)
    data = []
    for row in index:
        label = draft[row["review_index"]]
        source = PROJECT / row["path"]
        with Image.open(source) as f:
            image = f.convert("RGB")
        image = ImageOps.contain(image, (384, 384), Image.Resampling.BICUBIC)
        name = f"{row['review_index']:03d}.jpg"
        image.save(images / name, quality=88)
        data.append({**row, "image": f"images/{name}",
                     "degradation_types": label["degradation_types"],
                     "severity": label["severity"], "clutter": label["clutter"],
                     "uncertain": label["uncertain"], "note": label["note"],
                     "draft_types": label["degradation_types"],
                     "draft_severity": label["severity"]})
    html = (TEMPLATE.replace("__DATASET__", args.dataset)
            .replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__TYPES__", json.dumps(TYPES, ensure_ascii=False)))
    (root / "index.html").write_text(html)
    print(json.dumps({"dataset": args.dataset, "images": len(data),
                      "page": str((root / "index.html").relative_to(PROJECT))}))


if __name__ == "__main__":
    main()
