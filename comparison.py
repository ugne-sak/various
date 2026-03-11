import json
from pathlib import Path

import pandas as pd
import yaml

# ── User-configurable paths ──────────────────────────────────────────
DOCS_DIR = Path("docs/results")
MKDOCS_YML = Path("mkdocs.yml")
OUTPUT_PATH = DOCS_DIR / "comparison.md"

# Folders to include in the comparison (order is preserved)
EXPERIMENT_DIRS = [
    Path("data/experiment_20260306_1545"),
    Path("data/experiment_20260307_0912"),
    Path("data/experiment_20260308_1422"),
]

# Name of the CSV file inside each experiment folder
RESULTS_FILE = "extraction_results.csv"

# ── Field configuration ──────────────────────────────────────────────
# Column in each CSV that identifies the field name
FIELD_COL = "field"

# Columns shown with colour-coded metric styling (higher = greener)
METRIC_COLS = ["accuracy", "precision"]

# "Main" fields shown by default in the field toggle
TOP8_FIELDS = [
    "street",
    "street_name",
    "postcode",
    "city",
    "number_of_rooms",
    "usable_area",
    "price",
    "property_type",
]


def experiment_tag(folder: Path) -> str:
    return folder.name.removeprefix("experiment_")


def load_experiments() -> dict[str, pd.DataFrame]:
    return {
        experiment_tag(folder): pd.read_csv(folder / RESULTS_FILE)
        for folder in EXPERIMENT_DIRS
    }


def _infer_count_cols(experiments: dict[str, pd.DataFrame]) -> list[str]:
    """Return columns that are not FIELD_COL or METRIC_COLS, preserving first-seen order."""
    seen: dict[str, None] = {}
    for df in experiments.values():
        for c in df.columns:
            if c != FIELD_COL and c not in METRIC_COLS:
                seen[c] = None
    return list(seen)


def score_color(val: float) -> tuple[str, str]:
    if val >= 0.90:
        return "#1b5e20", "#c8e6c9"
    elif val >= 0.75:
        return "#e65100", "#ffe0b2"
    else:
        return "#b71c1c", "#ffcdd2"


def summary_cards(experiments: dict[str, pd.DataFrame]) -> str:
    tags = list(experiments.keys())
    summary_data = {
        tag: {m: round(df[m].mean(), 2) for m in METRIC_COLS}
        for tag, df in experiments.items()
    }
    summary_json = json.dumps(summary_data)
    tags_json = json.dumps(tags)
    metric_cols_json = json.dumps(METRIC_COLS)
    default_1 = tags[-2]
    default_2 = tags[-1]
    options = "".join(f'<option value="{t}">{t}</option>' for t in tags)

    return f"""
<div style="margin-bottom:32px">

<div style="display:flex;align-items:flex-start;gap:16px;margin-bottom:16px">

  <!-- Experiment 1 -->
  <div style="flex:1">
    <div style="font-size:11px;color:#999;letter-spacing:1px;margin-bottom:6px">EXPERIMENT 1</div>
    <select id="sel-1" onchange="render()"
      style="width:100%;padding:7px 10px;border-radius:6px;border:1px solid #ccc;font-family:monospace;font-size:13px;margin-bottom:12px;background:#fff">
      {options}
    </select>
    <div id="card-1"></div>
  </div>

  <!-- Experiment 2 -->
  <div style="flex:1">
    <div style="font-size:11px;color:#999;letter-spacing:1px;margin-bottom:6px">EXPERIMENT 2</div>
    <select id="sel-2" onchange="render()"
      style="width:100%;padding:7px 10px;border-radius:6px;border:1px solid #ccc;font-family:monospace;font-size:13px;margin-bottom:12px;background:#fff">
      {options}
    </select>
    <div id="card-2"></div>
  </div>

</div>

<!-- Dumbbell diff -->
<div id="diff-panel"></div>

</div>

<script>
const summaryData = {summary_json};
const summaryMetricCols = {metric_cols_json};

document.getElementById("sel-1").value = "{default_1}";
document.getElementById("sel-2").value = "{default_2}";

function colorFor(val) {{
  if (val >= 0.90) return ["#1b5e20", "#c8e6c9"];
  if (val >= 0.75) return ["#e65100", "#ffe0b2"];
  return ["#b71c1c", "#ffcdd2"];
}}

function card(tag) {{
  const d = summaryData[tag];
  const tiles = summaryMetricCols.map(m => {{
    const [text, bg] = colorFor(d[m]);
    return `<div style="flex:1;padding:10px;border-radius:4px;background:${{bg}};text-align:center">
      <div style="font-size:10px;color:#777;margin-bottom:2px">${{m.toUpperCase()}}</div>
      <div style="font-size:22px;font-weight:700;color:${{text}}">${{d[m].toFixed(2)}}</div>
    </div>`;
  }}).join("");
  return `<div style="padding:16px;border-radius:8px;background:#fafafa;border:1px solid #e0e0e0">
    <div style="display:flex;gap:8px">${{tiles}}</div>
  </div>`;
}}

function dumbbell(label, v1, v2) {{
  const MIN = 0.5, MAX = 1.0;
  const pct = v => ((v - MIN) / (MAX - MIN) * 100).toFixed(1) + "%";
  const delta = Math.round((v2 - v1) * 100) / 100;
  const color = delta > 0 ? "#2e7d32" : delta < 0 ? "#c62828" : "#9e9e9e";
  const badgeBg = delta > 0 ? "#e8f5e9" : delta < 0 ? "#ffebee" : "#f5f5f5";
  const sign = delta > 0 ? "+" : "";
  const segLeft = pct(Math.min(v1, v2));
  const segWidth = ((Math.abs(v2 - v1)) / (MAX - MIN) * 100).toFixed(1) + "%";
  return `
    <div style="display:flex;align-items:center;gap:14px;padding:10px 0;border-bottom:1px solid #f0f0f0">
      <div style="font-size:11px;color:#999;letter-spacing:1px;width:72px">${{label}}</div>
      <div style="font-size:13px;font-weight:600;color:#555;width:32px;text-align:right">${{v1.toFixed(2)}}</div>
      <div style="flex:1;position:relative;height:6px">
        <div style="position:absolute;inset:0;background:#eeeeee;border-radius:3px"></div>
        <div style="position:absolute;top:0;height:6px;left:${{segLeft}};width:${{segWidth}};background:${{color}};border-radius:3px"></div>
        <div style="position:absolute;top:-4px;left:${{pct(v1)}};transform:translateX(-50%);width:14px;height:14px;border-radius:50%;background:#fff;border:3px solid #90a4ae;box-sizing:border-box"></div>
        <div style="position:absolute;top:-4px;left:${{pct(v2)}};transform:translateX(-50%);width:14px;height:14px;border-radius:50%;background:${{color}};border:3px solid ${{color}};box-sizing:border-box"></div>
      </div>
      <div style="font-size:13px;font-weight:600;color:#555;width:32px">${{v2.toFixed(2)}}</div>
      <div style="padding:3px 10px;border-radius:10px;background:${{badgeBg}};color:${{color}};font-size:12px;font-weight:700;min-width:48px;text-align:center">${{sign}}${{delta.toFixed(2)}}</div>
    </div>`;
}}

function render() {{
  const t1 = document.getElementById("sel-1").value;
  const t2 = document.getElementById("sel-2").value;
  document.getElementById("card-1").innerHTML = card(t1);
  document.getElementById("card-2").innerHTML = card(t2);
  const d1 = summaryData[t1], d2 = summaryData[t2];
  const dumbbells = summaryMetricCols.map(m => dumbbell(m.toUpperCase(), d1[m], d2[m])).join("");
  document.getElementById("diff-panel").innerHTML = `
    <div style="padding:14px 20px;border-radius:8px;background:#fafafa;border:1px solid #e0e0e0">
      <div style="font-size:10px;color:#bbb;letter-spacing:2px;margin-bottom:6px">CHANGE (EXP 1 → EXP 2)</div>
      ${{dumbbells}}
    </div>`;
}}

render();
</script>
"""


def heatmap(experiments: dict[str, pd.DataFrame]) -> str:
    tags = list(experiments.keys())
    count_cols = _infer_count_cols(experiments)

    # Union of all fields across experiments, preserving first-seen order
    seen: dict[str, None] = {}
    for df in experiments.values():
        seen.update(dict.fromkeys(df[FIELD_COL].tolist()))
    fields = list(seen)

    # Build per-metric data: metric_data[metric][tag][field] = float | None
    metric_data: dict[str, dict] = {}
    for metric in METRIC_COLS:
        metric_data[metric] = {}
        for tag, df in experiments.items():
            rows = {row[FIELD_COL]: row for _, row in df.iterrows()}
            metric_data[metric][tag] = {
                f: round(rows[f][metric], 2) if f in rows else None for f in fields
            }

    # Build counts data: counts[tag][field] = {col: int, ...} | None
    counts_data: dict = {}
    for tag, df in experiments.items():
        rows = {row[FIELD_COL]: row for _, row in df.iterrows()}
        counts_data[tag] = {
            f: {c: int(rows[f][c]) for c in count_cols if c in rows[f].index}
            if f in rows
            else None
            for f in fields
        }

    fields_json = json.dumps(fields)
    tags_json = json.dumps(tags)
    metric_data_json = json.dumps(metric_data)
    counts_json = json.dumps(counts_data)
    top8_json = json.dumps(TOP8_FIELDS)
    metric_cols_json = json.dumps(METRIC_COLS)
    count_cols_json = json.dumps(count_cols)
    field_col_json = json.dumps(FIELD_COL)

    return f"""
<div>

<!-- Metric toggle (built by JS) -->
<div style="margin-bottom:20px">
  <span style="margin-right:8px;font-size:13px;color:#666">Metric:</span>
  <span id="metric-toggle" style="display:inline-flex"></span>
</div>

<!-- Heatmap: grey block -->
<div style="background:#f7f7f7;border-radius:8px;padding:12px 16px 16px;margin-bottom:24px">
<div style="margin-bottom:10px;display:flex;align-items:center;gap:8px;position:relative">
  <span style="font-size:11px;color:#999;letter-spacing:1px;margin-right:4px">FIELDS</span>
  <button id="hbtn-main"   onclick="setHmapFields('main')"   style="padding:4px 12px;border-radius:20px 0 0 20px;border:1px solid #546e7a;background:#546e7a;color:#fff;cursor:pointer;font-size:12px">Main</button>
  <button id="hbtn-custom" onclick="setHmapFields('custom')" style="padding:4px 12px;border-radius:0;border:1px solid #546e7a;margin-left:-1px;background:#fff;color:#546e7a;cursor:pointer;font-size:12px">Custom ▾</button>
  <button id="hbtn-all"    onclick="setHmapFields('all')"    style="padding:4px 12px;border-radius:0 20px 20px 0;border:1px solid #546e7a;margin-left:-1px;background:#fff;color:#546e7a;cursor:pointer;font-size:12px">All</button>
  <div id="hmap-dropdown" style="display:none;position:absolute;top:calc(100% + 4px);left:60px;background:#fff;border:1px solid #ddd;border-radius:6px;box-shadow:0 4px 12px rgba(0,0,0,0.1);z-index:200;min-width:180px;max-height:240px;overflow-y:auto;padding:6px 0"></div>
</div>
<div style="overflow-x:auto">
  <table id="heatmap-table" style="border-collapse:collapse;font-size:13px;width:100%">
    <thead><tr id="heatmap-header"></tr></thead>
    <tbody id="heatmap-body"></tbody>
  </table>
</div>
</div>

<!-- Detail table -->
<div id="detail-panel" style="display:none">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
    <div>
      <span style="font-size:11px;color:#999;letter-spacing:1px">EXPERIMENT&nbsp;</span>
      <span id="detail-tag" style="color:#1976d2;font-family:monospace;font-weight:700;font-size:13px"></span>
    </div>
    <button onclick="closeDetail()" style="padding:4px 10px;border-radius:4px;border:1px solid #ddd;background:#fff;color:#999;cursor:pointer;font-size:12px">✕ close details</button>
  </div>
  <div style="background:#f7f7f7;border-radius:8px;padding:12px 16px 16px">
  <div style="margin-bottom:10px;display:flex;align-items:center;gap:8px;position:relative">
    <span style="font-size:11px;color:#999;letter-spacing:1px;margin-right:4px">FIELDS</span>
    <button id="dbtn-main"   onclick="setDetailFields('main')"   style="padding:4px 12px;border-radius:20px 0 0 20px;border:1px solid #546e7a;background:#546e7a;color:#fff;cursor:pointer;font-size:12px">Main</button>
    <button id="dbtn-custom" onclick="setDetailFields('custom')" style="padding:4px 12px;border-radius:0;border:1px solid #546e7a;margin-left:-1px;background:#fff;color:#546e7a;cursor:pointer;font-size:12px">Custom ▾</button>
    <button id="dbtn-all"    onclick="setDetailFields('all')"    style="padding:4px 12px;border-radius:0 20px 20px 0;border:1px solid #546e7a;margin-left:-1px;background:#fff;color:#546e7a;cursor:pointer;font-size:12px">All</button>
    <div id="detail-dropdown" style="display:none;position:absolute;top:calc(100% + 4px);left:60px;background:#fff;border:1px solid #ddd;border-radius:6px;box-shadow:0 4px 12px rgba(0,0,0,0.1);z-index:200;min-width:180px;max-height:240px;overflow-y:auto;padding:6px 0"></div>
  </div>
  <div style="overflow-x:auto">
    <table style="border-collapse:collapse;font-size:13px;width:100%">
      <thead><tr id="detail-header"></tr></thead>
      <tbody id="detail-body"></tbody>
    </table>
  </div>
  </div>
</div>

<script>
const fieldCol   = {field_col_json};
const metricCols = {metric_cols_json};
const countCols  = {count_cols_json};
const detailCols = [fieldCol, ...countCols, ...metricCols];

const fields = {fields_json};
const top8   = {top8_json};
const tags   = {tags_json};
const data   = {metric_data_json};
const counts = {counts_json};

let expandedTag = null;
let currentMetric = metricCols[0];
let hmapCustomFields   = new Set(top8.filter(f => fields.includes(f)));
let hmapVisibleFields  = top8.filter(f => fields.includes(f));
let detailCustomFields = new Set(fields);
let detailVisibleFields = top8.filter(f => fields.includes(f));
let sortCol = null;
let sortAsc = false;

const on  = "background:#546e7a;color:#fff";
const off = "background:#fff;color:#546e7a";

function colorFor(val) {{
  if (val >= 0.90) return ["#1b5e20", "#c8e6c9"];
  if (val >= 0.75) return ["#e65100", "#ffe0b2"];
  return ["#b71c1c", "#ffcdd2"];
}}

// ── Metric toggle buttons ────────────────────────────────────────────
function buildMetricButtons() {{
  const base = "padding:6px 16px;border:1px solid #1976d2;cursor:pointer;font-size:13px";
  document.getElementById("metric-toggle").innerHTML = metricCols.map((m, i) => {{
    const isFirst = i === 0, isLast = i === metricCols.length - 1;
    const br = isFirst ? "border-radius:20px 0 0 20px" : isLast ? "border-radius:0 20px 20px 0" : "border-radius:0";
    const ml = i > 0 ? "margin-left:-1px;" : "";
    return `<button id="btn-${{m}}" onclick="showMetric('${{m}}')"
      style="${{base}};${{br}};${{ml}}background:#fff;color:#1976d2">
      ${{m.charAt(0).toUpperCase() + m.slice(1)}}
    </button>`;
  }}).join("");
}}

function showMetric(metric) {{
  currentMetric = metric;
  const base = "padding:6px 16px;border:1px solid #1976d2;cursor:pointer;font-size:13px";
  metricCols.forEach((m, i) => {{
    const isFirst = i === 0, isLast = i === metricCols.length - 1;
    const br = isFirst ? "border-radius:20px 0 0 20px" : isLast ? "border-radius:0 20px 20px 0" : "border-radius:0";
    const ml = i > 0 ? "margin-left:-1px;" : "";
    const active = m === metric ? "background:#1976d2;color:#fff" : "background:#fff;color:#1976d2";
    document.getElementById(`btn-${{m}}`).style.cssText = `${{base}};${{br}};${{ml}}${{active}}`;
  }});
  renderHeatmap();
}}

function fieldCheckboxes(selectedSet, onChangeFn) {{
  return [...fields].sort().map(f => `
    <label style="display:flex;align-items:center;gap:8px;padding:5px 14px;cursor:pointer;font-size:13px;font-family:monospace">
      <input type="checkbox" ${{selectedSet.has(f) ? "checked" : ""}} onchange="${{onChangeFn}}('${{f}}')" style="cursor:pointer">
      ${{f}}
    </label>`).join("");
}}

function setFieldButtons(prefix, mode) {{
  const b = "padding:4px 12px;border:1px solid #546e7a;cursor:pointer;font-size:12px;";
  document.getElementById(prefix+"btn-main").style.cssText   = b+"border-radius:20px 0 0 20px;"              +(mode==="main"  ?on:off);
  document.getElementById(prefix+"btn-custom").style.cssText = b+"border-radius:0;margin-left:-1px;"          +(mode==="custom"?on:off);
  document.getElementById(prefix+"btn-all").style.cssText    = b+"border-radius:0 20px 20px 0;margin-left:-1px;"+(mode==="all"   ?on:off);
}}

// ── Heatmap fields ───────────────────────────────────────────────────
function setHmapFields(mode) {{
  setFieldButtons("h", mode);
  const dd = document.getElementById("hmap-dropdown");
  if (mode === "custom") {{
    dd.innerHTML = fieldCheckboxes(hmapCustomFields, "toggleHmapField");
    dd.style.display = "block";
    hmapVisibleFields = fields.filter(f => hmapCustomFields.has(f));
  }} else {{
    dd.style.display = "none";
    hmapVisibleFields = mode === "main" ? top8.filter(f => fields.includes(f)) : [...fields];
  }}
  renderHeatmap();
}}

function toggleHmapField(f) {{
  hmapCustomFields.has(f) ? hmapCustomFields.delete(f) : hmapCustomFields.add(f);
  hmapVisibleFields = fields.filter(f => hmapCustomFields.has(f));
  renderHeatmap();
}}

// ── Detail fields ────────────────────────────────────────────────────
function setDetailFields(mode) {{
  setFieldButtons("d", mode);
  const dd = document.getElementById("detail-dropdown");
  if (mode === "custom") {{
    dd.innerHTML = fieldCheckboxes(detailCustomFields, "toggleDetailField");
    dd.style.display = "block";
    detailVisibleFields = fields.filter(f => detailCustomFields.has(f));
  }} else {{
    dd.style.display = "none";
    detailVisibleFields = mode === "main" ? top8.filter(f => fields.includes(f)) : [...fields];
  }}
  renderDetailHeader();
  renderDetailBody(expandedTag);
}}

function toggleDetailField(f) {{
  detailCustomFields.has(f) ? detailCustomFields.delete(f) : detailCustomFields.add(f);
  detailVisibleFields = fields.filter(f => detailCustomFields.has(f));
  renderDetailHeader();
  renderDetailBody(expandedTag);
}}

// ── Heatmap render ───────────────────────────────────────────────────
function renderHeatmap() {{
  document.getElementById("heatmap-header").innerHTML =
    `<th style="text-align:left;padding:10px 14px;border-bottom:2px solid #e0e0e0">Field</th>` +
    tags.map(t => {{
      const active = t === expandedTag;
      return `<th onclick="toggleExpand('${{t}}')"
        style="padding:10px 14px;border-bottom:2px solid ${{active?"#1976d2":"#e0e0e0"}};text-align:center;
               font-family:monospace;font-weight:600;cursor:pointer;color:${{active?"#1976d2":"inherit"}};
               white-space:nowrap;user-select:none">${{t}} ${{active?"▲":"▾"}}</th>`;
    }}).join("");

  document.getElementById("heatmap-body").innerHTML = hmapVisibleFields.map(field => {{
    const cells = tags.map(tag => {{
      const val = data[currentMetric][tag][field];
      if (val === null) return `<td style="padding:8px 14px;border-bottom:1px solid #f5f5f5;text-align:center;color:#ccc">—</td>`;
      const [text, bg] = colorFor(val);
      return `<td style="padding:8px 14px;border-bottom:1px solid #f5f5f5;text-align:center;background:${{bg}};color:${{text}};font-weight:600">${{val.toFixed(2)}}</td>`;
    }}).join("");
    return `<tr><td style="padding:8px 14px;border-bottom:1px solid #f5f5f5;font-family:monospace">${{field}}</td>${{cells}}</tr>`;
  }}).join("");
}}

// ── Detail render ────────────────────────────────────────────────────
function renderDetailHeader() {{
  document.getElementById("detail-header").innerHTML = detailCols.map(col => {{
    const isGrey   = countCols.includes(col);
    const isMetric = metricCols.includes(col);
    const active   = sortCol === col;
    const arrow    = active ? (sortAsc ? " ↑" : " ↓") : " ↕";
    return `<th onclick="sortDetail('${{col}}')"
      style="padding:8px 14px;border-bottom:2px solid #e0e0e0;text-align:${{col===fieldCol?"left":"center"}};
             font-weight:600;cursor:pointer;user-select:none;white-space:nowrap;
             color:${{active?"#1976d2":isGrey?"#888":"inherit"}}">${{col}}${{arrow}}</th>`;
  }}).join("");
}}

function renderDetailBody(tag) {{
  let rows = detailVisibleFields.map(field => {{
    const row = {{ [fieldCol]: field, _missing: counts[tag][field] === null }};
    countCols.forEach(c => {{ row[c] = counts[tag][field]?.[c] ?? null; }});
    metricCols.forEach(m => {{ row[m] = data[m][tag][field]; }});
    return row;
  }});

  if (sortCol) rows.sort((a, b) => {{
    if (a._missing && !b._missing) return 1;
    if (!a._missing && b._missing) return -1;
    const v = sortCol === fieldCol
      ? a[fieldCol].localeCompare(b[fieldCol])
      : (a[sortCol] ?? -1) - (b[sortCol] ?? -1);
    return sortAsc ? v : -v;
  }});

  const grey = "padding:8px 14px;border-bottom:1px solid #f5f5f5;text-align:center;color:#888";
  const dash = "padding:8px 14px;border-bottom:1px solid #f5f5f5;text-align:center;color:#ccc";

  document.getElementById("detail-body").innerHTML = rows.map(r => {{
    if (r._missing) {{
      const empties = detailCols.slice(1).map(() => `<td style="${{dash}}">—</td>`).join("");
      return `<tr><td style="padding:8px 14px;border-bottom:1px solid #f5f5f5;font-family:monospace;color:#bbb">${{r[fieldCol]}}</td>${{empties}}</tr>`;
    }}
    const cells = detailCols.map(col => {{
      if (col === fieldCol) return `<td style="padding:8px 14px;border-bottom:1px solid #f5f5f5;font-family:monospace">${{r[fieldCol]}}</td>`;
      if (countCols.includes(col)) return `<td style="${{grey}}">${{r[col]}}</td>`;
      const [t, bg] = colorFor(r[col]);
      return `<td style="padding:8px 14px;border-bottom:1px solid #f5f5f5;text-align:center;background:${{bg}};color:${{t}};font-weight:600">${{r[col].toFixed(2)}}</td>`;
    }}).join("");
    return `<tr>${{cells}}</tr>`;
  }}).join("");
}}

function renderDetail(tag) {{
  document.getElementById("detail-tag").textContent = tag;
  document.getElementById("detail-dropdown").style.display = "none";
  detailCustomFields = new Set(top8.filter(f => fields.includes(f)));
  setFieldButtons("d", "main");
  detailVisibleFields = top8.filter(f => fields.includes(f));
  sortCol = null;
  renderDetailHeader();
  renderDetailBody(tag);
  document.getElementById("detail-panel").style.display = "block";
}}

function sortDetail(col) {{
  if (sortCol === col) {{ sortAsc = !sortAsc; }} else {{ sortCol = col; sortAsc = false; }}
  renderDetailHeader(); renderDetailBody(expandedTag);
}}

function closeDetail() {{
  expandedTag = null; sortCol = null;
  document.getElementById("detail-panel").style.display = "none";
  renderHeatmap();
}}

function toggleExpand(tag) {{
  if (expandedTag === tag) {{ closeDetail(); }}
  else {{ expandedTag = tag; renderDetail(tag); renderHeatmap(); }}
}}

// Close dropdowns when clicking outside
document.addEventListener("click", e => {{
  if (!e.target.closest("#hmap-dropdown") && !e.target.closest("[onclick^='setHmapFields']")) {{
    document.getElementById("hmap-dropdown").style.display = "none";
  }}
  if (!e.target.closest("#detail-dropdown") && !e.target.closest("[onclick^='setDetailFields']")) {{
    const dd = document.getElementById("detail-dropdown");
    if (dd) dd.style.display = "none";
  }}
}});

buildMetricButtons();
setFieldButtons("h", "main");
showMetric(metricCols[0]);
</script>

</div>
"""


def update_mkdocs() -> None:
    config = yaml.safe_load(MKDOCS_YML.read_text())
    nav_entry = "results/comparison.md"

    for section in config["nav"]:
        if "Results" in section:
            existing = [list(p.values())[0] for p in section["Results"]]
            if nav_entry not in existing:
                section["Results"].insert(0, {"Comparison": nav_entry})
            break

    MKDOCS_YML.write_text(
        yaml.dump(config, sort_keys=False, default_flow_style=False, allow_unicode=True)
    )
    print(f"Updated → {MKDOCS_YML}")


def main():
    experiments = load_experiments()

    content = "# Experiment Comparison\n\n"
    content += summary_cards(experiments)
    content += heatmap(experiments)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(content)
    print(f"Written → {OUTPUT_PATH}")

    update_mkdocs()


if __name__ == "__main__":
    main()
