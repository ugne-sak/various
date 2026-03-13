import json
import re
from pathlib import Path

import pandas as pd
import yaml

# ── User-configurable paths ──────────────────────────────────────────
DOCS_DIR = Path("docs/results")
MKDOCS_YML = Path("mkdocs.yml")
OUTPUT_PATH = DOCS_DIR / "comparison.md"
NAV_ENTRY = "results/comparison.md"
TAB_NAME = "Results"

# Folders to include in the comparison (order is preserved)
EXPERIMENT_DIRS = [
    Path("data/experiment_20260306_1545_n100_d300"),
    Path("data/experiment_20260307_0912_n120_d350"),
    Path("data/experiment_20260308_1422_n150_d400"),
]

# Name of the CSV file inside each experiment folder
RESULTS_FILE = "extraction_results.csv"

# ── Field configuration ──────────────────────────────────────────────
# Column in each CSV that identifies the field name
FIELD_COL = "field"

# Columns shown with colour-coded metric styling (higher = greener)
METRIC_COLS = ["accuracy", "precision"]

# Tooltip shown on hover for each metric button (omit a key for no tooltip)
# Each entry: (short description, formula)
METRIC_DESCRIPTIONS = {
    "accuracy":  ("Share of cases where the model found the correct value", "n_correct / n_cases"),
    "precision": ("Share of extractions that turned out to be correct",     "n_correct / n_extracted"),
}

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

# ── Color thresholds: (min_value, text_color, bg_color, label) ────────
# Use None for the fallback (last) entry
COLOR_THRESHOLDS = [
    (0.90, "#1b5e20", "#c8e6c9", "≥ 0.90"),
    (0.80, "#33691e", "#dcedc8", "≥ 0.80"),
    (0.60, "#f57f17", "#fff9c4", "≥ 0.60"),
    (None, "#b71c1c", "#ffcdd2", "< 0.60"),
]

# ── Shared HTML fragments ────────────────────────────────────────────
_GREY_BOX = "background:#f7f7f7;border-radius:8px;padding:12px 16px 16px"

_LEGEND = (
    '<div style="display:flex;align-items:center;gap:12px;font-size:11px;color:#999">'
    + "".join(
        f'<span style="display:flex;align-items:center;gap:4px">'
        f'<span style="width:11px;height:11px;border-radius:2px;background:{bg};display:inline-block"></span>'
        f" {label}</span>"
        for _, _, bg, label in COLOR_THRESHOLDS
    )
    + "</div>"
)


def _field_selector(btn_prefix: str, set_fn: str, dd_id: str) -> str:
    return f"""<div style="display:flex;align-items:center;gap:8px">
    <span style="font-size:11px;color:#999;letter-spacing:1px;margin-right:4px">FIELDS</span>
    <button id="{btn_prefix}btn-main"   onclick="{set_fn}('main')"   style="padding:4px 12px;border-radius:20px 0 0 20px;border:1px solid #546e7a;background:#546e7a;color:#fff;cursor:pointer;font-size:12px">Main</button>
    <button id="{btn_prefix}btn-custom" onclick="{set_fn}('custom')" style="padding:4px 12px;border-radius:0;border:1px solid #546e7a;margin-left:-1px;background:#fff;color:#546e7a;cursor:pointer;font-size:12px">Custom ▾</button>
    <button id="{btn_prefix}btn-all"    onclick="{set_fn}('all')"    style="padding:4px 12px;border-radius:0 20px 20px 0;border:1px solid #546e7a;margin-left:-1px;background:#fff;color:#546e7a;cursor:pointer;font-size:12px">All</button>
    <div id="{dd_id}" style="display:none;position:absolute;top:calc(100% + 4px);left:60px;background:#fff;border:1px solid #ddd;border-radius:6px;box-shadow:0 4px 12px rgba(0,0,0,0.1);z-index:200;min-width:180px;max-height:240px;overflow-y:auto;padding:6px 0"></div>
  </div>"""


def experiment_tag(folder: Path) -> str:
    match = re.search(r"\d{8}_\d{4}", folder.name)
    return match.group() if match else folder.name


def parse_experiment_meta(folder: Path) -> dict:
    """Extract n (cases) and d (documents) from folder name, e.g. experiment_..._n100_d300."""
    n = re.search(r"_n(\d+)", folder.name)
    d = re.search(r"_d(\d+)", folder.name)
    return {
        "n_cases": int(n.group(1)) if n else None,
        "n_docs":  int(d.group(1)) if d else None,
    }


def load_experiments() -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    dfs, meta = {}, {}
    for folder in EXPERIMENT_DIRS:
        tag = experiment_tag(folder)
        df = pd.read_csv(folder / RESULTS_FILE)
        df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])
        dfs[tag] = df
        meta[tag] = parse_experiment_meta(folder)
    return dfs, meta


def _infer_count_cols(experiments: dict[str, pd.DataFrame]) -> list[str]:
    return list(dict.fromkeys(
        c for df in experiments.values() for c in df.columns
        if c != FIELD_COL and c not in METRIC_COLS
    ))


def summary_cards(experiments: dict[str, pd.DataFrame], meta: dict[str, dict]) -> str:
    tags = list(experiments.keys())
    summary_data = {
        tag: {m: round(df[m].mean(), 2) for m in METRIC_COLS}
        for tag, df in experiments.items()
    }
    summary_json = json.dumps(summary_data)
    metric_cols_json = json.dumps(METRIC_COLS)
    defaults = [tags[-2], tags[-1]]
    options = "".join(f'<option value="{t}">{t}</option>' for t in tags)

    selectors = "\n".join(f"""
  <div style="flex:1">
    <div style="font-size:11px;color:#999;letter-spacing:1px;margin-bottom:6px">EXPERIMENT {i + 1}</div>
    <select id="sel-{i + 1}" onchange="render()"
      style="width:100%;padding:7px 10px;border-radius:6px;border:1px solid #ccc;font-family:monospace;font-size:13px;margin-bottom:12px;background:#fff">
      {options}
    </select>
    <div id="card-{i + 1}"></div>
  </div>""" for i in range(len(defaults)))

    init_selectors = "\n".join(
        f'document.getElementById("sel-{i + 1}").value = "{t}";'
        for i, t in enumerate(defaults)
    )

    return f"""
<div style="margin-bottom:32px">

<div style="display:flex;align-items:flex-start;gap:16px;margin-bottom:16px">
{selectors}
</div>

<div id="diff-panel"></div>

</div>

<script>
const summaryData = {summary_json};
const summaryMetricCols = {metric_cols_json};

{init_selectors}

function colorFor(val) {{
  for (const [threshold, text, bg] of colorThresholds)
    if (threshold === null || val >= threshold) return [text, bg];
}}

function card(tag) {{
  const d = summaryData[tag];
  const tiles = summaryMetricCols.map(m => {{
    const [text, bg] = colorFor(d[m]);
    return `<div style="flex:1;padding:10px;border-radius:4px;background:${{bg}};text-align:center">
      <div style="font-size:10px;color:#777;margin-bottom:2px">AVG ${{m.toUpperCase()}}</div>
      <div style="font-size:22px;font-weight:700;color:${{text}}">${{d[m].toFixed(2)}}</div>
    </div>`;
  }}).join("");
  return `<div style="padding:16px;border-radius:8px;background:#fafafa;border:1px solid #e0e0e0">
    <div style="display:flex;gap:8px">${{tiles}}</div>
  </div>`;
}}

function deltaCard(m, v1, v2) {{
  const delta = Math.round((v2 - v1) * 100) / 100;
  const color = delta > 0 ? "#2e7d32" : delta < 0 ? "#c62828" : "#9e9e9e";
  const bg    = delta > 0 ? "#f1f8e9" : delta < 0 ? "#ffebee" : "#f5f5f5";
  const arrow = delta > 0 ? "▲" : delta < 0 ? "▼" : "▬";
  const sign  = delta > 0 ? "+" : "";
  return `
    <div style="flex:1;padding:14px 18px;border-radius:8px;background:${{bg}};text-align:center">
      <div style="font-size:10px;color:#aaa;letter-spacing:1px;margin-bottom:8px">AVG ${{m.toUpperCase()}}</div>
      <div style="font-size:32px;font-weight:800;color:${{color}};line-height:1">${{sign}}${{delta.toFixed(2)}}</div>
      <div style="font-size:11px;color:#999;margin-top:8px">${{arrow}}&nbsp;${{v1.toFixed(2)}} → ${{v2.toFixed(2)}}</div>
    </div>`;
}}

function render() {{
  const t1 = document.getElementById("sel-1").value;
  const t2 = document.getElementById("sel-2").value;
  document.getElementById("card-1").innerHTML = card(t1);
  document.getElementById("card-2").innerHTML = card(t2);
  const d1 = summaryData[t1], d2 = summaryData[t2];
  const cards = summaryMetricCols.map(m => deltaCard(m, d1[m], d2[m])).join("");
  document.getElementById("diff-panel").innerHTML = `
    <div>
      <div style="font-size:10px;color:#bbb;letter-spacing:2px;margin-bottom:8px">CHANGE (EXP 1 → EXP 2)</div>
      <div style="display:flex;gap:12px">${{cards}}</div>
    </div>`;
}}

render();
</script>
"""


def heatmap(experiments: dict[str, pd.DataFrame], meta: dict[str, dict]) -> str:
    tags = list(experiments.keys())
    count_cols = _infer_count_cols(experiments)

    fields = list(dict.fromkeys(
        f for df in experiments.values() for f in df[FIELD_COL].tolist()
    ))

    metric_data: dict[str, dict] = {}
    for metric in METRIC_COLS:
        metric_data[metric] = {}
        for tag, df in experiments.items():
            rows = {row[FIELD_COL]: row for _, row in df.iterrows()}
            metric_data[metric][tag] = {
                f: round(rows[f][metric], 2) if f in rows else None for f in fields
            }

    counts_data: dict = {}
    for tag, df in experiments.items():
        rows = {row[FIELD_COL]: row for _, row in df.iterrows()}
        counts_data[tag] = {
            f: {c: int(rows[f][c]) for c in count_cols if c in rows[f].index}
            if f in rows else None
            for f in fields
        }

    fields_json      = json.dumps(fields)
    tags_json        = json.dumps(tags)
    metric_data_json = json.dumps(metric_data)
    counts_json      = json.dumps(counts_data)
    top8_json        = json.dumps(TOP8_FIELDS)
    metric_cols_json = json.dumps(METRIC_COLS)
    count_cols_json  = json.dumps(count_cols)
    field_col_json   = json.dumps(FIELD_COL)
    metric_desc_json      = json.dumps(METRIC_DESCRIPTIONS)
    meta_json             = json.dumps(meta)
    color_thresholds_json = json.dumps([[t, tc, bg] for t, tc, bg, _ in COLOR_THRESHOLDS])

    hmap_sel   = _field_selector("h", "setHmapFields",   "hmap-dropdown")
    detail_sel = _field_selector("d", "setDetailFields", "detail-dropdown")

    return f"""
<div>

<!-- Metric toggle -->
<div style="margin-bottom:20px">
  <span style="margin-right:8px;font-size:13px;color:#666">Metric:</span>
  <span id="metric-toggle" style="display:inline-flex"></span>
</div>

<!-- Heatmap -->
<div style="{_GREY_BOX};margin-bottom:24px">
<div style="margin-bottom:10px;display:flex;align-items:center;justify-content:space-between;position:relative">
  {hmap_sel}
  {_LEGEND}
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
    <div style="display:flex;align-items:baseline;gap:16px">
      <div>
        <span style="font-size:11px;color:#999;letter-spacing:1px">EXPERIMENT&nbsp;</span>
        <span id="detail-tag" style="color:#1976d2;font-family:monospace;font-weight:700;font-size:13px"></span>
      </div>
      <span id="detail-meta" style="font-size:11px;color:#aaa"></span>
    </div>
    <button onclick="closeDetail()" style="padding:4px 10px;border-radius:4px;border:1px solid #ddd;background:#fff;color:#999;cursor:pointer;font-size:12px">✕ close details</button>
  </div>
  <div style="{_GREY_BOX}">
  <div style="margin-bottom:10px;display:flex;align-items:center;justify-content:space-between;position:relative">
    {detail_sel}
    {_LEGEND}
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
const metricDesc       = {metric_desc_json};
const colorThresholds  = {color_thresholds_json};

const fields = {fields_json};
const top8   = {top8_json};
const tags   = {tags_json};
const data   = {metric_data_json};
const counts = {counts_json};
const meta   = {meta_json};

let expandedTag = null;
let currentMetric = metricCols[0];
let hmapCustomFields    = new Set(top8.filter(f => fields.includes(f)));
let hmapVisibleFields   = top8.filter(f => fields.includes(f));
let detailCustomFields  = new Set(fields);
let detailVisibleFields = top8.filter(f => fields.includes(f));
let sortCol = null, sortAsc = false;

const on  = "background:#546e7a;color:#fff";
const off = "background:#fff;color:#546e7a";
const TD  = "padding:8px 14px;border-bottom:1px solid #f5f5f5";
const METRIC_BTN = "padding:6px 16px;border:1px solid #1976d2;cursor:pointer;font-size:13px";

// ── Metric tooltip ───────────────────────────────────────────────────
const _tip = document.createElement("div");
_tip.style.cssText = "display:none;position:fixed;z-index:9999;background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:10px 14px;box-shadow:0 4px 16px rgba(0,0,0,0.12);max-width:220px;pointer-events:none";
document.body.appendChild(_tip);

function showTip(m, btn) {{
  if (!metricDesc[m]) return;
  const [desc, formula] = metricDesc[m];
  _tip.innerHTML = `<div style="font-size:12px;color:#444;margin-bottom:6px">${{desc}}</div>
    <div style="font-family:monospace;font-size:11px;color:#1976d2;background:#e3f2fd;padding:3px 7px;border-radius:4px;display:inline-block">${{formula}}</div>`;
  const r = btn.getBoundingClientRect();
  _tip.style.display = "block";
  _tip.style.left = r.left + "px";
  _tip.style.top  = (r.bottom + 6) + "px";
}}

function hideTip() {{ _tip.style.display = "none"; }}

// ── Metric toggle ────────────────────────────────────────────────────
function buildMetricButtons() {{
  document.getElementById("metric-toggle").innerHTML = metricCols.map((m, i) => {{
    const isFirst = i === 0, isLast = i === metricCols.length - 1;
    const br = isFirst ? "border-radius:20px 0 0 20px" : isLast ? "border-radius:0 20px 20px 0" : "border-radius:0";
    const ml = i > 0 ? "margin-left:-1px;" : "";
    return `<button id="btn-${{m}}" onclick="showMetric('${{m}}')"
      onmouseenter="showTip('${{m}}', this)" onmouseleave="hideTip()"
      style="${{METRIC_BTN}};${{br}};${{ml}}background:#fff;color:#1976d2">
      ${{m.charAt(0).toUpperCase() + m.slice(1)}}</button>`;
  }}).join("");
}}

function showMetric(metric) {{
  currentMetric = metric;
  metricCols.forEach(m => {{
    document.getElementById(`btn-${{m}}`).style.background = m === metric ? "#1976d2" : "#fff";
    document.getElementById(`btn-${{m}}`).style.color      = m === metric ? "#fff"    : "#1976d2";
  }});
  renderHeatmap();
}}

// ── Field toggles ────────────────────────────────────────────────────
function fieldCheckboxes(selectedSet, onChangeFn) {{
  return [...fields].sort().map(f => `
    <label style="display:flex;align-items:center;gap:8px;padding:5px 14px;cursor:pointer;font-size:13px;font-family:monospace">
      <input type="checkbox" ${{selectedSet.has(f) ? "checked" : ""}} onchange="${{onChangeFn}}('${{f}}')" style="cursor:pointer">
      ${{f}}</label>`).join("");
}}

function setFieldButtons(prefix, mode) {{
  const b = "padding:4px 12px;border:1px solid #546e7a;cursor:pointer;font-size:12px;";
  document.getElementById(prefix+"btn-main").style.cssText   = b+"border-radius:20px 0 0 20px;"              +(mode==="main"  ?on:off);
  document.getElementById(prefix+"btn-custom").style.cssText = b+"border-radius:0;margin-left:-1px;"          +(mode==="custom"?on:off);
  document.getElementById(prefix+"btn-all").style.cssText    = b+"border-radius:0 20px 20px 0;margin-left:-1px;"+(mode==="all"   ?on:off);
}}

function applyFieldMode(prefix, ddId, mode, customSet, setVisible, toggleFn) {{
  setFieldButtons(prefix, mode);
  const dd = document.getElementById(ddId);
  if (mode === "custom") {{
    dd.innerHTML = fieldCheckboxes(customSet, toggleFn);
    dd.style.display = "block";
    setVisible(fields.filter(f => customSet.has(f)));
  }} else {{
    dd.style.display = "none";
    setVisible(mode === "main" ? top8.filter(f => fields.includes(f)) : [...fields]);
  }}
}}

function toggleField(f, customSet, setVisible, renderFn) {{
  customSet.has(f) ? customSet.delete(f) : customSet.add(f);
  setVisible(fields.filter(x => customSet.has(x)));
  renderFn();
}}

function setHmapFields(mode) {{
  applyFieldMode("h", "hmap-dropdown", mode, hmapCustomFields, v => hmapVisibleFields = v, "toggleHmapField");
  renderHeatmap();
}}

function toggleHmapField(f) {{
  toggleField(f, hmapCustomFields, v => hmapVisibleFields = v, renderHeatmap);
}}

function setDetailFields(mode) {{
  applyFieldMode("d", "detail-dropdown", mode, detailCustomFields, v => detailVisibleFields = v, "toggleDetailField");
  renderDetailHeader(); renderDetailBody(expandedTag);
}}

function toggleDetailField(f) {{
  toggleField(f, detailCustomFields, v => detailVisibleFields = v, () => {{ renderDetailHeader(); renderDetailBody(expandedTag); }});
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
      if (val === null) return `<td style="${{TD}};text-align:center;color:#ccc">—</td>`;
      const [text, bg] = colorFor(val);
      return `<td style="${{TD}};text-align:center;background:${{bg}};color:${{text}};font-weight:600">${{val.toFixed(2)}}</td>`;
    }}).join("");
    return `<tr><td style="${{TD}};font-family:monospace">${{field}}</td>${{cells}}</tr>`;
  }}).join("");
}}

// ── Detail render ────────────────────────────────────────────────────
function renderDetailHeader() {{
  document.getElementById("detail-header").innerHTML = detailCols.map(col => {{
    const active = sortCol === col;
    const arrow  = active ? (sortAsc ? " ↑" : " ↓") : " ↕";
    return `<th onclick="sortDetail('${{col}}')"
      style="padding:8px 14px;border-bottom:2px solid #e0e0e0;text-align:${{col===fieldCol?"left":"center"}};
             font-weight:600;cursor:pointer;user-select:none;white-space:nowrap;
             color:${{active?"#1976d2":countCols.includes(col)?"#888":"inherit"}}">${{col}}${{arrow}}</th>`;
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
    const v = sortCol === fieldCol ? a[fieldCol].localeCompare(b[fieldCol]) : (a[sortCol] ?? -1) - (b[sortCol] ?? -1);
    return sortAsc ? v : -v;
  }});

  document.getElementById("detail-body").innerHTML = rows.map(r => {{
    if (r._missing) {{
      const empties = detailCols.slice(1).map(() => `<td style="${{TD}};text-align:center;color:#ccc">—</td>`).join("");
      return `<tr><td style="${{TD}};font-family:monospace;color:#bbb">${{r[fieldCol]}}</td>${{empties}}</tr>`;
    }}
    const cells = detailCols.map(col => {{
      if (col === fieldCol) return `<td style="${{TD}};font-family:monospace">${{r[fieldCol]}}</td>`;
      if (countCols.includes(col)) return `<td style="${{TD}};text-align:center;color:#888">${{r[col]}}</td>`;
      const [t, bg] = colorFor(r[col]);
      return `<td style="${{TD}};text-align:center;background:${{bg}};color:${{t}};font-weight:600">${{r[col].toFixed(2)}}</td>`;
    }}).join("");
    return `<tr>${{cells}}</tr>`;
  }}).join("");
}}

function renderDetail(tag) {{
  document.getElementById("detail-tag").textContent = tag;
  const m = meta[tag] || {{}};
  const badge = (label, val) => val == null ? "" :
    `<span style="display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:12px;background:#e8f0fe;border:1px solid #c5d4f5">
      <span style="font-size:10px;color:#5c7cdb;letter-spacing:0.5px">${{label}}</span>
      <span style="font-size:13px;font-weight:700;color:#1a56c4">${{val}}</span>
    </span>`;
  document.getElementById("detail-meta").innerHTML = badge("cases:", m.n_cases) + " " + badge("docs:", m.n_docs);
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

document.addEventListener("click", e => {{
  if (!e.target.closest("#hmap-dropdown") && !e.target.closest("[onclick^='setHmapFields']"))
    document.getElementById("hmap-dropdown").style.display = "none";
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
    nav_entry = NAV_ENTRY

    for section in config["nav"]:
        if TAB_NAME in section:
            existing = [list(p.values())[0] for p in section[TAB_NAME]]
            if nav_entry not in existing:
                section[TAB_NAME].insert(0, {"Comparison": nav_entry})
            break

    MKDOCS_YML.write_text(
        yaml.dump(config, sort_keys=False, default_flow_style=False, allow_unicode=True)
    )
    print(f"Updated → {MKDOCS_YML}")


def main():
    experiments, meta = load_experiments()

    content = "# Experiment Comparison\n\n"
    content += summary_cards(experiments, meta)
    content += heatmap(experiments, meta)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(content)
    print(f"Written → {OUTPUT_PATH}")

    update_mkdocs()


if __name__ == "__main__":
    main()
