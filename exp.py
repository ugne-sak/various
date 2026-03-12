"""
Generates a MkDocs page for a single experiment and registers it in mkdocs.yml.
The experiment type is inferred from the folder name: folders containing
'extraction' produce a results table, folders containing 'classification'
produce a confusion matrix page.

Usage:
    uv run scripts/generate_experiment.py --tag experiment_20260308_1600_n150_d400_classification
    uv run scripts/generate_experiment.py --tag experiment_20260308_1450_n150_d400_extraction
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd
import yaml

# ── User-configurable ─────────────────────────────────────────────────
DATA_DIR = Path("data")
DOCS_DIR = Path("docs/results")
DOCS_NAV_DIR = "results"
MKDOCS_YML = Path("mkdocs.yml")
TAB_NAME = "Results"
RESULTS_FILE = "extraction_results.csv"
CM_FILE = "cm.png"
META_FILE = "meta.json"
META_EXCLUDE = []  # keys from meta.json to hide in the output page
# ─────────────────────────────────────────────────────────────────────


# ── Loaders ───────────────────────────────────────────────────────────


def extract_tag(folder_name: str) -> str:
    match = re.search(r"\d{8}_\d{4}", folder_name)
    return match.group() if match else folder_name


# ── Writers ───────────────────────────────────────────────────────────


def format_experiment_metadata(folder: Path) -> str:
    meta_path = folder / META_FILE
    if not meta_path.exists():
        return ""
    meta = json.loads(meta_path.read_text())[0]
    lines = "\n".join(
        f"- **{k}**: {v:,}" if isinstance(v, int) else f"- **{k}**: {v}"
        for k, v in meta.items()
        if k not in META_EXCLUDE
    )
    return f"## Experiment details\n\n{lines}\n\n"


def write_extraction_md(folder: Path, tag: str) -> Path:
    md_path = DOCS_DIR / f"experiment_{tag}_extraction.md"

    df = pd.read_csv(folder / RESULTS_FILE)
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])
    meta = format_experiment_metadata(folder)
    content = (
        f"# Experiment {tag}\n\n"
        f"{meta}"
        f"## Extraction results (per field)\n\n"
        f"{df.to_markdown(index=False)}\n"
    )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(content)
    print(f"Written → {md_path}")
    return md_path


def write_classification_md(folder: Path, tag: str) -> Path:
    md_path = DOCS_DIR / f"experiment_{tag}_classification.md"

    meta = format_experiment_metadata(folder)
    img_dest = DOCS_DIR / f"experiment_{tag}_cm.png"
    shutil.copy(folder / CM_FILE, img_dest)
    content = (
        f"# Experiment {tag} — Classification\n\n"
        f"{meta}"
        f"## Confusion Matrix\n\n"
        f"[![Confusion Matrix](experiment_{tag}_cm.png)](experiment_{tag}_cm.png)\n"
    )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(content)
    print(f"Written → {md_path}")
    return md_path


# ── Nav ───────────────────────────────────────────────────────────────


def update_mkdocs(tag: str, section_name: str, nav_path: str) -> None:
    config = yaml.safe_load(MKDOCS_YML.read_text())

    for tab in config["nav"]:
        if TAB_NAME not in tab:
            continue
        results = tab[TAB_NAME]
        for item in results:
            if section_name in item:
                existing = [list(e.values())[0] for e in item[section_name]]
                if nav_path not in existing:
                    item[section_name].append({f"Experiment {tag}": nav_path})
                break
        else:
            results.append({section_name: [{f"Experiment {tag}": nav_path}]})
        break

    MKDOCS_YML.write_text(
        yaml.dump(config, sort_keys=False, default_flow_style=False, allow_unicode=True)
    )
    print(f"Updated → {MKDOCS_YML}")


# ── Entry point ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        required=True,
        help="Experiment folder name, e.g. experiment_20260308_1600_n150_d400_classification",
    )
    args = parser.parse_args()

    folder = DATA_DIR / args.tag
    if not folder.exists():
        raise FileNotFoundError(f"{folder} not found")

    tag = extract_tag(args.tag)

    if "classification" in args.tag:
        md_path = write_classification_md(folder, tag)
        update_mkdocs(tag, "Classification", f"{DOCS_NAV_DIR}/{md_path.name}")
    elif "extraction" in args.tag:
        md_path = write_extraction_md(folder, tag)
        update_mkdocs(tag, "Extraction", f"{DOCS_NAV_DIR}/{md_path.name}")
    else:
        raise ValueError("Experiment tag must contain 'classification' or 'extraction'")


if __name__ == "__main__":
    main()
