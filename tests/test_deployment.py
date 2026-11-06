"""Deployment configuration checks (Vercel). Needs no ML libraries.

Run with NEUROSCAN_CHECK_BUNDLE=1 in an environment that contains only the
serving dependencies (requirements.txt) to also enforce the bundle-size budget.
"""
from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_BUDGET_MB = 250  # Vercel allows 500 MB for Python; stay under the older 250 MB limit too
VERCEL_PYTHONS = {"3.12", "3.13"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# .vercelignore simulation (gitignore syntax subset used in this repository)
# --------------------------------------------------------------------------- #
def _patterns():
    lines = (ROOT / ".vercelignore").read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _ignored(rel: str, patterns) -> bool:
    parts = rel.split("/")
    for raw in patterns:
        pattern, dir_only = raw.rstrip("/"), raw.endswith("/")
        pattern = pattern[3:] if pattern.startswith("**/") else pattern
        anchored = "/" in pattern
        candidates = []
        for i in range(1, len(parts) + 1):
            prefix = "/".join(parts[:i])
            is_dir = i < len(parts)
            if dir_only and not is_dir:
                continue
            candidates.append(prefix if anchored else parts[i - 1])
        if any(fnmatch.fnmatchcase(c, pattern) for c in candidates):
            return True
    return False


@pytest.fixture(scope="module")
def deployed_files():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("git checkout required")
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    patterns = _patterns()
    files = {f for f in out.splitlines() if f and (ROOT / f).is_file()}
    return {f for f in files if not _ignored(f, patterns)}


def test_runtime_files_are_deployed(deployed_files):
    required = {
        "server.py", "pyproject.toml", "requirements.txt", ".python-version", "vercel.json",
        "src/__init__.py", "src/config.py", "src/inference.py", "src/runtime.py",
        "src/preprocessing.py", "src/ensemble.py", "models/onnx/manifest.json",
        "public/index.html", "public/static/app.js", "public/static/style.css", "public/samples/index.json",
    }
    manifest = json.loads((ROOT / "models/onnx/manifest.json").read_text())
    required |= {f"models/onnx/{e['file']}" for e in manifest["models"].values()}
    missing = required - deployed_files
    assert not missing, f"excluded by .vercelignore or not committed: {sorted(missing)}"


def test_training_artifacts_are_not_deployed(deployed_files):
    leaked = [f for f in deployed_files if f.endswith((".pth", ".ipynb"))
              or f.split("/")[0] in {"tests", "notebooks", "scripts", "Dataset"}
              or f in {"app.py", "train.py", "evaluate.py", "compare.py", "predict.py"}]
    assert not leaked, leaked


def test_every_module_the_server_imports_is_deployed(deployed_files):
    """Walk server.py's imports through src/ and make sure each module ships."""
    seen, todo = set(), ["server.py"]
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        tree = ast.parse((ROOT / rel).read_text())
        for node in tree.body:  # module-level imports only (lazy imports are optional backends)
            if isinstance(node, ast.ImportFrom):
                if node.level == 1:
                    name = f"src.{node.module}" if node.module else "src"
                else:
                    name = node.module or ""
                if name.startswith("src"):
                    todo.append(name.replace(".", "/") + ".py" if name != "src" else "src/__init__.py")
    assert seen - deployed_files == set()


def test_deployed_size_within_budget(deployed_files):
    code_mb = sum((ROOT / f).stat().st_size for f in deployed_files) / 2**20
    assert code_mb < 80, f"deployed project files are {code_mb:.1f} MB"
    if os.environ.get("NEUROSCAN_CHECK_BUNDLE") == "1":
        site = Path(sysconfig.get_paths()["purelib"])
        deps_mb = sum(p.stat().st_size for p in site.rglob("*") if p.is_file()) / 2**20
        total = code_mb + deps_mb
        assert total < BUNDLE_BUDGET_MB, f"bundle ~{total:.0f} MB (deps {deps_mb:.0f} MB + project {code_mb:.0f} MB)"


# --------------------------------------------------------------------------- #
# Configuration consistency
# --------------------------------------------------------------------------- #
def _requirements(path: Path):
    reqs = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-r"):
            reqs.append(re.sub(r"\s+", "", line))
    return reqs


def test_pyproject_dependencies_match_requirements_txt():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert [re.sub(r"\s+", "", d) for d in project["dependencies"]] == _requirements(ROOT / "requirements.txt")
    banned = {"torch", "torchvision", "scipy", "scikit-learn", "matplotlib", "pandas", "seaborn", "gradio"}
    names = {re.split(r"[<>=!~\[ ]", d)[0].lower() for d in project["dependencies"]}
    assert not names & banned, f"heavy packages would be installed on Vercel: {names & banned}"


def test_python_version_is_supported_by_vercel():
    pinned = (ROOT / ".python-version").read_text().strip()
    assert pinned in VERCEL_PYTHONS
    requires = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"]
    major, minor = map(int, pinned.split("."))
    low = tuple(map(int, re.search(r">=\s*(\d+)\.(\d+)", requires).groups()))
    high = tuple(map(int, re.search(r"<\s*(\d+)\.(\d+)", requires).groups()))
    assert low <= (major, minor) < high


def test_vercel_entrypoint_resolves_to_a_fastapi_app():
    entry = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["vercel"]["entrypoint"]
    module, attr = entry.split(":")
    assert (ROOT / f"{module}.py").exists()
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}, fastapi; assert isinstance({module}.{attr}, fastapi.FastAPI)"],
        cwd=ROOT, env={**os.environ, "VERCEL": "1"}, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_vercel_json_is_valid_and_consistent_with_the_api():
    config = json.loads((ROOT / "vercel.json").read_text())
    assert not {"builds", "routes"} & set(config), "legacy keys disable zero-config FastAPI detection"
    by_source = {rule["source"]: {h["key"]: h["value"] for h in rule["headers"]} for rule in config["headers"]}
    tree = ast.parse((ROOT / "server.py").read_text())
    api_headers = next(ast.literal_eval(n.value) for n in tree.body
                       if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SECURITY_HEADERS")
    for source in ("/", "/index.html"):
        for key in ("Content-Security-Policy", "X-Frame-Options", "X-Content-Type-Options", "Referrer-Policy"):
            assert by_source[source][key] == api_headers[key], (source, key)


def test_static_site_references_resolve_and_respect_the_csp():
    html = (ROOT / "public/index.html").read_text()
    refs = re.findall(r'(?:src|href)="(/(?:static|samples)/[^"]+)"', html)
    assert refs, "no local asset references found"
    for ref in refs:
        assert (ROOT / "public" / ref.lstrip("/")).is_file(), ref
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html), "inline scripts are blocked by the CSP"
    assert not re.search(r'<script[^>]+src="https?://', html), "third-party scripts are blocked by the CSP"


def test_sample_index_is_current():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from build_sample_index import build_index
    finally:
        sys.path.pop(0)
    assert json.loads((ROOT / "public/samples/index.json").read_text()) == build_index(ROOT / "public/samples")


def test_onnx_exports_match_files_and_checkpoints():
    manifest = json.loads((ROOT / "models/onnx/manifest.json").read_text())
    for entry in manifest["models"].values():
        size_mb = (ROOT / "models/onnx" / entry["file"]).stat().st_size / 2**20
        assert size_mb < 95, f"{entry['file']} is {size_mb:.0f} MB (GitHub rejects files over 100 MB)"
    specs = {"custom_cnn": "custom_cnn_final_model.pth", "resnet18": "resnet18_final_model.pth",
             "mobilenet_v3": "mobilenet_v3_final_model.pth", "vit": "best_vit_model.pth"}
    for key, entry in manifest["models"].items():
        assert sha256(ROOT / "models/onnx" / entry["file"]) == entry["sha256"], f"{key}: ONNX file changed"
    excluded = manifest.get("excluded", {})
    assert "vit" in excluded or "vit" in manifest["models"], "ViT must be exported or explicitly excluded"
    for key, checkpoint in specs.items():
        path = ROOT / "models" / checkpoint
        if path.exists() and key not in excluded:
            assert key in manifest["models"], f"{checkpoint} has no ONNX export: run scripts/export_onnx.py"
            assert manifest["models"][key]["source_sha256"] == sha256(path), \
                f"{checkpoint} changed since export: run scripts/export_onnx.py"
