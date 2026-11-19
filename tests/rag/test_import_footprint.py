"""Import footprint, lazy export, and RSS/importtime tests for lean Pi RAG."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "tests" / "fixtures" / "rag_corpus"
FORBIDDEN_MODULES = (
    "numpy",
    "fastmcp",
    "markitdown",
    "rank_bm25",
    "voice_agent.rag.hybrid",
)
EXPECTED_RAG_ALL = [
    "Bm25Retriever",
    "Chunk",
    "DocIndex",
    "FileRecord",
    "HybridRetriever",
    "RefreshReport",
    "Retriever",
    "make_retriever_factory",
    "resolve_mode",
]


def _run_import_probe(
    code: str, *, env_overrides: dict[str, str] | None = None
) -> dict[str, object]:
    env = dict(os.environ)
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    importtime_ms: dict[str, float] = {}
    for line in proc.stderr.splitlines():
        if not line.startswith("import time:"):
            continue
        parts = line.split("|")
        if len(parts) != 3:
            continue
        cum_raw = parts[1].strip()
        mod_name = parts[2].strip()
        if cum_raw.isdigit():
            importtime_ms[mod_name] = round(int(cum_raw) / 1000.0, 2)

    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines, f"no stdout from probe; stderr tail:\n{proc.stderr[-500:]}"
    payload = json.loads(lines[-1])
    payload["importtime_ms"] = {
        k: v
        for k, v in importtime_ms.items()
        if k
        in {
            "voice_agent.mcp_servers.docs",
            "voice_agent.rag",
            "voice_agent.rag.docindex",
            "voice_agent.rag.fts5",
            "voice_agent.rag.backends",
            "voice_agent.rag.hybrid",
            "fastmcp",
            "mcp.server.fastmcp",
            "numpy",
        }
    }
    return payload


def _loaded_forbidden(loaded: list[str]) -> list[str]:
    return [
        mod
        for mod in loaded
        if any(mod == bad or mod.startswith(f"{bad}.") for bad in FORBIDDEN_MODULES)
    ]


def test_retrieval_only_path_leaves_heavy_modules_out_of_sys_modules(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".rag_cache"
    code = f"""
import json, sys, time
from pathlib import Path

def rss_kb():
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return -1

t0 = time.perf_counter()
import voice_agent.rag.docindex
import voice_agent.rag.fts5
import voice_agent.rag.backends
import_wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
rss_after_import_kb = rss_kb()

factory = voice_agent.rag.backends.make_retriever_factory(
    mode="fts5",
    cache_dir=Path({str(cache_dir)!r}),
)
idx = voice_agent.rag.docindex.DocIndex(
    Path({str(CORPUS)!r}),
    retriever_factory=factory,
    canonical_cache_dir=None,
)
idx.refresh()
hits = idx.search("daily meal allowance expense claims", k=4)
close_fn = getattr(idx.retriever, "close", None)
if callable(close_fn):
    close_fn()
rss_after_search_kb = rss_kb()

print(json.dumps({{
    "import_wall_ms": import_wall_ms,
    "rss_after_import_kb": rss_after_import_kb,
    "rss_after_search_kb": rss_after_search_kb,
    "hit_sources": [h["source"] for h in hits],
    "loaded_modules": sorted(sys.modules),
}}))
"""
    payload = _run_import_probe(code)
    summary = {k: v for k, v in payload.items() if k != "loaded_modules"}
    print("RETRIEVAL_ONLY_PROBE:", json.dumps(summary))
    hit_sources = payload["hit_sources"]
    assert isinstance(hit_sources, list) and hit_sources
    assert hit_sources[0] == "handbook.pdf"

    loaded = payload["loaded_modules"]
    assert isinstance(loaded, list)
    assert _loaded_forbidden(loaded) == []


def test_docs_server_import_leaves_heavy_modules_out_of_sys_modules() -> None:
    code = """
import json, sys, time
from pathlib import Path

def rss_kb():
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return -1

t0 = time.perf_counter()
import voice_agent.mcp_servers.docs
import_wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
rss_after_import_kb = rss_kb()

print(json.dumps({
    "import_wall_ms": import_wall_ms,
    "rss_after_import_kb": rss_after_import_kb,
    "loaded_modules": sorted(sys.modules),
}))
"""
    payload = _run_import_probe(code)
    summary = {k: v for k, v in payload.items() if k != "loaded_modules"}
    print("DOCS_IMPORT_PROBE:", json.dumps(summary))
    loaded = payload["loaded_modules"]
    assert isinstance(loaded, list)
    assert _loaded_forbidden(loaded) == []


def test_rag_package_lazy_exports_and_all() -> None:
    code = """
import json, sys
import voice_agent.rag as rag

all_names = list(rag.__all__)
dir_names = dir(rag)
after_pkg = sorted(sys.modules)

from voice_agent.rag import (
    Bm25Retriever,
    Chunk,
    DocIndex,
    FileRecord,
    RefreshReport,
    Retriever,
    make_retriever_factory,
    resolve_mode,
)
after_lean_from = sorted(sys.modules)

factory = make_retriever_factory(mode="bm25")
mode = resolve_mode("fts5", {})
hybrid_cls = rag.HybridRetriever

try:
    _ = rag.does_not_exist_attr
    missing_raised = False
except AttributeError:
    missing_raised = True

print(json.dumps({
    "all_names": all_names,
    "all_in_dir": all(name in dir_names for name in all_names),
    "after_pkg": after_pkg,
    "after_lean_from": after_lean_from,
    "bm25_name": Bm25Retriever.__name__,
    "chunk_name": Chunk.__name__,
    "docindex_name": DocIndex.__name__,
    "filerecord_name": FileRecord.__name__,
    "refresh_name": RefreshReport.__name__,
    "retriever_name": Retriever.__name__,
    "factory_is_bm25": factory is Bm25Retriever,
    "mode": mode,
    "hybrid_name": hybrid_cls.__name__,
    "hybrid_loaded_after_access": "voice_agent.rag.hybrid" in sys.modules,
    "missing_raised": missing_raised,
}))
"""
    payload = _run_import_probe(code)
    assert payload["all_names"] == EXPECTED_RAG_ALL
    assert payload["all_in_dir"] is True
    assert _loaded_forbidden(payload["after_pkg"]) == []  # type: ignore[arg-type]
    assert _loaded_forbidden(payload["after_lean_from"]) == []  # type: ignore[arg-type]
    assert payload["factory_is_bm25"] is True
    assert payload["mode"] == "fts5"
    assert payload["hybrid_name"] == "HybridRetriever"
    assert payload["hybrid_loaded_after_access"] is True
    assert payload["missing_raised"] is True
