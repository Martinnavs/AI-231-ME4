"""Tests for soft/hard per-file chunk cap, score gate, and cross-file dedup."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from voice_agent.rag import docindex
from voice_agent.rag.docindex import (
    DEDUP_JACCARD,
    HARD_CHUNKS_PER_FILE,
    MAX_CHUNKS_PER_FILE,
    OVERFLOW_SCORE_RATIO,
    Chunk,
    DocIndex,
)
from voice_agent.rag.metrics import load_golden

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "rag_corpus"
GOLDEN_PATH = REPO_ROOT / "config" / "eval" / "rag_golden.jsonl"


class _FixedScoreRetriever:
    """Fake Retriever returning caller-specified scores keyed by chunk ID."""

    def __init__(self, scores_by_id: dict[str, float]) -> None:
        self._scores_by_id = scores_by_id
        self._chunks: list[Chunk] = []

    def build(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        ranked = [
            (idx, self._scores_by_id[chunk.id])
            for idx, chunk in enumerate(self._chunks)
            if chunk.id in self._scores_by_id
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:k]


def _write_long_md(path: Path, sections: list[tuple[str, str]]) -> None:
    body = "\n\n".join(f"# {title}\n\n{text}" for title, text in sections)
    path.write_text(body + "\n", encoding="utf-8")


def test_cap_constants_match_spec() -> None:
    assert MAX_CHUNKS_PER_FILE == 2
    assert HARD_CHUNKS_PER_FILE == 4
    assert OVERFLOW_SCORE_RATIO == 0.75
    assert DEDUP_JACCARD == 0.75


def test_soft_cap_holds_when_third_chunk_is_below_score_gate(tmp_path: Path) -> None:
    _write_long_md(
        tmp_path / "long.md",
        [
            ("First", "Zephyr turbine inspection checklist covers rotor blades."),
            ("Second", "Zephyr turbine oil pressure gauge sits near the valve."),
            ("Third", "Zephyr backup gasket dimensions and storage bin label."),
            ("Fourth", "Zephyr mounting bracket torque chart for annual service."),
        ],
    )
    _write_long_md(
        tmp_path / "other.md",
        [
            ("Alpha", "Zephyr field calibration notes for coastal stations."),
            ("Beta", "Zephyr spare fuse inventory for southern relay."),
        ],
    )

    scores = {
        "long.md#0": 10.0,
        "long.md#1": 9.0,
        "long.md#2": 7.49,
        "long.md#3": 7.0,
        "other.md#0": 6.0,
        "other.md#1": 5.0,
    }
    idx = DocIndex(tmp_path, retriever_factory=lambda: _FixedScoreRetriever(scores))
    idx.refresh()

    hits = idx.search("zephyr", k=4)
    assert [h["id"] for h in hits] == ["long.md#0", "long.md#1", "other.md#0", "other.md#1"]

    # Default BM25 retriever: 3rd/4th chunks in long.md score well below 0.75 * best.
    _write_long_md(
        tmp_path / "long.md",
        [
            ("One", "Quasar beacon frequency calibration."),
            ("Two", "Quasar beacon antenna alignment."),
            (
                "Three",
                "Quasar general maintenance notes covering paint, hinges, latches, bolts, "
                "brackets, cleaning rags, gloves, ladders, toolboxes, and logbooks.",
            ),
        ],
    )
    _write_long_md(
        tmp_path / "other.md",
        [
            ("Summary", "Quasar beacon battery replacement schedule."),
            ("Relay", "Quasar beacon field signal check."),
        ],
    )
    bm25_idx = DocIndex(tmp_path)
    bm25_idx.refresh()
    bm25_hits = bm25_idx.search("quasar beacon", k=4)
    sources = [h["source"] for h in bm25_hits]
    assert sources.count("long.md") == 2
    assert sources.count("other.md") == 2


def test_third_and_fourth_chunks_appear_when_clearing_score_gate(tmp_path: Path) -> None:
    _write_long_md(
        tmp_path / "long.md",
        [
            ("Sec0", "Telemetry uplink packet format header field."),
            ("Sec1", "Telemetry downlink acknowledgement timing window."),
            ("Sec2", "Telemetry buffer overflow recovery procedure."),
            ("Sec3", "Telemetry checksum verification polynomial table."),
            ("Sec4", "Telemetry archive rotation schedule monthly."),
        ],
    )
    _write_long_md(
        tmp_path / "other.md",
        [
            ("Misc", "Telemetry ground station roof access key."),
        ],
    )

    scores = {
        "long.md#0": 10.0,
        "long.md#1": 9.0,
        "long.md#2": 8.0,
        "long.md#3": 7.5,  # exactly OVERFLOW_SCORE_RATIO * 10.0
        "other.md#0": 7.0,
    }
    idx = DocIndex(tmp_path, retriever_factory=lambda: _FixedScoreRetriever(scores))
    idx.refresh()

    hits = idx.search("telemetry", k=4)
    assert [h["id"] for h in hits] == [
        "long.md#0",
        "long.md#1",
        "long.md#2",
        "long.md#3",
    ]

    # Default BM25 retriever: four equally strong distinct chunks from one file clear the gate.
    _write_long_md(
        tmp_path / "long.md",
        [
            ("Alpha", "Cobalt catalyst reactor pressure check."),
            ("Bravo", "Cobalt catalyst thermal loop valve."),
            ("Charlie", "Cobalt catalyst feedstock purity spec."),
            ("Delta", "Cobalt catalyst vessel drain procedure."),
        ],
    )
    _write_long_md(
        tmp_path / "other.md",
        [
            (
                "Background",
                "Cobalt general warehouse inventory list including pallets, crates, "
                "forklifts, gloves, aprons, goggles, drums, labels, and manifests.",
            ),
        ],
    )
    bm25_idx = DocIndex(tmp_path)
    bm25_idx.refresh()
    bm25_hits = bm25_idx.search("cobalt catalyst", k=4)
    assert [h["source"] for h in bm25_hits].count("long.md") >= 3


def test_hard_cap_of_four_holds(tmp_path: Path) -> None:
    _write_long_md(
        tmp_path / "long.md",
        [(f"Part{i}", f"Helios solar inverter diagnostic code {i}.") for i in range(6)],
    )
    _write_long_md(
        tmp_path / "other.md",
        [("Backup", "Helios solar inverter backup fuse specification.")],
    )

    scores = {
        "long.md#0": 10.0,
        "long.md#1": 9.5,
        "long.md#2": 9.0,
        "long.md#3": 8.5,
        "long.md#4": 8.0,
        "long.md#5": 7.8,
        "other.md#0": 5.0,
    }
    idx = DocIndex(tmp_path, retriever_factory=lambda: _FixedScoreRetriever(scores))
    idx.refresh()

    hits = idx.search("helios", k=5)
    assert [h["id"] for h in hits] == [
        "long.md#0",
        "long.md#1",
        "long.md#2",
        "long.md#3",
        "other.md#0",
    ]


def test_cross_file_near_duplicate_skipped_and_same_file_kept(tmp_path: Path) -> None:
    shared_body = (
        "Amber emergency generator startup requires checking oil level coolant pressure "
        "battery voltage fuel valve position breaker switch load transfer sequence."
    )
    _write_long_md(
        tmp_path / "doc_a.md",
        [
            ("Startup", shared_body),
            ("Repeat", shared_body),
        ],
    )
    _write_long_md(
        tmp_path / "doc_b.md",
        [
            ("Startup", shared_body + " verified."),
        ],
    )
    _write_long_md(
        tmp_path / "doc_c.md",
        [
            (
                "Cooling",
                "Amber radiator thermostat opens at eighty degrees during sustained load tests.",
            ),
        ],
    )

    scores = {
        "doc_a.md#0": 10.0,
        "doc_a.md#1": 9.5,
        "doc_b.md#0": 9.0,
        "doc_c.md#0": 6.0,
    }
    idx = DocIndex(tmp_path, retriever_factory=lambda: _FixedScoreRetriever(scores))
    idx.refresh()

    hits = idx.search("amber", k=3)
    assert [h["id"] for h in hits] == ["doc_a.md#0", "doc_a.md#1", "doc_c.md#0"]

    # Default BM25 retriever also skips doc_b.md#0 in favor of distinct doc_c.md#0.
    bm25_idx = DocIndex(tmp_path)
    bm25_idx.refresh()
    bm25_hits = bm25_idx.search("amber generator startup", k=3)
    bm25_ids = [h["id"] for h in bm25_hits]
    assert "doc_b.md#0" not in bm25_ids
    assert "doc_c.md#0" in bm25_ids


def test_non_positive_best_score_keeps_soft_cap(tmp_path: Path) -> None:
    _write_long_md(
        tmp_path / "long.md",
        [(f"Sec{i}", f"Boreal station weather log entry {i}.") for i in range(4)],
    )
    _write_long_md(
        tmp_path / "other.md",
        [
            ("North", "Boreal north anemometer reading."),
            ("South", "Boreal south barometer reading."),
        ],
    )

    for best in (0.0, -1.0):
        scores = {
            "long.md#0": best,
            "long.md#1": best,
            "long.md#2": best,
            "long.md#3": best,
            "other.md#0": best - 1.0,
            "other.md#1": best - 2.0,
        }
        idx = DocIndex(tmp_path, retriever_factory=lambda s=scores: _FixedScoreRetriever(s))
        idx.refresh()
        hits = idx.search("boreal", k=4)
        assert [h["id"] for h in hits] == [
            "long.md#0",
            "long.md#1",
            "other.md#0",
            "other.md#1",
        ]


def test_fixture_golden_rows_mostly_unchanged_when_the_gates_are_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    golden = load_golden(GOLDEN_PATH)
    idx = DocIndex(CORPUS_DIR)
    idx.refresh()
    answerable = [row for row in golden if row["type"] not in {"unanswerable", "off_topic"}]
    assert len(answerable) == 29

    gated = {
        str(row["id"]): [(h["source"], h["id"]) for h in idx.search(str(row["q"]), k=4)]
        for row in answerable
    }
    monkeypatch.setattr(docindex, "HARD_CHUNKS_PER_FILE", docindex.MAX_CHUNKS_PER_FILE)
    monkeypatch.setattr(docindex, "DEDUP_JACCARD", 2.0)
    plain = {
        str(row["id"]): [(h["source"], h["id"]) for h in idx.search(str(row["q"]), k=4)]
        for row in answerable
    }
    changed = sorted(rid for rid in gated if gated[rid] != plain[rid])
    # The gates intervene on a minority of the short-file fixtures (4 of 29 when written).
    assert len(changed) <= 6, changed
