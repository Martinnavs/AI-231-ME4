"""Selection-option tests: per_file_cap, adaptive_cap, whole_files (ticket 204)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from voice_agent.rag.dbindex import DbIndex
from voice_agent.rag.docindex import (
    Chunk,
    DocIndex,
    select_results,
)
from voice_agent.rag.ingest import ingest


def _mk(source: str, position: int, text: str) -> Chunk:
    """Build a minimal Chunk for unit tests."""
    return Chunk(
        id=f"{source}#{position}",
        file_sha256="x",
        chunk_sha256=hashlib.sha256(text.encode()).hexdigest(),
        position=position,
        page=None,
        section="",
        file_date="2026-01-01",
        source=source,
        header=source,
        text=text,
    )


def _ranked(chunks_and_scores: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
    return list(chunks_and_scores)


class TestDefaultsByteIdentical:
    """Defaults (no options) return identical results to the current select_results."""

    @pytest.fixture(scope="class")
    def fixture_chunks(self, tmp_path_factory: pytest.TempPathFactory) -> list[Chunk]:
        src = tmp_path_factory.mktemp("src")
        for i in range(6):
            (src / f"file{i}.txt").write_text(
                "\n\n".join(
                    f"Section {j} of file {i} covers topic {j * 10 + i}.\n" for j in range(8)
                ),
                encoding="utf-8",
            )
        ingest(src, src / "idx.db")
        dbi = DbIndex(src / "idx.db")
        chunks = list(dbi.chunks)
        dbi.close()
        return chunks

    def test_defaults_identical_select(self, fixture_chunks: list[Chunk]) -> None:
        scored = [(c, float(10 - i)) for i, c in enumerate(fixture_chunks)]
        k = 8
        q_tokens = {"section", "topic", "covers"}
        results_new, filled_new = select_results(scored, k, q_tokens)
        results_old, filled_old = select_results(
            scored,
            k,
            q_tokens,
            per_file_cap=None,
            adaptive_cap=False,
            whole_files=0,
            whole_max_chunks=40,
        )
        assert results_new == results_old
        assert filled_new == filled_old

    def test_defaults_on_index(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        for i in range(4):
            (src / f"doc{i}.txt").write_text(
                "\n\n".join(f"Paragraph {j} discusses value {j * 100 + i}.\n" for j in range(5)),
                encoding="utf-8",
            )
        ingest(src, tmp_path / "idx.db")

        from voice_agent.rag.backends import _make_lexical_factory

        factory = _make_lexical_factory("fts5", cache_dir=None, fts5_tokenizer="pretok")
        di = DocIndex(src, retriever_factory=factory)
        di.refresh()
        dbi = DbIndex(tmp_path / "idx.db")

        queries = [
            "value 100",
            "value 201",
            "value 302",
            "value 403",
            "paragraph discusses",
            "value",
        ]
        for q in queries:
            doc_r = di.search(q, k=4)
            db_r = dbi.search(q, k=4)
            assert doc_r == db_r, f"query={q!r}: mismatch"

        dbi.close()


class TestPerFileCap:
    """per_file_cap overrides the built-in cap."""

    def test_cap_4_recovers_rank_5(self) -> None:
        """Fact at raw rank 5 of a file with 4 chunks — cap=4 loses it, cap=6 recovers."""
        ranked: list[tuple[Chunk, float]] = []
        for i in range(10):
            ranked.append((_mk("file_a.txt", i, f"chunk content {i} of file_a"), float(20 - i)))
        for i in range(8):
            ranked.append((_mk(f"other{i}.txt", 0, f"other content {i}"), float(9 - i)))

        k = 8
        q_tokens = {"chunk", "content"}

        res_default, _ = select_results(ranked, k, q_tokens)
        file_a_default = sum(1 for r in res_default if r["source"] == "file_a.txt")

        res_cap6, _ = select_results(ranked, k, q_tokens, per_file_cap=6)
        file_a_cap6 = sum(1 for r in res_cap6 if r["source"] == "file_a.txt")

        assert file_a_default < file_a_cap6

    def test_cap_4_exactly(self) -> None:
        """per_file_cap=4 means soft=4, hard=8."""
        scored: list[tuple[Chunk, float]] = [
            (_mk("only.txt", i, f"item {i}"), float(10 - i)) for i in range(12)
        ]
        res, _ = select_results(scored, k=6, q_tokens={"item"}, per_file_cap=4)
        assert len(res) == 6

    def test_cap_lower_than_default(self) -> None:
        """per_file_cap=1 enforces initial cap; overflow fills the gap."""
        scored: list[tuple[Chunk, float]] = []
        for src in ["a.txt", "b.txt"]:
            for i in range(6):
                scored.append((_mk(src, i, f"data {src} item {i}"), float(10 - i * 0.5)))

        # k=6, cap=1: soft=hard=1, 1 from each via select, rest from overflow
        res1, filled1 = select_results(scored, k=6, q_tokens={"data", "item"}, per_file_cap=1)
        # k=6, default cap: soft=2, hard=4
        res2, filled2 = select_results(scored, k=6, q_tokens={"data", "item"})

        # Both reach k=6; cap=1 fills via overflow after initial 1 per source
        assert len(res1) == 6
        assert filled1 is False  # needed overflow to fill
        assert filled2 is True  # default fills without overflow


class TestAdaptiveCap:
    """adaptive_cap expands the file that owns the top-ranked chunk."""

    def test_adaptive_expands_top_file(self) -> None:
        """When top file has >= 3 of top-10, it may hold up to k-2."""
        ranked: list[tuple[Chunk, float]] = []
        for i in range(5):
            ranked.append((_mk("dominant.txt", i, f"dominant chunk {i}"), float(20 - i)))
        for i in range(5):
            ranked.append((_mk(f"minor{i}.txt", 0, f"minor {i}"), float(14 - i)))

        k = 8
        q_tokens = {"chunk"}

        res_off, _ = select_results(ranked, k, q_tokens, adaptive_cap=False)
        dom_off = sum(1 for r in res_off if r["source"] == "dominant.txt")

        res_on, _ = select_results(ranked, k, q_tokens, adaptive_cap=True)
        dom_on = sum(1 for r in res_on if r["source"] == "dominant.txt")

        assert dom_off < dom_on

    def test_adaptive_no_expand_when_not_top_3(self) -> None:
        """When no file has >= 3 of top-10, adaptive has no effect."""
        ranked: list[tuple[Chunk, float]] = []
        for i in range(5):
            ranked.append((_mk(f"spread{i}.txt", 0, f"spread chunk {i}"), float(20 - i)))
        for i in range(6):
            ranked.append((_mk(f"extra{i}.txt", 0, f"extra {i}"), float(14 - i)))

        res_off, _ = select_results(ranked, k=8, q_tokens={"chunk"})
        res_on, _ = select_results(ranked, k=8, q_tokens={"chunk"}, adaptive_cap=True)
        assert res_off == res_on


class TestWholeFiles:
    """whole_files adds all chunks of small top files."""

    def test_whole_includes_all_small_file_chunks(self) -> None:
        """A file with <= 40 chunks gets all its chunks added."""
        small = [_mk("small.txt", i, f"small line {i}") for i in range(5)]
        other = [_mk("other.txt", i, f"other line {i}") for i in range(2)]

        ranked: list[tuple[Chunk, float]] = [
            (small[0], 20.0),
            (other[0], 19.0),
            (small[1], 18.0),
            (other[1], 17.0),
            (small[2], 16.0),
            (small[3], 15.0),
            (small[4], 14.0),
        ]

        res, _ = select_results(ranked, k=3, q_tokens={"small"}, whole_files=1)
        small_ids = {r["id"] for r in res if r["source"] == "small.txt"}
        expected_ids = {f"small.txt#{i}" for i in range(5)}
        assert small_ids == expected_ids

    def test_whole_skips_large_file(self) -> None:
        """A file with > 40 chunks is skipped."""
        big = [_mk("big.txt", i, f"big line {i}") for i in range(45)]
        tiny = [_mk("tiny.txt", 0, "tiny content")]

        ranked: list[tuple[Chunk, float]] = [
            (big[0], 20.0),
            (tiny[0], 19.0),
        ] + [(c, float(18 - i)) for i, c in enumerate(big[1:])]

        res, _ = select_results(ranked, k=2, q_tokens={"big"}, whole_files=1)
        big_in_res = [r for r in res if r["source"] == "big.txt"]
        assert len(big_in_res) <= 1

    def test_whole_no_duplicates(self) -> None:
        """Chunks already in the k selection are not duplicated."""
        chunks = [_mk("dup.txt", i, f"dup line {i}") for i in range(3)]
        ranked: list[tuple[Chunk, float]] = [
            (chunks[0], 20.0),
            (chunks[1], 19.0),
            (chunks[2], 18.0),
        ]

        res, _ = select_results(ranked, k=3, q_tokens={"dup"}, whole_files=1)
        ids = [r["id"] for r in res]
        assert len(ids) == len(set(ids))
        assert len(ids) == 3

    def test_whole_files_in_document_order(self) -> None:
        """Added chunks appear in position (document) order."""
        chunks = [_mk("order.txt", i, f"order line {i}") for i in range(5)]
        ranked: list[tuple[Chunk, float]] = [
            (chunks[4], 20.0),
            (chunks[0], 19.0),
            (chunks[2], 18.0),
            (chunks[1], 17.0),
            (chunks[3], 16.0),
        ]

        res, _ = select_results(ranked, k=1, q_tokens={"order"}, whole_files=1)
        order_ids = [r["id"] for r in res]
        assert order_ids[0] == "order.txt#4"
        assert order_ids[1:] == [f"order.txt#{i}" for i in range(4)]


class TestDbIndexDocIndexParity:
    """DbIndex and DocIndex give identical results for every option combination."""

    @pytest.fixture(scope="class")
    def parity_index(self, tmp_path_factory: pytest.TempPathFactory):
        src = tmp_path_factory.mktemp("src")
        for i in range(3):
            (src / f"parity{i}.txt").write_text(
                "\n\n".join(f"P {j} says val {j * 10 + i}.\n" for j in range(4)),
                encoding="utf-8",
            )
        db_dir = tmp_path_factory.mktemp("db")
        db_path = db_dir / "p.db"
        ingest(src, db_path)
        return src, db_path

    @pytest.mark.parametrize("k", [2, 8])
    @pytest.mark.parametrize("per_file_cap", [None, 4])
    @pytest.mark.parametrize("adaptive_cap", [False, True])
    @pytest.mark.parametrize("whole_files", [0, 1])
    def test_parity(
        self,
        parity_index: tuple[Path, Path],
        k: int,
        per_file_cap: int | None,
        adaptive_cap: bool,
        whole_files: int,
    ) -> None:
        src, db_path = parity_index
        from voice_agent.rag.backends import _make_lexical_factory

        factory = _make_lexical_factory("fts5", cache_dir=None, fts5_tokenizer="pretok")
        di = DocIndex(src, retriever_factory=factory)
        di.refresh()
        dbi = DbIndex(db_path)

        query = "val says"
        kwargs = dict(
            k=k,
            per_file_cap=per_file_cap,
            adaptive_cap=adaptive_cap,
            whole_files=whole_files,
        )
        doc_r = di.search(query, **kwargs)  # type: ignore[arg-type]
        db_r = dbi.search(query, **kwargs)  # type: ignore[arg-type]
        assert doc_r == db_r, f"k={k} cap={per_file_cap} adapt={adaptive_cap} whole={whole_files}"

        dbi.close()


class TestAblateScript:
    """Run the ablation script on a tiny fixture and assert counts."""

    def test_ablate_tiny_fixture(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "budget.txt").write_text(
            "The Q4 budget was 1.2 million pesos approved in October 2026.\n"
            "The team headcount is 45 people across three offices.\n"
            "Revenue grew 12 percent year over year.\n",
            encoding="utf-8",
        )
        (corpus / "policy.md").write_text(
            "# Leave Policy\n\nEmployees get 15 days of annual leave.\n"
            "## Sick Leave\n\nSick leave is 10 days per year with a medical certificate.\n",
            encoding="utf-8",
        )
        (corpus / "small.txt").write_text(
            "Project kickoff date is January 15 2027.\n",
            encoding="utf-8",
        )

        db_path = tmp_path / "corpus.db"
        ingest(corpus, db_path)

        qset = tmp_path / "q.jsonl"
        questions = [
            {
                "id": "q1",
                "q": "What is the Q4 budget?",
                "answerable": True,
                "relevant": [{"contains": "1.2 million pesos"}],
            },
            {
                "id": "q2",
                "q": "How many annual leave days?",
                "answerable": True,
                "relevant": [{"contains": "15 days of annual leave"}],
            },
            {
                "id": "q3",
                "q": "When is the kickoff?",
                "answerable": True,
                "relevant": [{"contains": "January 15 2027"}],
            },
            {
                "id": "q4",
                "q": "How many people in the team?",
                "answerable": True,
                "relevant": [{"contains": "45 people"}],
            },
        ]
        qset.write_text("\n".join(json.dumps(q) for q in questions) + "\n", encoding="utf-8")

        out = tmp_path / "ablate_out.json"
        env = {"PYTHONPATH": f"{Path.cwd()}/src", "TMPDIR": str(tmp_path)}
        result = subprocess.run(
            [
                sys.executable,
                "scripts/me4_selection_ablate.py",
                "--db",
                str(db_path),
                "--questions",
                str(qset),
                "--out",
                str(out),
                "--set-name",
                "tiny",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )

        assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
        assert out.exists(), "output JSON not created"

        data = json.loads(out.read_text())
        assert "sets" in data
        tiny = data["sets"]["tiny"]
        assert tiny["answerable"] == 4
        assert tiny["variants"]["base"]["fact@8"] == 4
