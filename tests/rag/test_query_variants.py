"""Tests for query variants and retrieval ablation."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from voice_agent.rag import query as _query_mod
from voice_agent.rag.docindex import Chunk, select_results
from voice_agent.rag.fts5 import _build_match_query

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunk(id_: str, source: str, header: str, text: str, position: int = 0):
    return Chunk(
        id=id_,
        source=source,
        header=header,
        text=text,
        position=position,
        file_sha256="x",
        chunk_sha256="y",
        page=None,
        section="",
        file_date="2026-01-15",
    )


@pytest.fixture
def sample_questions() -> list[str]:
    """Return questions from the heldout dev set for baseline identity check."""
    dev_path = Path.home() / "rag-testsets" / "heldout" / "dev.jsonl"
    questions: list[str] = []
    if dev_path.exists():
        for line in dev_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                questions.append(row.get("q", ""))
    # Pad with synthetic questions if we don't have the dev set
    if len(questions) < 200:
        base = [
            "What is the security incident notification time?",
            "According to the documents, what Advisory Circular covers",
            " Aeronautical Decision Making?",
            "What school grade level should everyday public communications aim for?",
            "What is the estimated population of Oregon in 2024?",
            "Which platform hosts the class timetable if left out of the syllabus?",
            "How does the Module 2 rationale characterise toxicity?",
            "What numbering do preliminary pages use?",
            "What encoding must JSON text use?",
            "How large must the top margin be?",
            "Which grades may graduate courses not use?",
        ]
        for i in range(250):
            base_q = base[i % len(base)]
            questions.append(f"Query {i + 1}: {base_q}")
    return questions[:250]


@pytest.fixture
def fixture_chunks() -> list[Chunk]:
    """6-file test fixture where cue word 'documents' pollutes baseline."""
    return [
        _make_chunk(
            "f1#c1",
            "file1.md",
            "file1.md > Intro",
            "The documents cover security policies and procedures.",
            0,
        ),
        _make_chunk(
            "f1#c2",
            "file1.md",
            "file1.md > Details",
            "Security incidents must be reported within one hour.",
            1,
        ),
        _make_chunk(
            "f2#c1",
            "file2.pdf",
            "file2.pdf > p. 1",
            "The budget for 2025 is allocated for training.",
            0,
        ),
        _make_chunk(
            "f2#c2", "file2.pdf", "file2.pdf > p. 2", "Training programs start in March.", 1
        ),
        _make_chunk("f3#c1", "file3.csv", "file3.csv", "col_1: name; col_2: score; Alice: 95", 0),
        _make_chunk(
            "f4#c1",
            "file4.txt",
            "file4.txt",
            "The system requires authentication before access.",
            0,
        ),
    ]


# ---------------------------------------------------------------------------
# Baseline byte-identity
# ---------------------------------------------------------------------------


class TestBaselineIdentity:
    """Baseline variant must produce byte-identical output to _build_match_query."""

    def test_baseline_single_question(self) -> None:
        q = "What is the security incident notification time?"
        expected = _build_match_query(q)
        steps = _query_mod.build_query(q, variant="baseline")
        assert len(steps) == 1
        assert steps[0].match == expected

    def test_baseline_empty_query(self) -> None:
        steps = _query_mod.build_query("", variant="baseline")
        assert len(steps) == 0

    def test_baseline_200_questions(self, sample_questions: list[str]) -> None:
        """Baseline must be byte-identical for 200+ real questions."""
        qs = sample_questions[: max(200, len(sample_questions))]
        for q in qs:
            expected = _build_match_query(q)
            steps = _query_mod.build_query(q, variant="baseline")
            if expected:
                assert len(steps) == 1
                assert steps[0].match == expected, f"Mismatch for: {q[:60]}"
            else:
                assert len(steps) == 0

    def test_baseline_env_default(self, sample_questions: list[str]) -> None:
        """Default env (baseline) must match _build_match_query."""
        old = os.environ.get("VOICE_AGENT_RAG_QUERY")
        os.environ.pop("VOICE_AGENT_RAG_QUERY", None)
        try:
            q = sample_questions[0] if sample_questions else "test query"
            expected = _build_match_query(q)
            steps = _query_mod.build_query(q)  # uses env default
            if expected:
                assert steps[0].match == expected
        finally:
            if old is not None:
                os.environ["VOICE_AGENT_RAG_QUERY"] = old


# ---------------------------------------------------------------------------
# Variant tests
# ---------------------------------------------------------------------------


class TestQ1CueStrip:
    """q1 removes cue/stop words but keeps >= 2 terms."""

    def test_removes_cue_words(self) -> None:
        q = "According to the documents, what is the security time?"
        steps = _query_mod.build_query(q, variant="q1")
        match = steps[0].match
        # "documents", "according", "the", "what", "is" should be stripped
        assert '"documents"' not in match
        assert '"according"' not in match
        # "security", "time" should remain
        assert '"security"' in match
        assert '"time"' in match

    def test_keeps_original_when_few_terms(self) -> None:
        """If stripping leaves < 2 terms, keep original."""
        q = "What are the documents?"
        steps = _query_mod.build_query(q, variant="q1")
        baseline = _build_match_query(q)
        assert steps[0].match == baseline

    def test_q1_on_polluted_query(self) -> None:
        """Cue word 'documents' should not pollute the query."""
        q = "According to the documents, what is the notification time for security incidents?"
        steps = _query_mod.build_query(q, variant="q1")
        match = steps[0].match
        assert '"documents"' not in match
        assert '"security"' in match
        assert '"time"' in match


class TestQ2IDFPrune:
    """q2 drops high-df terms; never leaves < 2."""

    def test_q2_without_df_table(self) -> None:
        """Without df data, q2 passes through."""
        q = "What is the security time?"
        steps = _query_mod.build_query(q, variant="q2")
        baseline = _build_match_query(q)
        assert steps[0].match == baseline

    def test_q2_keeps_rarest(self) -> None:
        """q2 keeps terms below the df threshold."""
        df_table = {
            "what": 100.0,
            "is": 100.0,
            "the": 100.0,
            "security": 5.0,
            "incident": 3.0,
            "time": 10.0,
        }
        q = "What is the security incident time?"
        steps = _query_mod.build_query(q, variant="q2", df_table=df_table)
        match = steps[0].match
        # Common terms should be dropped
        assert '"what"' not in match
        assert '"the"' not in match
        # Rare terms should remain
        assert '"security"' in match or '"incident"' in match

    def test_q2_never_below_two(self) -> None:
        """q2 must keep at least 2 terms."""
        df_table = {t: 100.0 for t in ["a", "b", "c", "d"]}
        q = "a b c d"
        steps = _query_mod.build_query(q, variant="q2", df_table=df_table)
        match = steps[0].match
        terms = re.findall(r'"([^"]+)"', match)
        assert len(terms) >= 2


class TestQ3Tiered:
    """q3 returns two steps: AND of rarest 3, then OR."""

    def test_q3_two_steps(self) -> None:
        q = "What is the security incident notification time for reports?"
        steps = _query_mod.build_query(q, variant="q3")
        assert len(steps) == 2
        assert steps[0].is_and is True
        assert steps[1].is_and is False

    def test_q3_single_step_few_terms(self) -> None:
        """With < 3 terms, q3 returns a single OR step."""
        q = "What time?"
        steps = _query_mod.build_query(q, variant="q3")
        assert len(steps) == 1
        assert steps[0].is_and is False

    def test_q3_and_has_three_terms(self) -> None:
        steps = _query_mod.build_query(
            "security incident notification protocol requirements", variant="q3"
        )
        and_match = steps[0].match
        terms = re.findall(r'"([^"]+)"', and_match)
        assert len(terms) == 3


class TestQ4Phrases:
    """q4 adds bigram phrases."""

    def test_q4_adds_phrases(self) -> None:
        q = "What is the security incident notification time?"
        steps = _query_mod.build_query(q, variant="q4")
        match = steps[0].match
        assert '"security incident"' in match or '"incident notification"' in match

    def test_q4_valid_fts5_syntax(self) -> None:
        """q4 output must be valid FTS5."""
        q = 'What "quoted" term: with-hyphens?'
        steps = _query_mod.build_query(q, variant="q4")
        match = steps[0].match
        # FTS5 requires balanced quotes
        assert match.count('"') % 2 == 0


class TestComposites:
    """Composite variants (q1+q2, q1+q3, etc.)."""

    def test_q1_q2_composite(self) -> None:
        steps = _query_mod.build_query(
            "According to the documents, what is the security time?",
            variant="q1+q2",
        )
        match = steps[0].match
        assert '"documents"' not in match

    def test_q1_q3_composite(self) -> None:
        steps = _query_mod.build_query(
            "According to the documents, what is the security incident time for reports?",
            variant="q1+q3",
        )
        assert len(steps) == 2
        assert steps[0].is_and is True

    def test_q1_q4_composite(self) -> None:
        steps = _query_mod.build_query(
            "According to the documents, what is the security incident time?",
            variant="q1+q4",
        )
        match = steps[0].match
        assert '"documents"' not in match


class TestFileScoping:
    """q5 file scoping."""

    def test_get_scoped_files(self) -> None:
        files = ["budget/plan.xlsx", "notes/report.md", "data/raw.csv"]
        q = "What is in the plan spreadsheet?"
        scoped = _query_mod.get_scoped_files(q, files)
        assert "budget/plan.xlsx" in scoped

    def test_no_match(self) -> None:
        files = ["budget.xlsx"]
        q = "What is the capital of France?"
        scoped = _query_mod.get_scoped_files(q, files)
        assert len(scoped) == 0


class TestSelectResults:
    """select_results with per_file_cap."""

    def test_default_caps(self, fixture_chunks: list[Chunk]) -> None:
        ranked = [(c, 1.0 - i * 0.1) for i, c in enumerate(fixture_chunks)]
        results, filled = select_results(ranked, 4, set())
        assert len(results) <= 4

    def test_per_file_cap_override(self, fixture_chunks: list[Chunk]) -> None:
        """per_file_cap=4 should allow more from same file."""
        # All from same file
        same = [_make_chunk(f"same#c{i}", "same.md", "h", f"text {i}", i) for i in range(6)]
        ranked = [(c, 1.0 - i * 0.1) for i, c in enumerate(same)]
        # Default cap=2
        r1, _ = select_results(ranked, 4, set())
        # Cap=4
        r2, _ = select_results(ranked, 4, set(), per_file_cap=4)
        assert len(r2) >= len(r1)


class TestEnvVariable:
    """VOICE_AGENT_RAG_QUERY selects the active variant."""

    def test_env_sets_variant(self) -> None:
        old = os.environ.get("VOICE_AGENT_RAG_QUERY")
        os.environ["VOICE_AGENT_RAG_QUERY"] = "q1"
        try:
            q = "According to the documents, what is the time?"
            steps = _query_mod.build_query(q)  # no explicit variant
            match = steps[0].match
            assert '"documents"' not in match
        finally:
            if old is not None:
                os.environ["VOICE_AGENT_RAG_QUERY"] = old
            else:
                os.environ.pop("VOICE_AGENT_RAG_QUERY", None)

    def test_env_baseline(self) -> None:
        old = os.environ.get("VOICE_AGENT_RAG_QUERY")
        os.environ["VOICE_AGENT_RAG_QUERY"] = "baseline"
        try:
            q = "What is the time?"
            steps = _query_mod.build_query(q)
            baseline = _build_match_query(q)
            assert steps[0].match == baseline
        finally:
            if old is not None:
                os.environ["VOICE_AGENT_RAG_QUERY"] = old
            else:
                os.environ.pop("VOICE_AGENT_RAG_QUERY", None)
