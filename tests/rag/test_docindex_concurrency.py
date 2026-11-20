import threading
import time
from pathlib import Path

import pytest

from voice_agent.rag import docindex
from voice_agent.rag.docindex import DocIndex


def _write(folder: Path, i: int) -> None:
    (folder / f"doc{i}.md").write_text(f"# Doc {i}\n\nalpha beta gamma token{i} shared words\n")


def test_search_is_consistent_while_files_change(tmp_path: Path) -> None:
    for i in range(3):
        _write(tmp_path, i)
    index = DocIndex(tmp_path, canonical_cache_dir=None)
    index.refresh()
    stop = threading.Event()
    errors: list[BaseException] = []

    def mutate() -> None:
        try:
            n = 0
            while not stop.is_set():
                path = tmp_path / f"extra{n % 4}.md"
                if path.exists():
                    path.unlink()
                else:
                    _write(tmp_path, 10 + n % 4)
                    path.write_text(f"# Extra\n\nalpha extra{n % 4} beta\n")
                index.refresh()
                n += 1
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=mutate)
    worker.start()
    try:
        for _ in range(300):
            hits = index.search("alpha beta", k=4)
            chunks = index.chunks
            assert hits
            for hit in hits:
                assert hit["text"]
            assert all(c.text for c in chunks)
    finally:
        stop.set()
        worker.join(10)
    assert not errors


def test_search_never_pairs_chunks_with_a_different_retriever(tmp_path: Path) -> None:
    class Probe(docindex.Bm25Retriever):
        def __init__(self) -> None:
            super().__init__()
            self.built_for = -1

        def build(self, chunks):  # type: ignore[no-untyped-def]
            super().build(chunks)
            time.sleep(0.002)
            self.built_for = len(chunks)

    mismatches: list[tuple[int, int]] = []
    holder: list[DocIndex] = []

    class Checked(Probe):
        def rank(self, query, k):  # type: ignore[no-untyped-def]
            if self.built_for != len(holder[0]._chunks):
                mismatches.append((self.built_for, len(holder[0]._chunks)))
            return super().rank(query, k)

    for i in range(3):
        _write(tmp_path, i)
    index = DocIndex(tmp_path, retriever_factory=Checked, canonical_cache_dir=None)
    holder.append(index)
    index.refresh()
    stop = threading.Event()

    def mutate() -> None:
        while not stop.is_set():
            path = tmp_path / "flip.md"
            if path.exists():
                path.unlink()
            else:
                path.write_text("# Flip\n\nalpha beta flipped\n")
            index.refresh()

    worker = threading.Thread(target=mutate)
    worker.start()
    try:
        for _ in range(300):
            assert index.search("alpha beta flipped", k=8)
    finally:
        stop.set()
        worker.join(10)
    assert mismatches == []


def test_parsing_reports_file_during_slow_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "slow.md").write_text("# Slow\n\nsome text here\n")
    real = docindex._load_chunks_for_mode
    entered = threading.Event()
    release = threading.Event()

    def slow(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        entered.set()
        release.wait(5)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(docindex, "_load_chunks_for_mode", slow)
    index = DocIndex(tmp_path, canonical_cache_dir=None)
    assert index.parsing() is None
    t = threading.Thread(target=index.refresh)
    t.start()
    assert entered.wait(5)
    assert index.parsing() == "slow.md"
    assert index.search("some text") == []
    release.set()
    t.join(5)
    assert index.parsing() is None
    assert index.search("some text")


def test_concurrent_refreshes_serialise_without_double_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.md").write_text("# A\n\nsome text here\n")
    real = docindex._load_chunks_for_mode
    calls: list[float] = []

    def counting(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls.append(time.monotonic())
        time.sleep(0.3)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(docindex, "_load_chunks_for_mode", counting)
    index = DocIndex(tmp_path, canonical_cache_dir=None)
    done: list[float] = []

    def run() -> None:
        index.refresh()
        done.append(time.monotonic())

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(calls) == 1
    assert len(done) == 2
    assert max(done) - min(done) >= 0.0
