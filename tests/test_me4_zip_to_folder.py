from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from me4_zip_to_folder import main


def test_dry_run_executes_nothing(tmp_path: Path, capsys) -> None:
    zpath = tmp_path / "one.zip"
    outdir = tmp_path / "out"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("a/b.txt", b"hello world")

    rc = main([str(zpath), str(outdir)])
    out = capsys.readouterr().out
    assert rc == 0
    assert not outdir.exists()
    assert "members: 1" in out
    assert "total_bytes: 11" in out
    assert "free >= 3x zip:" in out
    assert "dry run: nothing extracted, nothing executed" in out
    assert "ingest:" not in out


def test_malicious_members_refused_and_reported(tmp_path: Path, capsys) -> None:
    zpath = tmp_path / "mal.zip"
    outdir = tmp_path / "out"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("good.txt", b"safe")
        zf.writestr("../evil.txt", b"bad")
        sym = zipfile.ZipInfo("link.txt")
        sym.external_attr = 0o120000 << 16
        zf.writestr(sym, b"/etc/passwd")

    rc = main([str(zpath), str(outdir), "--execute"])
    out = capsys.readouterr().out
    assert rc == 0
    assert (outdir / "good.txt").read_bytes() == b"safe"
    assert not (tmp_path / "evil.txt").exists()
    assert "extracted: 1" in out
    refused_lines = [line for line in out.splitlines() if line.startswith("refused:")]
    assert any("path escapes the folder" in line for line in refused_lines)
    assert any("symlink" in line for line in refused_lines)


def test_valid_zip_extracts_same_file_set(tmp_path: Path) -> None:
    zpath = tmp_path / "valid.zip"
    outdir = tmp_path / "out"
    files = {"a/b.txt": b"alpha", "c d/e.csv": b"col1,col2\n1,2\n"}
    with zipfile.ZipFile(zpath, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)

    rc = main([str(zpath), str(outdir), "--execute"])
    assert rc == 0
    extracted = {
        p.relative_to(outdir).as_posix(): p.read_bytes() for p in outdir.rglob("*") if p.is_file()
    }
    assert extracted == files


def test_dry_run_prints_ingest_command_with_flags(tmp_path: Path, capsys) -> None:
    zpath = tmp_path / "one.zip"
    outdir = tmp_path / "out"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("a.txt", b"abc")

    rc = main([str(zpath), str(outdir), "--ingest", "--workers", "2", "--deadline-min", "45"])
    out = capsys.readouterr().out
    assert rc == 0
    assert not outdir.exists()
    ingest_lines = [line for line in out.splitlines() if line.startswith("ingest: ")]
    assert len(ingest_lines) == 1
    assert "--workers 2" in ingest_lines[0]
    assert "--deadline-min 45" in ingest_lines[0]
    assert f"voice_agent.rag.ingest {outdir}" in ingest_lines[0]
