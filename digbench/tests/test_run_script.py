import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("schema_rc,basic_rc", [(0, 0), (7, 0), (0, 9), (7, 9)])
def test_batch_reports_child_failures(tmp_path, schema_rc, basic_rc):
    script = tmp_path / "run.sh"
    shutil.copy2(Path(__file__).resolve().parents[1] / "run.sh", script)
    for name in ("schema", "basic_harness"):
        (tmp_path / name).mkdir()
    interpreter = tmp_path / "fake python"
    interpreter.write_text(
        '#!/bin/bash\n'
        'touch completed\n'
        'if [ "$1" = "-m" ]; then exit "$SCHEMA_TEST_EXIT"; fi\n'
        'exit "$BASIC_TEST_EXIT"\n'
    )
    interpreter.chmod(0o755)
    env = dict(os.environ, PY=str(interpreter),
               SCHEMA_TEST_EXIT=str(schema_rc), BASIC_TEST_EXIT=str(basic_rc))
    result = subprocess.run(["bash", str(script), "medium", "P-1"],
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == (1 if schema_rc or basic_rc else 0)
    for name in ("schema", "basic_harness"):
        assert (tmp_path / name / "completed").is_file()
    assert ("P-1 schema failed" in result.stderr) == bool(schema_rc)
    assert ("P-1 basic failed" in result.stderr) == bool(basic_rc)


def test_batch_requires_at_least_one_game(tmp_path):
    script = Path(__file__).resolve().parents[1] / "run.sh"
    result = subprocess.run(["bash", str(script), "medium"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert "Usage:" in result.stderr
