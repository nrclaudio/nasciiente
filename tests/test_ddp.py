import os
import subprocess
import sys


def test_ddp_two_process_smoke():
    """Run the conditioned training path under real 2-process DDP (gloo).

    Catches reducer errors from parameters that only conditionally join
    the autograd graph (e.g. the conditioning null token) — a failure
    mode that only appears under DDP, i.e. on the actual multi-GPU run.
    """
    repo = os.path.join(os.path.dirname(__file__), "..")
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", os.path.join("tests", "ddp_smoke.py")],
        capture_output=True, text=True, timeout=600, cwd=repo,
    )
    assert result.returncode == 0, (
        f"DDP smoke failed:\n{result.stdout}\n{result.stderr}")
    assert "DDP SMOKE OK" in result.stdout
