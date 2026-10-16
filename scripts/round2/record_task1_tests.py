"""Task 1: run the trainer tests on this machine and record the result the runner gates on.

Runs pytest over the patched learning-concepts checkout and prints the numbers the
report needs: the Figure 1 gradients (test I) and the largest |grad U - grad (R + B)|
through compute_loss on a real transformer (test B).

    python scripts/round2/record_task1_tests.py
Writes results/round2/task1_tests.json.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

import r2common as C


def measurements():
    sys.path.insert(0, str(C.EXT))
    sys.path.insert(0, str(C.EXT / "tests"))
    import torch
    import test_factorial as T

    out = {}
    logits, _ = T._figure1()
    x = logits.clone().requires_grad_(True)
    terms = T._terms(x, [0], [0], [0], [[1, 2]])
    for name in ("mass", "balance", "uncapped"):
        x.grad = None
        terms[name].sum().backward(retain_graph=True)
        out[f"figure1_neg_grad_{name}"] = [round(v, 4) for v in (-x.grad[0, 0, :3]).tolist()]
    t = T._stub()
    t.alt_aux = "uniform"
    x.grad = None
    t._compute_alt_aux_loss(x, [0], [0], [[1, 2]]).backward()
    out["figure1_neg_grad_uniform"] = [round(v, 4) for v in (-x.grad[0, 0, :3]).tolist()]
    worst = {}
    for gamma in (0.0625, 0.125, 1.0):
        model, batch = T._tiny_model(), T._batch()
        _, gu = T._grads(model, T._full("uniform", gamma), batch)
        _, grb = T._grads(model, T._full(gm=gamma, gb=gamma), batch)
        worst[str(gamma)] = max((gu[n] - grb[n]).abs().max().item() for n in gu)
    out["testB_max_abs_grad_diff"] = worst
    logits, b, s, y, alts = T._slots(seed=0)
    terms = T._terms(logits, b, s, y, alts)
    u = T._uniform_slot(logits, b, s, alts)
    import math
    out["testA_max_abs_U_minus_R_B_logA"] = max(
        abs(float(u[i] - terms["mass"][i] - terms["balance"][i] - math.log(n)))
        for i, n in enumerate(T.SIZES) if n)
    out["torch"] = torch.__version__
    return out


def main():
    junit = C.R2 / "task1_pytest.xml"
    junit.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", "-p", "no:cacheprovider",
                           f"--junitxml={junit}"], cwd=C.EXT, capture_output=True, text=True)
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    failed = [f"{c.get('classname')}::{c.get('name')}" for c in suite.iter("testcase")
              if c.find("failure") is not None or c.find("error") is not None]
    record = {
        "passed": proc.returncode == 0 and not failed,
        "tests": int(suite.get("tests")), "failures": failed,
        "factorial_tests": sum(1 for c in suite.iter("testcase") if "test_factorial" in c.get("classname", "")),
        "patch_sha256": C.sha256(C.PATCH), "git_commit": C.git_commit(),
        "machine": f"{platform.node()} {platform.machine()} python {platform.python_version()}",
        "measurements": measurements() if proc.returncode == 0 else None,
        "written_utc": C.now_utc(),
    }
    C.write_json(C.R2 / "task1_tests.json", record)
    print(proc.stdout[-1500:])
    print(json.dumps(record, indent=1))
    sys.exit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
