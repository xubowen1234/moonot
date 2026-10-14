#!/usr/bin/env python3
"""Independent POT comparison. Optional deps; never installs or edits the repo."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import warnings

POT_VERSION = "0.9.6.post1"
ATOL = 2e-9
RTOL = 2e-8
SOLVER_TOL = 1e-13
MAX_ITER = 100000


def cases():
    """Portable integer LCG (not NumPy RNG); no upstream fixtures or data."""
    result = []

    def add(kind, name, a, b, c, eps=0.7, rs=1.0, rt=2.0, weight=1.0):
        result.append(dict(kind=kind, name=name, a=a, b=b, c=c,
                           eps=eps, rs=rs, rt=rt, weight=weight))

    add("balanced", "one_cell", [2.0], [2.0], [[0.4]], 0.3)
    add("balanced", "symmetric", [0.5, 0.5], [0.5, 0.5],
        [[0.0, 1.0], [1.0, 0.0]], 1.0)
    add("balanced", "zero_cost_rectangle", [0.5, 1.5], [0.4, 0.6, 1.0],
        [[0.0] * 3 for _ in range(2)])
    add("balanced", "zero_support", [0.0, 2.0, 1.0], [1.5, 0.0, 1.5, 0.0],
        [[0.1, 0.5, 0.9, 0.3], [0.8, 0.2, 0.1, 0.7], [0.2, 0.9, 0.6, 0.4]])
    add("balanced", "zero_total", [0.0, 0.0], [0.0, 0.0, 0.0], [[0.2] * 3] * 2)
    add("unbalanced", "one_cell", [2.0], [3.0], [[4.0]], 2.0, 1.0, 3.0, 0.5)
    add("unbalanced", "no_penalties", [0.2, 1.3], [0.8, 0.4, 1.8],
        [[0.0, 1.0, 0.3], [0.6, 0.1, 0.9]], 0.6, 0.0, 0.0, 1.2)
    add("unbalanced", "no_cost_no_penalties", [0.5, 1.5], [0.2, 0.0, 2.0],
        [[0.3] * 3] * 2, 0.9, 0.0, 0.0, 0.0)
    add("unbalanced", "zero_support", [0.0, 2.0, 1.0], [1.5, 0.0, 0.3, 0.0],
        [[0.1, 0.5, 0.9, 0.3], [0.8, 0.2, 0.1, 0.7], [0.2, 0.9, 0.6, 0.4]],
        0.5, 0.0, 4.0, 0.7)
    add("unbalanced", "zero_source", [0.0, 0.0], [1.0, 2.0, 0.0],
        [[0.2] * 3] * 2, 0.7, 2.0, 3.0)
    add("unbalanced", "zero_target", [1.0, 2.0], [0.0, 0.0, 0.0],
        [[0.2] * 3] * 2, 0.7, 2.0, 3.0)
    add("unbalanced", "zero_both", [0.0], [0.0, 0.0], [[0.0, 0.2]])

    state = 20261014

    def draw():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xffffffff
        return (state % 997 + 1) / 998.0

    shapes = [(1, 4), (4, 1), (2, 3), (3, 2), (3, 5), (5, 3)]
    penalties = [(0.0, 0.0), (0.0, 2.0), (3.0, 0.0),
                 (0.05, 7.0), (8.0, 0.1), (1.0, 1.0)]
    for k in range(24):
        n, m = shapes[k % len(shapes)]
        a = [0.15 + draw() for _ in range(n)]
        b = [0.15 + draw() for _ in range(m)]
        if k % 4 == 0 and n > 1:
            a[0] = 0.0
        if k % 5 == 0 and m > 1:
            b[-1] = 0.0
        c = [[2.0 * draw() for _ in range(m)] for _ in range(n)]
        eps = [0.15, 0.4, 1.2, 2.5][k % 4]
        total = [0.3, 1.0, 2.7][k % 3]
        ab = [x * total / sum(a) for x in a]
        bb = [x * total / sum(b) for x in b]
        add("balanced", "lcg_%02d" % k, ab, bb, c, eps)
        rs, rt = penalties[k % len(penalties)]
        add("unbalanced", "lcg_%02d" % k, a, b, c, eps, rs, rt,
            [0.0, 0.3, 1.0, 1.7][k % 4])
    return result


def literal(value):
    if isinstance(value, list):
        return "[" + ", ".join(literal(x) for x in value) + "]"
    text = repr(float(value))
    if "e" in text:
        mantissa, exponent = text.split("e")
        if "." not in mantissa:
            mantissa += ".0"
        text = mantissa + "e" + exponent
    return text


def bridge_source(items):
    code = [r"""///|
fn reference_plan_json(p : Array[Array[Double]]) -> String {
  let mut text = "["
  for i in 0..<p.length() {
    if i > 0 { text = text + "," }
    text = text + "["
    for j in 0..<p[i].length() {
      if j > 0 { text = text + "," }
      text = text + p[i][j].to_string()
    }
    text = text + "]"
  }
  text + "]"
}
"""]
    for k, case in enumerate(items):
        kind = case["kind"]
        opts = "epsilon: %s, tolerance: %s, max_iterations: %d" % (
            literal(case["eps"]), literal(SOLVER_TOL), MAX_ITER)
        if kind == "unbalanced":
            opts += ", rho_source: %s, rho_target: %s, cost_weight: %s" % tuple(
                literal(case[x]) for x in ("rs", "rt", "weight"))
        fields = (["transport_cost", "entropy_regularizer", "regularized_objective",
                   "marginal_residual"] if kind == "balanced" else
                  ["transport_cost", "reference_kl", "source_marginal_kl",
                   "target_marginal_kl", "objective", "scaling_residual"])
        message = "MOONOT_REFERENCE|%d|\\{reference_plan_json(r.coupling)}" % k
        message += "".join("|\\{r.%s}" % f for f in fields)
        message += "|\\{r.iterations}"
        term = "SinkhornTermination" if kind == "balanced" else "UnbalancedSinkhornTermination"
        code.append('''///|
fn case_%d() -> Unit {
  match @moonot.%s_sinkhorn(%s, %s, %s, { %s, }) {
    Err(_) => abort("reference solver error: %d")
    Ok(r) => {
      if !(r.termination is @moonot.%s::Converged) {
        abort("reference iteration limit: %d")
      }
      println("%s")
    }
  }
}
''' % (k, kind, literal(case["a"]), literal(case["b"]), literal(case["c"]),
       opts, k, term, k, message))
    code.append("///|\nfn main {\n" + "\n".join("  case_%d()" % k for k in range(len(items))) + "\n}\n")
    return "\n".join(code)


def moon_results(root, items, moon, target):
    """Compile unchanged root production files in a disposable module, not repo."""
    with tempfile.TemporaryDirectory(prefix="moonot-pot-") as directory:
        temp = Path(directory)
        production_hashes = {}
        for name in ("moon.mod", "moon.pkg"):
            shutil.copy2(root / name, temp / name)
        for source in root.glob("*.mbt"):
            if not source.name.endswith(("_test.mbt", "_wbtest.mbt")):
                shutil.copy2(source, temp / source.name)
        for copied in sorted(temp.iterdir()):
            if copied.is_file():
                production_hashes[copied.name] = hashlib.sha256(copied.read_bytes()).hexdigest()
        cmd = temp / "cmd" / "reference"
        cmd.mkdir(parents=True)
        (cmd / "moon.pkg").write_text(
            'import { "xubowen1234/moonot", }\npkgtype(kind: "executable")\n', encoding="utf-8")
        (cmd / "main.mbt").write_text(bridge_source(items), encoding="utf-8")
        proc = subprocess.run([moon, "run", "cmd/reference", "--target", target],
                              cwd=temp, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=300)
        if proc.returncode:
            raise RuntimeError("MoonBit bridge failed:\n" + proc.stdout + proc.stderr)
        found = {}
        for line in proc.stdout.splitlines():
            if line.startswith("MOONOT_REFERENCE|"):
                fields = line.split("|")
                key = int(fields[1])
                if key in found:
                    raise RuntimeError("duplicate bridge result")
                found[key] = (json.loads(fields[2]), [float(x) for x in fields[3:-1]], int(fields[-1]))
        if set(found) != set(range(len(items))):
            raise RuntimeError("missing bridge output:\n" + proc.stdout + proc.stderr)
        return found, production_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moon", default="moon", help="MoonBit CLI executable")
    parser.add_argument("--target", choices=("wasm", "wasm-gc", "js", "native"), default="wasm")
    parser.add_argument("--report", type=Path, help="optional JSON report outside the repository")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.report and args.report.resolve().is_relative_to(root):
        parser.error("--report must be outside the repository")
    try:
        import numpy as np
        import scipy
        import ot
    except ImportError as exc:
        print("BLOCKED: optional reference dependencies unavailable: %s\n"
              "Use an external private venv and scripts/requirements-reference.txt." % exc, file=sys.stderr)
        return 2
    if ot.__version__ != POT_VERSION:
        print("BLOCKED: expected POT %s, found %s" % (POT_VERSION, ot.__version__), file=sys.stderr)
        return 2
    items = cases()
    moon, production_hashes = moon_results(root, items, args.moon, args.target)
    maxima = {}
    reports = []
    entropy_checks = 0
    analytic_boundaries = 0

    def check(label, actual, expected):
        x, y = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        if x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise AssertionError(label + ": shape/nonfinite mismatch")
        delta = float(np.max(np.abs(x - y)))
        maxima[label.split(":")[0]] = max(maxima.get(label.split(":")[0], 0.0), delta)
        if not np.all(np.abs(x - y) <= ATOL + RTOL * np.abs(y)):
            raise AssertionError("%s: max absolute error %.17g" % (label, delta))
        return delta

    def kl(x, y):
        # Boundary conventions, independently evaluated; no POT source copied.
        x, y = np.asarray(x), np.asarray(y)
        if np.any((x > 0) & (y == 0)):
            return float("inf")
        active = x > 0
        return float(np.sum(y - x) + np.sum(x[active] * np.log(x[active] / y[active])))

    # Promote POT numerical warnings to failures; zero supports handled explicitly below.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings(
            "ignore", category=UserWarning,
            message=r"^If reg_type = entropy, then the matrix c is overwritten by the one matrix\.$")
        for k, case in enumerate(items):
            a, b, c = (np.asarray(case[x], dtype=float) for x in ("a", "b", "c"))
            eps, rs, rt, weight = (case[x] for x in ("eps", "rs", "rt", "weight"))
            p, fields, iterations = moon[k]
            p = np.asarray(p, dtype=float)
            if np.any(p < 0):
                raise AssertionError("negative MoonBit coupling")
            ai, bi = np.flatnonzero(a > 0), np.flatnonzero(b > 0)
            ix = np.ix_(ai, bi)
            ref = np.zeros_like(c)
            name = case["kind"] + "/" + case["name"]
            if len(ai) and len(bi):
                aa, bb, cc = a[ai], b[bi], c[ix]
                if case["kind"] == "balanced":
                    # ot.solve explicitly selects entropy; POT value uses sum(P log P).
                    solved = ot.solve(cc, aa, bb, reg=eps, reg_type="entropy",
                                      max_iter=MAX_ITER, tol=SOLVER_TOL)
                    ref[ix] = solved.plan
                    objective = float(solved.value) - eps * float(np.sum(ref))
                    check("pot_linear:" + name, solved.value_linear, np.sum(ref * c))
                else:
                    g, log = ot.unbalanced.sinkhorn_unbalanced(
                        aa, bb, weight * cc, reg=eps, reg_m=(rs, rt),
                        reg_type="kl", c=np.outer(aa, bb), method="sinkhorn",
                        numItermax=MAX_ITER, stopThr=SOLVER_TOL, log=True)
                    if not log["err"] or log["err"][-1] > SOLVER_TOL:
                        raise AssertionError("POT did not converge: " + name)
                    ref[ix] = g
                    # POT 0.9.6.post1 logs KL with backend mass=False, omitting -x+y.
                    mass = float(g.sum())
                    objective = (float(log["total_cost"]) + eps * (aa.sum() * bb.sum() - mass)
                                 + rs * (aa.sum() - mass) + rt * (bb.sum() - mass))
                    check("pot_linear:" + name, log["cost"], np.sum(ref * weight * c))
                    # Entropy uses c=ones: shift linear cost and restore constant.
                    shifted = weight * cc - eps * np.log(np.outer(aa, bb))
                    ge, le = ot.unbalanced.sinkhorn_unbalanced(
                        aa, bb, shifted, reg=eps, reg_m=(rs, rt),
                        reg_type="entropy", method="sinkhorn",
                        numItermax=MAX_ITER, stopThr=SOLVER_TOL, log=True)
                    if not le["err"] or le["err"][-1] > SOLVER_TOL:
                        raise AssertionError("POT entropy did not converge: " + name)
                    check("entropy_plan:" + name, ge, g)
                    emass = float(ge.sum())
                    entropy_generalized = (float(le["total_cost"]) + eps * (ge.size - emass)
                                           + rs * (aa.sum() - emass) + rt * (bb.sum() - emass))
                    adjusted = entropy_generalized + eps * (aa.sum() * bb.sum() - ge.size)
                    check("entropy_objective:" + name, adjusted, objective)
                    entropy_checks += 1
            else:
                # POT iteration is undefined on empty positive supports; no POT claim here.
                analytic_boundaries += 1
                objective = 0.0 if case["kind"] == "balanced" else rs * a.sum() + rt * b.sum()
            if case["kind"] == "balanced":
                active = ref > 0
                transport = float(np.sum(ref * c))
                entropy = eps * float(np.sum(ref[active] * (np.log(ref[active]) - 1.0)))
                expected = [transport, entropy, objective]
                check("balanced_components:" + name, fields[:3], expected)
                check("balanced_recomputed:" + name, objective, transport + entropy)
                check("balanced_rows:" + name, p.sum(axis=1), a)
                check("balanced_columns:" + name, p.sum(axis=0), b)
                if not 0 <= fields[3] <= SOLVER_TOL:
                    raise AssertionError("MoonBit marginal stopping failure: " + name)
            else:
                parts = [float(np.sum(ref * weight * c)), kl(ref, np.outer(a, b)),
                         kl(ref.sum(axis=1), a), kl(ref.sum(axis=0), b)]
                recomputed = parts[0] + eps * parts[1] + rs * parts[2] + rt * parts[3]
                check("unbalanced_components:" + name, fields[:5], parts + [objective])
                check("unbalanced_recomputed:" + name, objective, recomputed)
                if not 0 <= fields[5] <= SOLVER_TOL:
                    raise AssertionError("MoonBit scaling stopping failure: " + name)
            error = check(case["kind"] + "_coupling:" + name, p, ref)
            obj_error = check(case["kind"] + "_objective:" + name, fields[2 if case["kind"] == "balanced" else 4], objective)
            # Zeros are exact (including when marginal penalty is zero).
            if np.any(p[a == 0, :] != 0) or np.any(p[:, b == 0] != 0):
                raise AssertionError("nonzero forbidden support: " + name)
            reports.append(dict(name=name, shape=list(c.shape), coupling_error=error,
                                objective_error=obj_error, moon_iterations=iterations,
                                reference="POT" if len(ai) and len(bi) else "analytic boundary"))
    report = dict(status="PASS", pot=ot.__version__, numpy=np.__version__, scipy=scipy.__version__,
                  python=sys.version.split()[0], target=args.target, cases=len(items),
                  balanced=sum(x["kind"] == "balanced" for x in items),
                  unbalanced=sum(x["kind"] == "unbalanced" for x in items),
                  pot_comparisons=len(items) - analytic_boundaries,
                  analytic_boundaries=analytic_boundaries, entropy_cross_checks=entropy_checks,
                  atol=ATOL, rtol=RTOL, solver_tolerance=SOLVER_TOL,
                  production_sha256=production_hashes, maxima=maxima, results=reports)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, RuntimeError, subprocess.TimeoutExpired, OSError, Warning) as exc:
        print("FAIL: " + str(exc), file=sys.stderr)
        sys.exit(1)
