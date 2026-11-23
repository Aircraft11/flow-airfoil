"""Reproduce the main-model fixed-angle XFOIL evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

COEFFICIENTS = ("cl", "cd", "cm")
DEFAULT_SETTINGS = {
    "reynolds": 1e6,
    "iterations": 160,
    "panel_count": 400,
    "timeout": 45.0,
    "alpha_step": 1.0,
    "alpha_tolerance": 0.001,
    "mach_tolerance": 0.00051,
    "reynolds_relative_tolerance": 0.001,
}
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
LOG_PREFIX_BYTES = 1024 * 1024


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def build_commands(alpha, mach, settings=DEFAULT_SETTINGS):
    if not math.isfinite(alpha) or not 0 <= mach < 1:
        raise ValueError("Expected finite alpha and 0 <= Mach < 1")
    solve = []
    if abs(alpha) > 1e-10:
        # Continue from zero incidence using increments bounded by alpha_step.
        steps = max(1, math.ceil(abs(alpha) / settings["alpha_step"]))
        solve.append(f"ASEQ 0 {alpha:.10g} {alpha / steps:.10g}")
    # Check the printed polar endpoint against the explicitly requested alpha.
    solve.append(f"ALFA {alpha:.10g}")
    return "\n".join(
        [
            "PLOP",
            "G",
            "",
            "LOAD airfoil.dat",
            "PPAR",
            f"N {int(settings['panel_count'])}",
            "",
            "",
            "PANE",
            "OPER",
            f"VISC {settings['reynolds']:.10g}",
            f"MACH {mach:.10g}",
            f"ITER {int(settings['iterations'])}",
            "PACC",
            "polar.txt",
            "",
            *solve,
            "PACC",
            "",
            "QUIT",
            "",
        ]
    )


def parse_polar(text):
    records, started = [], False
    for line in text.splitlines():
        stripped = line.strip().replace("\x00", "")
        if stripped.startswith("------"):
            started = True
            continue
        parts = stripped.split()
        if not started or len(parts) < 5:
            continue
        try:
            values = [float(parts[index]) for index in (0, 1, 2, 4)]
        except ValueError:
            continue
        if all(math.isfinite(value) for value in values):
            records.append(dict(zip(("alpha", *COEFFICIENTS), values)))
    mach_match = re.search(r"\bMach\s*=\s*([-+0-9.Ee]+)", text, re.IGNORECASE)
    re_match = re.search(r"\bRe\s*=\s*([-+0-9.]+)\s*[eE]\s*([-+]?\d+)", text)
    if re_match:
        reynolds = float(re_match[1]) * 10 ** int(re_match[2])
    else:
        re_match = re.search(r"\bRe\s*=\s*([-+0-9.]+)", text)
        reynolds = float(re_match[1]) if re_match else None
    mach = float(mach_match[1]) if mach_match else None
    return records, mach, reynolds


def classify(polar_text, alpha, mach, status, returncode, settings=DEFAULT_SETTINGS):
    """Check solver completion, terminal incidence, polar headers, and positive drag."""
    records, polar_mach, reynolds = parse_polar(polar_text)
    raw = records[-1] if records else None
    reasons = []
    if status != "completed" or returncode != 0:
        reasons.append(status if status != "completed" else "nonzero_exit")
    if raw is None:
        reasons.append("no_finite_polar_record")
    elif abs(raw["alpha"] - alpha) > settings["alpha_tolerance"]:
        reasons.append("terminal_alpha_mismatch")
    if polar_mach is None or abs(polar_mach - mach) > settings["mach_tolerance"]:
        reasons.append("polar_mach_missing_or_mismatch")
    if reynolds is None or not math.isclose(
        reynolds,
        settings["reynolds"],
        rel_tol=settings["reynolds_relative_tolerance"],
        abs_tol=0.0,
    ):
        reasons.append("polar_reynolds_missing_or_mismatch")
    if raw is not None and raw["cd"] <= 0:
        reasons.append("nonpositive_drag")
    return {
        "converged": not reasons,
        "failure_reason": ";".join(reasons) or None,
        "raw_coefficients": raw,
        "xfoil": None if reasons else raw,
    }


def panel_count(text):
    patterns = (
        r"number\s+of\s+panel\s+nodes\s*[:=]?\s*(\d+)",
        r"panel\s+nodes\s*[:=]\s*(\d+)",
        r"(\d+)\s+panel\s+nodes",
    )
    observed = [
        int(match)
        for line in text.splitlines()
        if not re.search(r"maximum|limit|reducing|reduced|clipp", line, re.IGNORECASE)
        for pattern in patterns
        for match in re.findall(pattern, line, re.IGNORECASE)
    ]
    return observed[-1] if observed else None


def solve(xfoil, dat, *, mach, alpha, folder, settings=DEFAULT_SETTINGS):
    """Run one DAT payload with the paper's fixed-angle protocol."""
    solver = Path(xfoil).resolve(strict=True)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "airfoil.dat").write_bytes(dat)
    commands = build_commands(alpha, mach, settings).encode("ascii")
    (folder / "stdin.txt").write_bytes(commands)
    # Each case owns its working directory and runs an independent solver process.
    env = {**os.environ, **dict.fromkeys(THREAD_VARIABLES, "1")}
    status, returncode = "completed", None
    with (folder / "stdout.txt").open("wb") as stdout, (folder / "stderr.txt").open("wb") as stderr:
        try:
            result = subprocess.run(
                [str(solver)],
                input=commands,
                stdout=stdout,
                stderr=stderr,
                cwd=folder,
                timeout=settings["timeout"],
                env=env,
                check=False,
            )
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            status = "timeout"
        except OSError as exc:
            status = "launch_failed"
            stderr.write(str(exc).encode("utf-8"))
    polar = folder / "polar.txt"
    if not polar.exists():
        polar.write_bytes(b"")
    log = []
    for name in ("stdout.txt", "stderr.txt"):
        with (folder / name).open("rb") as stream:
            log.append(stream.read(LOG_PREFIX_BYTES).decode("utf-8", errors="replace"))
    return {
        "status": status,
        "returncode": returncode,
        "actual_panel_count": panel_count("\n".join(log)),
        **classify(
            polar.read_text(encoding="utf-8", errors="replace"),
            alpha,
            mach,
            status,
            returncode,
            settings,
        ),
    }


def summarize(rows, targets):
    """Count every attempt and compute aerodynamic errors over converged cases."""
    accepted = [row for row in rows if row["converged"]]
    metrics = {}
    if accepted:
        for coefficient in COEFFICIENTS:
            labels = [targets[row["target_index"]]["coefficients"][coefficient] for row in accepted]
            errors = [row["xfoil"][coefficient] - ref for row, ref in zip(accepted, labels)]
            mean = math.fsum(labels) / len(labels)
            sst = math.fsum((ref - mean) ** 2 for ref in labels)
            metrics[coefficient] = {
                "mae": math.fsum(abs(error) for error in errors) / len(errors),
                "r2": 1.0 - math.fsum(error**2 for error in errors) / sst if sst else None,
            }
    return {"attempted": len(rows), "converged": len(accepted), "metrics": metrics}


def check_case(archive, case, target, settings):
    """Validate the inputs and reclassify the saved polar without running XFOIL."""
    dat = archive.read(f"dat/{case['id']}.dat")
    polar = archive.read(f"polar/{case['id']}.txt")
    commands = build_commands(target["alpha"], target["mach"], settings).encode("ascii")
    for kind, payload in (("dat", dat), ("polar", polar), ("stdin", commands)):
        if sha256(payload) != case[f"{kind}_sha256"]:
            raise ValueError(f"{case['id']}: {kind} digest mismatch")
    recorded = case["recorded"]
    classified = classify(
        polar.decode("utf-8", errors="replace"),
        target["alpha"],
        target["mach"],
        recorded["status"],
        recorded["returncode"],
        settings,
    )
    if any(recorded[key] != value for key, value in classified.items()):
        raise ValueError(f"{case['id']}: recorded result does not agree with its polar")
    return dat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("xfoil/manifest.json"))
    parser.add_argument("--xfoil", help="Path or command name of the XFOIL executable")
    parser.add_argument("--output", type=Path, help="New directory for solver logs and results")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Run the first N requests")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Require the paper solver binary and compare each replayed result",
    )
    parser.add_argument(
        "--check-bundle", action="store_true", help="Verify the supplied inputs and polars"
    )
    parser.add_argument("--dat", type=Path, help="Evaluate a single airfoil DAT file")
    parser.add_argument("--mach", type=float, help="Mach number for --dat")
    parser.add_argument("--alpha", type=float, help="Angle of attack in degrees for --dat")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("worker count and case limit must be positive")
    if args.check_bundle and args.verify:
        parser.error("Use --check-bundle for offline checks or --verify for solver replay")
    if not args.check_bundle and (not args.xfoil or not args.output):
        parser.error("--xfoil and --output are required to replay cases")
    if args.dat is not None:
        if args.check_bundle or args.verify or args.limit is not None:
            parser.error("--dat cannot be combined with --check-bundle, --limit or --verify")
        if args.mach is None or args.alpha is None:
            parser.error("--dat requires --mach and --alpha")
        solver = Path(shutil.which(args.xfoil) or args.xfoil).resolve(strict=True)
        result = solve(
            solver, args.dat.read_bytes(), mach=args.mach, alpha=args.alpha, folder=args.output
        )
        result["solver_sha256"] = sha256(solver.read_bytes())
        report = json.dumps(result, indent=2)
        (args.output / "result.json").write_text(report + "\n", encoding="utf-8")
        print(report)
        return
    if args.mach is not None or args.alpha is not None:
        parser.error("--mach and --alpha require --dat")
    archive_path = args.manifest.with_name("cases.zip")
    if not args.manifest.is_file() or not archive_path.is_file():
        parser.error(
            "Study replay requires an authorized copy of manifest.json and cases.zip. "
            "Pass --manifest with the archive beside it, or use --dat for a single airfoil."
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    settings = manifest["settings"]
    targets = manifest["targets"]
    cases = manifest["cases"]
    with ZipFile(archive_path) as archive:
        data = [
            check_case(archive, case, targets[case["target_index"]], settings) for case in cases
        ]
    historical = [{**case["recorded"], "target_index": case["target_index"]} for case in cases]
    print(json.dumps(summarize(historical, targets), indent=2))
    if args.check_bundle:
        return
    solver = Path(shutil.which(args.xfoil) or args.xfoil).resolve(strict=True)
    solver_hash = sha256(solver.read_bytes())
    if args.verify and solver_hash != manifest["solver_sha256"]:
        parser.error("solver binary does not match the original SHA-256")
    args.output.mkdir(parents=True, exist_ok=False)

    def run(item):
        case, dat = item
        target = targets[case["target_index"]]
        result = solve(
            solver,
            dat,
            mach=target["mach"],
            alpha=target["alpha"],
            folder=args.output / case["id"],
            settings=settings,
        )
        return {"id": case["id"], "target_index": case["target_index"], **result}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        replayed = list(pool.map(run, zip(cases[: args.limit], data[: args.limit])))
    summary = summarize(replayed, targets)
    mismatches = [
        case["id"]
        for case, row in zip(cases, replayed)
        if any(row[key] != value for key, value in case["recorded"].items())
    ]
    output = {
        "solver_sha256": solver_hash,
        "original_solver": solver_hash == manifest["solver_sha256"],
        "summary": summary,
        "differences_from_recorded": mismatches,
        "cases": replayed,
    }
    (args.output / "results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**summary, "differences_from_recorded": len(mismatches)}, indent=2))
    if args.verify and mismatches:
        raise SystemExit("Replayed results differ from recorded results")


if __name__ == "__main__":
    main()
