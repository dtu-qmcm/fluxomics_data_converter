"""Command-line interface ``fluxo``: ingest, validate, compare and export.

One install, a handful of commands (FluxoLink PATH_FORWARD §12)::

    fluxo meta init EC.fml -o study.yaml
    fluxo ingest EC.fml --meta study.yaml --tier benchmark -o ec.yaml
    fluxo ingest model.netw --results model_res -o run1.yaml
    fluxo validate ec.yaml --benchmark
    fluxo compare run1.yaml run2.yaml
    fluxo compare a.yaml b.yaml --inputs
    fluxo export ec.yaml --to mtf -o out/EC

Every command that writes a dataset validates it first (LinkML schema plus
the semantic and tier checks of ``fluxolink-validate``) and refuses to write
a document with errors unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Sequence

FORMATS = ("fluxml", "mtf", "freeflux", "inca")

# Sidecar keys handled here rather than by to_fluxo_dataset.
_CLI_SIDECAR_KEYS = ("data_origin",)

# What a benchmark built from experimental data must get from the sidecar.
_REQUIRED_CONDITION = (
    "organism_name",
    "strain",
    "specific_growth_rate",
    "growth_rate_unit",
)


def detect_format(path: Path, fmt: str | None) -> str:
    """Return the converter format key for an input path.

    Args:
        path: Input file, MTF prefix/.netw, or FreeFlux directory.
        fmt: Explicit format, overriding detection.

    Returns:
        One of ``FORMATS``.

    Raises:
        SystemExit: If the format cannot be determined.
    """
    if fmt:
        return fmt
    if path.is_dir():
        return "freeflux"
    suffix = path.suffix.lower()
    by_suffix = {
        ".fml": "fluxml",
        ".xml": "fluxml",
        ".netw": "mtf",
        ".tsv": "freeflux",
        ".xlsx": "freeflux",
        ".m": "inca",
    }
    if suffix in by_suffix:
        return by_suffix[suffix]
    if path.with_suffix(".netw").exists():
        return "mtf"
    raise SystemExit(
        f"cannot detect the format of {path}; pass --format {'|'.join(FORMATS)}"
    )


def parse_native(path: Path, fmt: str) -> Any:
    """Parse a native model with the converter's parsers.

    Args:
        path: Input path.
        fmt: One of ``FORMATS``.

    Returns:
        The parsed ``FluxomicsData``.
    """
    from fluxomics_data_converter import io

    if fmt == "fluxml":
        return io.parse_fluxml_file(str(path))
    if fmt == "mtf":
        prefix = path.with_suffix("") if path.suffix == ".netw" else path
        return io.parse_mtf(str(prefix))
    if fmt == "freeflux":
        return io.parse_freeflux(str(path if path.is_dir() else path.parent))
    return io.parse_inca(str(path))


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        doc = yaml.safe_load(handle) or {}
    if not isinstance(doc, dict):
        raise SystemExit(f"{path}: expected a YAML mapping")
    return doc


def _write_yaml(doc: dict[str, Any], path: Path, header: str | None) -> None:
    from fluxomics_data_converter.io.fluxolink_writer import to_yaml_text

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_yaml_text(doc, header), encoding="utf-8")


def _report(doc: dict[str, Any], *, benchmark: bool | None, label: str) -> bool:
    """Validate a document and print the issues.

    Returns:
        True when the document has no errors.
    """
    from fluxolink.validate import ERROR, schema_issues, validate_dataset

    issues = schema_issues(doc) or validate_dataset(doc, benchmark=benchmark)
    for issue in issues:
        print(f"{label}: {issue}", file=sys.stderr)
    return not any(issue.severity == ERROR for issue in issues)


def _prune(value: Any) -> Any:
    """Drop empty (``~``) entries recursively, so blanks mean "not given".

    Returns:
        The value without empty entries.
    """
    if isinstance(value, dict):
        pruned = {k: _prune(v) for k, v in value.items()}
        return {k: v for k, v in pruned.items() if v not in (None, {}, [])}
    if isinstance(value, list):
        items = [_prune(v) for v in value]
        return [v for v in items if v not in (None, {}, [])]
    return value


def clean_sidecar(sidecar: dict[str, Any]) -> dict[str, Any]:
    """Turn a filled-in ``fluxo meta init`` template into adapter metadata.

    Blank fields are dropped. Entries that are left without their required
    content are dropped too, instead of failing with a validation error: a
    contributor with no name, an investigation with no name, a condition
    that says nothing beyond its id and unit. The benchmark check then
    reports what is still missing.

    Args:
        sidecar: The parsed sidecar.

    Returns:
        The cleaned sidecar.
    """
    clean = _prune(sidecar)
    clean["contributors"] = [
        c for c in clean.get("contributors") or [] if c.get("name")
    ]
    if not (clean.get("investigation") or {}).get("name"):
        clean.pop("investigation", None)
        clean.pop("contributors", None)
        clean.pop("publication_doi", None)
    clean["conditions"] = [
        c
        for c in clean.get("conditions") or []
        if set(c) - {"id", "name", "growth_rate_unit"}
    ]
    return {k: v for k, v in clean.items() if v not in (None, {}, [])}


def missing_sidecar_fields(doc: dict[str, Any]) -> list[str]:
    """List the metadata a benchmark dataset still lacks.

    No native format carries investigation or cultivation metadata, so a
    benchmark built from experimental data needs a sidecar (PATH_FORWARD
    §12). Synthetic and simulated data need no cultivation.

    Args:
        doc: The dataset as a dict.

    Returns:
        Human-readable names of the missing fields, empty when complete.
    """
    missing: list[str] = []
    if not doc.get("investigations"):
        missing.append("investigation.name")
    conditions = {c.get("id"): c for c in doc.get("conditions") or []}
    for exp in doc.get("experiments") or []:
        origin = exp.get("data_origin")
        if origin is None:
            missing.append(
                f"data_origin (experiment {exp.get('id')}): "
                "experimental | synthetic | simulated"
            )
            continue
        if origin != "experimental":
            continue
        cond = conditions.get(exp.get("condition"))
        if cond is None:
            missing.append(f"conditions (experiment {exp.get('id')})")
            continue
        missing += [
            f"conditions[{cond.get('id')}].{slot}"
            for slot in _REQUIRED_CONDITION
            if cond.get(slot) in (None, "")
        ]
    return sorted(set(missing))


# ---------------------------------------------------------------- commands ---


def cmd_ingest(args: argparse.Namespace) -> int:
    """Parse a native model (and optional results) into a dataset file.

    Returns:
        Exit code.
    """
    from fluxomics_data_converter.io.fluxolink_writer import to_fluxo_dataset

    source = Path(args.input)
    fmt = detect_format(source, args.format)
    sidecar = clean_sidecar(_load_yaml(Path(args.meta))) if args.meta else {}
    data_origin = args.data_origin or sidecar.get("data_origin")
    metadata = {k: v for k, v in sidecar.items() if k not in _CLI_SIDECAR_KEYS}
    try:
        dataset = to_fluxo_dataset(
            parse_native(source, fmt),
            dataset_id=args.dataset_id or source.stem.replace(".", "_"),
            source_path=source,
            source_format=fmt,
            software=args.software,
            software_version=args.software_version,
            dataset_type=args.tier,
            data_origin=data_origin,
            metadata=metadata,
            source_root=Path(args.source_root) if args.source_root else None,
        )
    except (ValueError, TypeError) as error:
        print(f"cannot build the dataset: {error}", file=sys.stderr)
        return 1
    doc = dataset.model_dump(mode="json", exclude_unset=True, exclude_none=True)
    if args.results:
        run = read_results(Path(args.results), args.results_format, source)
        networks = doc.get("networks") or []
        if len(networks) == 1:
            run.setdefault("network_model_id", networks[0]["id"])
        investigations = doc.get("investigations") or []
        if len(investigations) == 1:
            run.setdefault("investigation_id", investigations[0]["id"])
        doc.setdefault("simulation_runs", []).append(run)

    if args.tier == "benchmark":
        missing = missing_sidecar_fields(doc)
        if missing:
            print(
                "benchmark tier: the metadata sidecar (--meta) must supply:",
                file=sys.stderr,
            )
            for item in missing:
                print(f"  - {item}", file=sys.stderr)
            print("scaffold one with `fluxo meta init`", file=sys.stderr)
            if not args.force:
                return 2

    output = Path(args.output)
    ok = _report(doc, benchmark=args.tier == "benchmark", label=str(output))
    if not ok and not args.force:
        print(
            "not written: fix the errors above or pass --force", file=sys.stderr
        )
        return 1
    header = f"Generated by `fluxo ingest` from {source.name} ({fmt})."
    _write_yaml(doc, output, header)
    print(f"wrote {output}")
    return 0


def read_results(path: Path, kind: str | None, source: Path) -> dict[str, Any]:
    """Read a tool's results into a SimulationRun dict.

    Args:
        path: influx_si ``<prefix>_res`` directory or INCA ``.mat`` file.
        kind: ``influx_si`` or ``inca``; inferred from the path when None.
        source: The ingested input, used to find influx_si's input files.

    Returns:
        The run as a dict.

    Raises:
        SystemExit: If ``kind`` is not a known results format.
    """
    kind = kind or ("inca" if path.suffix == ".mat" else "influx_si")
    if kind == "influx_si":
        from fluxomics_data_converter.io.results.influx_si import (
            read_influx_si_results,
        )

        run = read_influx_si_results(path, input_dir=source.parent)
    elif kind == "inca":
        from fluxomics_data_converter.io.results.inca import (
            read_inca_results,
        )

        run = read_inca_results(path)
    else:
        raise SystemExit(f"unknown --results-format {kind!r}")
    return run.model_dump(mode="json", exclude_unset=True, exclude_none=True)


def cmd_validate(args: argparse.Namespace) -> int:
    """Delegate to ``fluxolink-validate``.

    Returns:
        Exit code.
    """
    from fluxolink.validate import main as validate_main

    argv = [str(f) for f in args.files]
    argv += ["--benchmark"] if args.benchmark else []
    argv += ["--strict"] if args.strict else []
    argv += ["-q"] if args.quiet else []
    return validate_main(argv)


def _flux_columns(
    doc: dict[str, Any], label: str, aliases: dict[str, str]
) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    """Collect flux results of one file by (reaction, flux type).

    Returns:
        ``{(reaction, flux_type): {run column: result}}``.
    """
    table: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for run in doc.get("simulation_runs") or []:
        solution = run.get("flux_solution") or {}
        tool = (run.get("software") or {}).get("software_name", "?")
        column = f"{label}:{run.get('id')} ({tool})"
        for res in solution.get("flux_results") or []:
            rid = str(res.get("reaction_id"))
            key = (aliases.get(rid, rid), str(res.get("flux_type")))
            table.setdefault(key, {})[column] = res
    return table


def _alias_map(doc: dict[str, Any]) -> dict[str, str]:
    """Map tool-native reaction ids (from aliases) to canonical ids.

    Returns:
        Native id to canonical reaction id.
    """
    mapping: dict[str, str] = {}
    for net in doc.get("networks") or []:
        for rxn in net.get("reactions") or []:
            for alias in rxn.get("aliases") or []:
                mapping[alias.split(":", 1)[1]] = rxn["id"]
    return mapping


def _cell(res: dict[str, Any] | None) -> str:
    if res is None:
        return "-"
    cell = f"{res['value']:.6g}"
    if res.get("lower_ci") is not None and res.get("upper_ci") is not None:
        cell += f" [{res['lower_ci']:.4g}, {res['upper_ci']:.4g}]"
    return cell


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare flux results by reaction id, or input content with --inputs.

    Returns:
        Exit code.
    """
    docs = [(Path(p).stem, _load_yaml(Path(p))) for p in args.files]
    if args.inputs:
        return _compare_inputs(docs, args)
    merged: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for label, doc in docs:
        aliases = _alias_map(doc) if args.use_aliases else {}
        for key, cols in _flux_columns(doc, label, aliases).items():
            merged.setdefault(key, {}).update(cols)
    columns = sorted({c for cols in merged.values() for c in cols})
    if len(columns) < 2:
        print("need at least two runs with flux results", file=sys.stderr)
        return 1
    print("\t".join(["reaction", "flux_type", *columns, "max_abs_diff"]))
    worst = 0.0
    for key in sorted(merged):
        cols = merged[key]
        values = [cols[c]["value"] for c in columns if c in cols]
        diff = max(values) - min(values) if len(values) > 1 else 0.0
        worst = max(worst, diff)
        cells = [_cell(cols.get(c)) for c in columns]
        print("\t".join([key[0], key[1], *cells, f"{diff:.3g}"]))
    print(
        f"# {len(merged)} fluxes, {len(columns)} runs, "
        f"largest disagreement {worst:.3g}"
    )
    return 0


def _compare_inputs(
    docs: list[tuple[str, dict[str, Any]]], args: argparse.Namespace
) -> int:
    from fluxolink.datamodel.fluxolink import FluxoDataset

    from fluxomics_data_converter.equivalence import compare
    from fluxomics_data_converter.io.fluxolink_parser import from_fluxo_dataset

    if len(docs) != 2:
        print("--inputs compares exactly two datasets", file=sys.stderr)
        return 1
    models = [
        from_fluxo_dataset(FluxoDataset.model_validate(doc)) for _, doc in docs
    ]
    report = compare(models[0], models[1], rtol=args.rtol, atol=args.atol)
    print(report.summary(limit=args.limit))
    return 0 if report.equivalent else 1


def cmd_export(args: argparse.Namespace) -> int:
    """Write a tool's native input files from a dataset.

    Returns:
        Exit code.
    """
    from fluxolink.datamodel.fluxolink import FluxoDataset

    from fluxomics_data_converter import io
    from fluxomics_data_converter.io.fluxolink_parser import from_fluxo_dataset

    doc = _load_yaml(Path(args.dataset))
    data = from_fluxo_dataset(
        FluxoDataset.model_validate(doc), network_id=args.network
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.to == "fluxml":
        io.write_fluxml(data, str(out))
    else:
        names = [exp.name for exp in data.experiments]
        if len(names) > 1 and not args.experiment:
            print(
                f"{args.to} holds one configuration; choose one with "
                f"--experiment ({', '.join(names)})",
                file=sys.stderr,
            )
            return 1
        writer = io.write_mtf if args.to == "mtf" else io.write_freeflux
        writer(data, str(out), experiment_name=args.experiment)
    print(f"wrote {args.to} input to {out}")
    return 0


_SIDECAR_TEMPLATE = """\
# Metadata sidecar for {name}. No {fmt} file carries these fields.
# Use: fluxo ingest {name} --meta THIS_FILE [--tier benchmark] -o dataset.yaml
#
# Fill what you know and delete what does not apply. Leave a field empty
# (~) rather than guessing: an empty field is reported, a guess is not.
# For synthetic or simulated data, delete `conditions`: a synthetic
# benchmark has no cultivation. Never fill specific_growth_rate from a
# fitted biomass flux.

data_origin: ~            # experimental | synthetic | simulated

dataset:
  name: ~

investigation:
  name: ~                 # required
  description: {description}
  investigation_type: MFA # MFA | FBA

contributors:
  - id: C1
    name: {modeler}
    orcid: ~

publication_doi: ~

conditions:               # one entry per cultivation; see experiment_conditions
  - id: COND1
    organism_name: ~
    strain: {strain}
    culture_medium: ~
    specific_growth_rate: ~
    growth_rate_unit: 1/h

# Map configuration / experiment names to condition ids when there is more
# than one condition (configurations here: {configurations}).
experiment_conditions: {{}}
"""


def _yaml_scalar(value: Any) -> str:
    if value in (None, ""):
        return "~"
    return (
        yaml.safe_dump(value, default_style=None)
        .strip()
        .removesuffix("...")
        .strip()
    )


def cmd_meta_init(args: argparse.Namespace) -> int:
    """Scaffold a metadata sidecar with what the native file cannot supply.

    Returns:
        Exit code.
    """
    source = Path(args.input)
    fmt = detect_format(source, args.format)
    data = parse_native(source, fmt)
    info = data.metadata
    text = _SIDECAR_TEMPLATE.format(
        name=source.name,
        fmt=fmt,
        description=_yaml_scalar(info.comment if info else None),
        modeler=_yaml_scalar(info.modeler if info else None),
        strain=_yaml_scalar(info.strain if info else None),
        configurations=", ".join(e.name for e in data.experiments) or "none",
    )
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``fluxo`` argument parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="fluxo",
        description="Ingest, validate, compare and export FluxoLink datasets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser(
        "ingest", help="native model (+ results) -> dataset"
    )
    ingest.add_argument(
        "input", help=".fml, .netw, FreeFlux dir/table, INCA .m"
    )
    ingest.add_argument("-o", "--output", required=True)
    ingest.add_argument("--format", choices=FORMATS)
    ingest.add_argument(
        "--meta", help="metadata sidecar (see `fluxo meta init`)"
    )
    ingest.add_argument("--results", help="influx_si <prefix>_res or INCA .mat")
    ingest.add_argument("--results-format", choices=("influx_si", "inca"))
    ingest.add_argument("--dataset-id")
    ingest.add_argument(
        "--tier", choices=("catalog", "benchmark"), default="catalog"
    )
    ingest.add_argument(
        "--data-origin", choices=("experimental", "synthetic", "simulated")
    )
    ingest.add_argument("--software", help="tool (default: the format's own)")
    ingest.add_argument("--software-version")
    ingest.add_argument(
        "--source-root", help="directory SourceFile paths are relative to"
    )
    ingest.add_argument(
        "--force", action="store_true", help="write despite errors"
    )
    ingest.set_defaults(func=cmd_ingest)

    validate = sub.add_parser(
        "validate", help="schema + semantic + tier checks"
    )
    validate.add_argument("files", nargs="+")
    validate.add_argument("--benchmark", action="store_true")
    validate.add_argument("--strict", action="store_true")
    validate.add_argument("-q", "--quiet", action="store_true")
    validate.set_defaults(func=cmd_validate)

    compare = sub.add_parser("compare", help="compare results or inputs")
    compare.add_argument("files", nargs="+")
    compare.add_argument("--inputs", action="store_true", help="compare inputs")
    compare.add_argument("--use-aliases", action="store_true")
    compare.add_argument("--rtol", type=float, default=1e-6)
    compare.add_argument("--atol", type=float, default=1e-9)
    compare.add_argument(
        "--limit", type=int, default=20, help="differences shown"
    )
    compare.set_defaults(func=cmd_compare)

    export = sub.add_parser("export", help="dataset -> native tool input")
    export.add_argument("dataset")
    export.add_argument(
        "--to", choices=("fluxml", "mtf", "freeflux"), required=True
    )
    export.add_argument("-o", "--output", required=True)
    export.add_argument(
        "--network", help="network id if the dataset has several"
    )
    export.add_argument(
        "--experiment",
        help="configuration to write (MTF and FreeFlux hold one)",
    )
    export.set_defaults(func=cmd_export)

    meta = sub.add_parser("meta", help="metadata sidecar helpers")
    meta_sub = meta.add_subparsers(dest="meta_command", required=True)
    init = meta_sub.add_parser(
        "init", help="scaffold a sidecar for a native file"
    )
    init.add_argument("input")
    init.add_argument("-o", "--output")
    init.add_argument("--format", choices=FORMATS)
    init.set_defaults(func=cmd_meta_init)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``fluxo`` command.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
