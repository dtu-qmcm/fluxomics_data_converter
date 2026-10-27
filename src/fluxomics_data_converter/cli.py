"""Command-line interface ``fluxo``: translate tool inputs and link them.

Translating needs no record::

    fluxo convert EC.fml --to mtf freeflux -o runs/

A FluxoLink record (``ec.yaml``) ties one study's tool files together: the
reference model, every tool input with its checksum and equivalence check,
id aliases, and each tool's results::

    fluxo meta init EC.fml -o study.yaml
    fluxo ingest EC.fml --meta study.yaml --tier benchmark -o ec.yaml
    fluxo convert EC.fml --to mtf freeflux -o runs/ --link ec.yaml
    fluxo add-input ec.yaml my_EC.netw --experiment a_STAT
    fluxo add-results ec.yaml runs/mtf/EC_a_STAT_res --tool influx_si
    fluxo export ec.yaml --to fluxml -o runs/fluxml/EC.fml
    fluxo validate ec.yaml --benchmark
    fluxo compare ec.yaml
    fluxo compare a.yaml b.yaml --inputs

Every command that writes a record validates it first (LinkML schema plus
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
WRITABLE = ("fluxml", "mtf", "freeflux")

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
    """Return the tool key of an input path.

    Args:
        path: Input file, MTF prefix/.netw, or FreeFlux directory.
        fmt: Explicit tool key, overriding detection.

    Returns:
        One of ``FORMATS``.

    Raises:
        SystemExit: If the format cannot be determined.
    """
    from fluxomics_data_converter.link.tool_inputs import detect_tool

    try:
        return detect_tool(path, fmt)
    except ValueError as error:
        raise SystemExit(str(error)) from error


def parse_native(path: Path, fmt: str) -> Any:
    """Parse a native model with the converter's parsers.

    Args:
        path: Input path.
        fmt: One of ``FORMATS``.

    Returns:
        The parsed ``FluxomicsData``.
    """
    from fluxomics_data_converter.link.tool_inputs import read_tool_input

    return read_tool_input(path, fmt)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        doc = yaml.safe_load(handle) or {}
    if not isinstance(doc, dict):
        raise SystemExit(f"{path}: expected a YAML mapping")
    return doc


def _write_yaml(doc: dict[str, Any], path: Path, header: str | None) -> None:
    from fluxomics_data_converter.link.record import record_text

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record_text(doc, header), encoding="utf-8")


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
    from fluxomics_data_converter.link import to_fluxo_dataset

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
            source_root=(
                Path(args.source_root)
                if args.source_root
                else _record_root(Path(args.output))
            ),
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
        from fluxomics_data_converter.link.results.influx_si import (
            read_influx_si_results,
        )

        run = read_influx_si_results(path, input_dir=source.parent)
    elif kind == "inca":
        from fluxomics_data_converter.link.results.inca import (
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
    """Map tool-native ids (from the record's aliases) to canonical ids.

    Returns:
        Native id to canonical id.
    """
    from fluxomics_data_converter.link.aliases import alias_map

    return alias_map(doc)


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
        aliases = _alias_map(doc)
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

    from fluxomics_data_converter.link import compare, from_fluxo_dataset

    if len(docs) != 2:
        print("--inputs compares exactly two datasets", file=sys.stderr)
        return 1
    models = [
        from_fluxo_dataset(FluxoDataset.model_validate(doc)) for _, doc in docs
    ]
    report = compare(models[0], models[1], rtol=args.rtol, atol=args.atol)
    print(report.summary(limit=args.limit))
    return 0 if report.equivalent else 1


def _record_root(record: Path) -> Path:
    return record.resolve().parent


def _save_record(ds: Any, record: Path, header: str, force: bool) -> int:
    """Validate a record and write it unless it has errors.

    Returns:
        Exit code.
    """
    from fluxomics_data_converter.link import record_document

    doc = record_document(ds)
    if not _report(doc, benchmark=None, label=str(record)) and not force:
        print(
            f"{record} not updated: fix the errors above or pass --force",
            file=sys.stderr,
        )
        return 1
    _write_yaml(doc, record, header)
    print(f"updated {record}")
    return 0


def _print_check(ti: Any, check: Any) -> None:
    status = "same problem" if check.equivalent else "DIFFERS"
    print(
        f"  {ti.tool} {ti.path}: {status} "
        f"({len(check.differences)} difference(s), "
        f"{len(check.expected_differences)} by design)"
    )
    for line in check.differences[:5]:
        print(f"      {line}")
    if len(check.differences) > 5:
        print(f"      ... and {len(check.differences) - 5} more (see record)")


def _write_inputs(
    data: Any, tools: list[str], out: Path, stem: str, experiment: str | None
) -> list[Any]:
    """Write each tool's input under ``out/<tool>/``.

    Returns:
        The written inputs.
    """
    from fluxomics_data_converter.link import write_tool_input
    from fluxomics_data_converter.link.tool_inputs import SINGLE_CONFIGURATION

    names = [e.name for e in data.experiments]
    written = []
    for tool in tools:
        if tool not in SINGLE_CONFIGURATION:
            target = out / tool / f"{stem}.fml"
            written.append(
                write_tool_input(data, tool, target, experiment=experiment)
            )
            continue
        for name in [experiment] if experiment else names or [None]:
            label = f"{stem}_{name}" if len(names) > 1 else stem
            target = out / tool / label
            written.append(
                write_tool_input(data, tool, target, experiment=name)
            )
    for ti in written:
        print(f"wrote {ti.tool} input: {ti.path}")
    return written


def _register_source(
    ds: Any, source: Any, root: Path, reference: Any
) -> tuple[Any, list[str]]:
    """Make sure the record holds the files a conversion read.

    A source already recorded (same path and checksum) is reused; another
    one is checked against the record like any hand-written input.

    Returns:
        The record and the ids of the source's files.
    """
    from fluxomics_data_converter.link import (
        check_tool_input,
        infer_aliases,
        record_tool_input,
    )
    from fluxomics_data_converter.link.files import file_checksum, record_path

    known = {(f.path, f.checksum): f.id for f in ds.source_files or []}
    keys = [
        (record_path(path, root), file_checksum(path))
        for path, _, _ in source.files()
    ]
    if keys and all(k in known for k in keys):
        return ds, [known[k] for k in keys]
    match = infer_aliases(reference, source.read())
    check = check_tool_input(reference, source, id_map=match.id_map)
    _print_check(source, check)
    ds = record_tool_input(
        ds, source, check, root=root, generated=False, aliases=match
    )
    known = {(f.path, f.checksum): f.id for f in ds.source_files or []}
    return ds, [known[k] for k in keys if k in known]


def cmd_convert(args: argparse.Namespace) -> int:
    """Translate a tool input into other tools' inputs; optionally link them.

    Returns:
        Exit code.
    """
    from fluxomics_data_converter.link import (
        ToolInput,
        check_tool_input,
        from_fluxo_dataset,
        load_record,
        record_tool_input,
        to_fluxo_dataset,
    )

    source = ToolInput.locate(Path(args.input), args.format)
    data = source.read()
    stem = source.path.stem if source.path.suffix else source.path.name
    try:
        written = _write_inputs(
            data, args.to, Path(args.output), stem, args.experiment
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    if not args.link:
        return 0

    record = Path(args.link)
    root = _record_root(record)
    if record.exists():
        ds = load_record(record)
        reference = from_fluxo_dataset(ds)
        ds, source_ids = _register_source(ds, source, root, reference)
    else:
        ds = to_fluxo_dataset(
            data,
            dataset_id=args.dataset_id or stem.replace(".", "_"),
            source_path=source.path,
            source_format=source.fmt,
            dataset_type="catalog",
            data_origin=args.data_origin,
            source_root=root,
        )
        source_ids = [f.id for f in ds.source_files or []]
        reference = from_fluxo_dataset(ds)
    print("checking each input against the record:")
    for ti in written:
        check = check_tool_input(reference, ti)
        _print_check(ti, check)
        ds = record_tool_input(
            ds, ti, check, root=root, generated=True, derived_from=source_ids
        )
    return _save_record(ds, record, f"FluxoLink record for {stem}.", args.force)


def cmd_export(args: argparse.Namespace) -> int:
    """Write a tool's input from a record's model and record it there.

    Returns:
        Exit code.
    """
    from fluxomics_data_converter.link import (
        check_tool_input,
        from_fluxo_dataset,
        load_record,
        record_tool_input,
        write_tool_input,
    )

    record = Path(args.dataset)
    ds = load_record(record)
    data = from_fluxo_dataset(ds, network_id=args.network)
    try:
        ti = write_tool_input(
            data, args.to, Path(args.output), experiment=args.experiment
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"wrote {args.to} input to {ti.path}")
    if args.no_link:
        return 0
    check = check_tool_input(data, ti)
    _print_check(ti, check)
    ds = record_tool_input(
        ds,
        ti,
        check,
        root=_record_root(record),
        generated=True,
        network_id=args.network,
    )
    return _save_record(ds, record, f"FluxoLink record {ds.id}.", args.force)


def cmd_add_input(args: argparse.Namespace) -> int:
    """Check a tool input written elsewhere and add it to a record.

    Returns:
        Exit code.
    """
    from fluxomics_data_converter.link import (
        ToolInput,
        check_tool_input,
        from_fluxo_dataset,
        infer_aliases,
        load_record,
        record_tool_input,
    )

    record = Path(args.record)
    ds = load_record(record)
    reference = from_fluxo_dataset(ds, network_id=args.network)
    ti = ToolInput.locate(Path(args.input), args.format, args.experiment)
    match = infer_aliases(reference, ti.read())
    try:
        check = check_tool_input(reference, ti, id_map=match.id_map)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    _print_check(ti, check)
    renamed = len(match.metabolites) + len(match.reactions)
    if renamed:
        print(f"  {renamed} id(s) differ from the record's; stored as aliases")
    ds = record_tool_input(
        ds,
        ti,
        check,
        root=_record_root(record),
        generated=False,
        aliases=match,
        software=args.software,
        software_version=args.software_version,
        network_id=args.network,
    )
    return _save_record(ds, record, f"FluxoLink record {ds.id}.", args.force)


def cmd_add_results(args: argparse.Namespace) -> int:
    """Read a tool's results and attach them to the record's planned run.

    Returns:
        Exit code.
    """
    from fluxolink.datamodel.fluxolink import SimulationRun

    from fluxomics_data_converter.link import (
        add_results,
        load_record,
        result_files,
    )

    record = Path(args.record)
    ds = load_record(record)
    path = Path(args.results)
    kind = args.tool or ("inca" if path.suffix == ".mat" else "influx_si")
    run = SimulationRun.model_validate(read_results(path, kind, path))
    if args.configuration:
        run.configuration_name = args.configuration
    tool = "INCA" if kind == "inca" else "influx_si"
    ds = add_results(
        ds,
        run,
        output_files=result_files(path, tool),
        root=_record_root(record),
    )
    return _save_record(ds, record, f"FluxoLink record {ds.id}.", args.force)


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
        description=(
            "Translate 13C-MFA tool inputs and link them in FluxoLink records."
        ),
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
        "--source-root",
        help="directory file paths are relative to (default: the record's)",
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
    compare.add_argument("--rtol", type=float, default=1e-6)
    compare.add_argument("--atol", type=float, default=1e-9)
    compare.add_argument(
        "--limit", type=int, default=20, help="differences shown"
    )
    compare.set_defaults(func=cmd_compare)

    convert = sub.add_parser(
        "convert", help="tool input -> other tools' inputs (+ record)"
    )
    convert.add_argument(
        "input", help=".fml, .netw, FreeFlux dir/table, INCA .m"
    )
    convert.add_argument("--format", choices=FORMATS)
    convert.add_argument("--to", nargs="+", choices=WRITABLE, required=True)
    convert.add_argument(
        "-o", "--output", required=True, help="directory; one folder per tool"
    )
    convert.add_argument(
        "--experiment", help="only this configuration (default: all)"
    )
    convert.add_argument(
        "--link", help="record to add the inputs to (created if missing)"
    )
    convert.add_argument("--dataset-id", help="id of a new record")
    convert.add_argument(
        "--data-origin", choices=("experimental", "synthetic", "simulated")
    )
    convert.add_argument(
        "--force", action="store_true", help="write despite errors"
    )
    convert.set_defaults(func=cmd_convert)

    export = sub.add_parser("export", help="record -> tool input (recorded)")
    export.add_argument("dataset")
    export.add_argument("--to", choices=WRITABLE, required=True)
    export.add_argument("-o", "--output", required=True)
    export.add_argument(
        "--network", help="network id if the dataset has several"
    )
    export.add_argument(
        "--experiment",
        help="configuration to write (MTF and FreeFlux hold one)",
    )
    export.add_argument(
        "--no-link",
        action="store_true",
        help="do not add the written files to the record",
    )
    export.add_argument(
        "--force", action="store_true", help="write despite errors"
    )
    export.set_defaults(func=cmd_export)

    add_input = sub.add_parser(
        "add-input", help="check a hand-written tool input, add it to a record"
    )
    add_input.add_argument("record")
    add_input.add_argument("input")
    add_input.add_argument("--format", choices=FORMATS)
    add_input.add_argument(
        "--experiment", help="configuration an MTF/FreeFlux input holds"
    )
    add_input.add_argument("--network")
    add_input.add_argument(
        "--software", help="tool (default: the format's own)"
    )
    add_input.add_argument("--software-version")
    add_input.add_argument(
        "--force", action="store_true", help="write despite errors"
    )
    add_input.set_defaults(func=cmd_add_input)

    add_res = sub.add_parser(
        "add-results", help="attach a tool's results to a record"
    )
    add_res.add_argument("record")
    add_res.add_argument(
        "results", help="influx_si <prefix>_res directory or INCA .mat"
    )
    add_res.add_argument("--tool", choices=("influx_si", "inca"))
    add_res.add_argument(
        "--configuration", help="configuration the results are for"
    )
    add_res.add_argument(
        "--force", action="store_true", help="write despite errors"
    )
    add_res.set_defaults(func=cmd_add_results)

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
