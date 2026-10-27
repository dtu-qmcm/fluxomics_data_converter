"""Write, check and record the input files each tool is given.

The converter translates directly between tools (``io``); this module
ties the resulting files to a record:

1. :func:`write_tool_input` writes one tool's input from a model;
2. :func:`check_tool_input` parses that input back and compares it with the
   record's model, separating the differences a writer makes by design
   (a drain for a terminal pool, an explicit unlabelled input) from real
   losses;
3. :func:`record_tool_input` adds the files (with checksums and
   ``derived_from``), the check and one planned simulation run per
   configuration to the record, plus aliases for a hand-written input
   whose ids differ.

A hand-written input goes through the same steps without the first one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from fluxolink.datamodel import fluxolink as fl

from ..core.core import FluxomicsData
from ..io import format_conventions as conv
from .aliases import AliasMatch, add_aliases
from .equivalence import Difference, EquivalenceReport, compare
from .files import (
    FileEntry,
    file_checksum,
    record_path,
    source_file,
    tool_files,
)

#: Tool key -> ``InputFormatEnum`` value of its input.
TOOL_FORMATS: Dict[str, str] = {
    "fluxml": conv.FLUXML,
    "mtf": conv.MTF,
    "freeflux": conv.FREEFLUX_TSV,
    "inca": conv.INCA,
}

#: Tools the converter can write input for.
WRITABLE_TOOLS = ("fluxml", "mtf", "freeflux")

#: Tools whose input holds a single configuration.
SINGLE_CONFIGURATION = frozenset({"mtf", "freeflux"})

#: Value of ``generated_by`` / ``checked_by``.
GENERATOR = "fluxomics_data_converter"


def generator_version() -> str:
    """Return the converter's version.

    Returns:
        The installed distribution version, else the package's.
    """
    try:
        from importlib.metadata import version

        return version("fluxomics-data-converter")
    except Exception:  # noqa: BLE001 - not installed as a distribution
        from .. import __version__

        return __version__


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def detect_tool(path: Path, tool: Optional[str] = None) -> str:
    """Return the tool key of an input path.

    Args:
        path: A FluxML or INCA file, an MTF prefix or ``.netw`` file, or a
            FreeFlux directory or table.
        tool: Explicit tool key, overriding detection.

    Returns:
        One of :data:`TOOL_FORMATS`.

    Raises:
        ValueError: If the tool cannot be determined.
    """
    if tool:
        if tool not in TOOL_FORMATS:
            raise ValueError(
                f"unknown tool {tool!r}; expected one of {sorted(TOOL_FORMATS)}"
            )
        return tool
    path = Path(path)
    if path.is_dir():
        return "freeflux"
    by_suffix = {
        ".fml": "fluxml",
        ".xml": "fluxml",
        ".netw": "mtf",
        ".tsv": "freeflux",
        ".csv": "freeflux",
        ".xlsx": "freeflux",
        ".m": "inca",
    }
    if path.suffix.lower() in by_suffix:
        return by_suffix[path.suffix.lower()]
    if path.with_suffix(".netw").exists():
        return "mtf"
    raise ValueError(
        f"cannot tell which tool {path} is for; "
        f"pass one of {sorted(TOOL_FORMATS)}"
    )


def tool_software(tool: str) -> str:
    """Return the software name that runs a tool's input.

    Returns:
        A ``MFASoftwareEnum`` value, also the alias namespace.
    """
    return conv.NATIVE_SOFTWARE[TOOL_FORMATS[tool]]


@dataclass
class ToolInput:
    """One tool's input on disk.

    Attributes:
        tool: Tool key (:data:`TOOL_FORMATS`).
        path: What the tool and the parser are given: a FluxML or INCA
            file, an MTF prefix, or a FreeFlux directory.
        experiment: The configuration it holds, for single-configuration
            tools; ``None`` when it holds every configuration of the model.
        data: The model parsed back from the files, once read.
    """

    tool: str
    path: Path
    experiment: Optional[str] = None
    data: Optional[FluxomicsData] = field(default=None, repr=False)

    @classmethod
    def locate(
        cls,
        path: Path,
        tool: Optional[str] = None,
        experiment: Optional[str] = None,
    ) -> "ToolInput":
        """Describe an existing input from any of its paths.

        Args:
            path: A FluxML or INCA file, an MTF prefix or ``.netw`` file, or
                a FreeFlux directory or one of its tables.
            tool: Tool key; detected from ``path`` when omitted.
            experiment: The configuration a single-configuration input
                holds, when the reference model has several.

        Returns:
            The input, with ``path`` normalised to what the parser takes.
        """
        path = Path(path)
        tool = detect_tool(path, tool)
        if tool == "mtf" and path.suffix == ".netw":
            path = path.with_suffix("")
        elif tool == "freeflux" and path.is_file():
            path = path.parent
        return cls(tool=tool, path=path, experiment=experiment)

    @property
    def fmt(self) -> str:
        """Return the ``InputFormatEnum`` value of the input."""
        if self.tool == "freeflux":
            return conv.freeflux_format(self.path) or conv.FREEFLUX_TSV
        return TOOL_FORMATS[self.tool]

    def files(self) -> List[FileEntry]:
        """List the physical files of the input with their roles.

        Returns:
            One entry per existing file.
        """
        return tool_files(self.path, self.fmt, self.data)

    def read(self) -> FluxomicsData:
        """Parse the input with the converter (cached).

        Returns:
            The parsed model.
        """
        if self.data is None:
            self.data = read_tool_input(self.path, self.tool)
        return self.data


def read_tool_input(path: Path, tool: str) -> FluxomicsData:
    """Parse a tool input with the matching converter parser.

    Args:
        path: The input (see :class:`ToolInput`).
        tool: Tool key.

    Returns:
        The parsed model.
    """
    from .. import io

    path = Path(path)
    if tool == "fluxml":
        return io.parse_fluxml_file(str(path))
    if tool == "mtf":
        prefix = path.with_suffix("") if path.suffix == ".netw" else path
        return io.parse_mtf(str(prefix))
    if tool == "freeflux":
        return io.parse_freeflux(str(path if path.is_dir() else path.parent))
    return io.parse_inca(str(path))


def write_tool_input(
    data: FluxomicsData,
    tool: str,
    out: Path,
    *,
    experiment: Optional[str] = None,
    natural_abundance: Optional[bool] = None,
) -> ToolInput:
    """Write one tool's input from a model.

    Args:
        data: The model.
        tool: One of :data:`WRITABLE_TOOLS`.
        out: A ``.fml`` file (FluxML), a prefix (MTF) or a directory
            (FreeFlux).
        experiment: Configuration to write; required for MTF and FreeFlux
            when the model has several.
        natural_abundance: FluxML only: write ``purity`` so 13CFLUX adds
            natural abundance. ``None`` turns it on exactly when the model's
            tracers carry a purity.

    Returns:
        The written input.

    Raises:
        ValueError: If the tool cannot be written or the configuration is
            ambiguous.
    """
    from .. import io

    if tool not in WRITABLE_TOOLS:
        raise ValueError(
            f"the converter cannot write {tool} input; "
            f"writable: {', '.join(WRITABLE_TOOLS)}"
        )
    names = [e.name for e in data.experiments]
    if tool in SINGLE_CONFIGURATION:
        if experiment is None and len(names) > 1:
            raise ValueError(
                f"{tool} input holds one configuration; choose one of "
                f"{', '.join(names)}"
            )
        experiment = experiment or (names[0] if names else None)
    elif experiment is not None:
        data = data.model_copy(
            update={
                "experiments": [
                    e for e in data.experiments if e.name == experiment
                ]
            }
        )
    out = Path(out)
    if tool == "fluxml":
        if natural_abundance is None:
            natural_abundance = any(
                label.purity is not None
                for exp in data.experiments
                for tracer in exp.tracers
                for label in tracer.labels
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        io.write_fluxml(data, str(out), natural_abundance=natural_abundance)
    elif tool == "mtf":
        out.parent.mkdir(parents=True, exist_ok=True)
        io.write_mtf(data, str(out), experiment_name=experiment)
    else:
        out.mkdir(parents=True, exist_ok=True)
        io.write_freeflux(data, str(out), experiment_name=experiment)
    return ToolInput(tool=tool, path=out, experiment=experiment)


# -- checking ------------------------------------------------------------------


@dataclass
class InputCheck:
    """Result of :func:`check_tool_input`.

    Attributes:
        report: The full equivalence report.
        differences: Unexpected differences, one ``"[kind] key: detail"``
            line each.
        expected_differences: Differences the writer makes by design.
    """

    report: EquivalenceReport
    differences: List[str] = field(default_factory=list)
    expected_differences: List[str] = field(default_factory=list)

    @property
    def equivalent(self) -> bool:
        """True when no unexpected difference was found."""
        return not self.differences


def _restrict(reference: FluxomicsData, ti: ToolInput) -> FluxomicsData:
    """Keep the reference configurations the input holds.

    Returns:
        The reference, with only the matching experiments.

    Raises:
        ValueError: If a single-configuration input cannot be matched.
    """
    names = [e.name for e in reference.experiments]
    wanted = ti.experiment
    if wanted is None and ti.tool in SINGLE_CONFIGURATION and names:
        parsed = [e.name for e in ti.read().experiments]
        common = [n for n in parsed if n in names]
        if len(names) == 1:
            common = names
        if len(common) != 1:
            raise ValueError(
                f"{ti.path} holds one configuration; say which of "
                f"{', '.join(names)} it is"
            )
        wanted = common[0]
        ti.experiment = wanted
    if wanted is None:
        return reference
    if wanted not in names:
        raise ValueError(
            f"configuration {wanted!r} is not in the record "
            f"({', '.join(names)})"
        )
    return reference.model_copy(
        update={
            "experiments": [
                e for e in reference.experiments if e.name == wanted
            ]
        }
    )


def _consumed(data: FluxomicsData) -> Set[str]:
    out: Set[str] = set()
    for r in data.model.reactions:
        out.update(r.reactants)
        if r.reversibility:
            out.update(r.products)
    return out


def _produced(data: FluxomicsData) -> Set[str]:
    out: Set[str] = set()
    for r in data.model.reactions:
        out.update(r.products)
        if r.reversibility:
            out.update(r.reactants)
    return out


class _ByDesign:
    """Recognise the differences a converter writer makes on purpose."""

    def __init__(
        self, reference: FluxomicsData, written: FluxomicsData, tool: str
    ):
        """Index both models."""
        self.tool = tool
        self.ref_mets = set(reference.model.metabolites.ids)
        self.written = written
        self.terminal = self.ref_mets - _consumed(reference)
        self.sources = self.ref_mets - _produced(reference)
        self.added_sinks = {
            m
            for m in written.model.metabolites.ids
            if m not in self.ref_mets and m not in _consumed(written)
        }
        self.sink_products: Set[str] = set()

    def note(self, diff: Difference) -> Optional[str]:
        """Explain a by-design difference.

        Returns:
            The explanation, or ``None`` for an unexpected difference.
        """
        key, kind = diff.key, diff.kind
        only_written = diff.message == "only in b"
        if kind == "metabolite" and only_written and key in self.added_sinks:
            return f"sink pool added by the {self.tool} writer"
        if kind == "reaction" and only_written:
            reaction = self.written.model.reactions.get_by_id(key)
            reactants = set(reaction.reactants)
            if (
                len(reactants) == 1
                and reactants <= self.terminal
                and set(reaction.products) <= self.added_sinks
            ):
                return f"drain added for terminal pool {reaction.reactants[0]}"
        if kind == "stoichiometry" and isinstance(diff.a, dict):
            a, b = diff.a, diff.b or {}
            extra = {m: c for m, c in b.items() if m not in a}
            if (
                extra
                and all(b.get(m) == c for m, c in a.items())
                and set(extra) <= self.added_sinks
                and all(c > 0 for c in extra.values())
            ):
                self.sink_products.add(key)
                return f"sink pool {', '.join(sorted(extra))} added as product"
        if (
            kind == "atom_transition"
            and key in self.sink_products
            and diff.message == "atom transitions only in b"
        ):
            return "atoms followed into the added sink pool"
        if kind == "tracer" and diff.message == "tracer input only in b":
            metabolite = key.split("/", 1)[-1]
            if metabolite in self.sources and self._unlabelled(metabolite):
                return f"unlabelled input declared for source pool {metabolite}"
        return None

    def _unlabelled(self, metabolite: str) -> bool:
        for exp in self.written.experiments:
            for tracer in exp.tracers:
                if tracer.metabolite == metabolite:
                    return all(
                        set(label.labeled_pattern) <= {"0"}
                        and label.purity is None
                        for label in tracer.labels
                    )
        return False


def check_tool_input(
    reference: FluxomicsData,
    ti: ToolInput,
    *,
    id_map: Optional[Dict[str, str]] = None,
) -> InputCheck:
    """Parse a tool input back and compare it with the reference model.

    Args:
        reference: The record's model.
        ti: The input.
        id_map: Tool id -> canonical id renames (see :mod:`.aliases`).

    Returns:
        The check, with by-design differences listed separately.
    """
    reference = _restrict(reference, ti)
    written = ti.read()
    report = compare(reference, written, id_map=id_map)
    by_design = _ByDesign(reference, written, ti.tool)
    # Stoichiometry first: it tells which atom-map differences of the same
    # reaction come from an added sink pool.
    notes = {
        i: by_design.note(diff)
        for i, diff in enumerate(report.differences)
        if diff.kind == "stoichiometry"
    }
    check = InputCheck(report=report)
    for i, diff in enumerate(report.differences):
        note = notes[i] if i in notes else by_design.note(diff)
        if note is None:
            check.differences.append(str(diff))
        else:
            check.expected_differences.append(
                f"[{diff.kind}] {diff.key}: {note}"
            )
    return check


# -- recording -----------------------------------------------------------------


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "file"


def _locked_ids(ds: fl.FluxoDataset) -> Set[str]:
    """Return the file ids that finished runs point at.

    Returns:
        Ids whose entries must not be overwritten.
    """
    locked: Set[str] = set()
    for run in ds.simulation_runs or []:
        if run.run_status not in (None, "planned"):
            locked.update(run.input_files or [])
            locked.update(run.output_files or [])
    return locked


def _free_id(wanted: str, taken: Iterable[str]) -> str:
    taken = set(taken)
    if wanted not in taken:
        return wanted
    n = 2
    while f"{wanted}.v{n}" in taken:
        n += 1
    return f"{wanted}.v{n}"


def upsert_files(
    ds: fl.FluxoDataset,
    entries: List[FileEntry],
    *,
    root: Optional[Path],
    **fields,
) -> List[str]:
    """Add or refresh ``SourceFile`` entries for files on disk.

    A file already recorded under the same path and checksum keeps its
    entry. A changed file replaces its entry, unless a finished run points
    at it; then it gets a new id, so the run keeps its provenance.

    Args:
        ds: The record, changed in place.
        entries: Files with roles and formats.
        root: Directory the recorded paths are relative to.
        **fields: Further ``SourceFile`` slots for every file.

    Returns:
        The ids of the entries, in order.
    """
    files = list(ds.source_files or [])
    locked = _locked_ids(ds)
    ids: List[str] = []
    for path, roles, fmt in entries:
        recorded = record_path(path, root)
        digest = file_checksum(path)
        same = [f for f in files if f.path == recorded and f.checksum == digest]
        if same:
            ids.append(same[0].id)
            continue
        wanted = f"{ds.id}.src.{_slug(recorded)}"
        index = next((i for i, f in enumerate(files) if f.id == wanted), None)
        if index is not None and wanted in locked:
            wanted, index = _free_id(wanted, [f.id for f in files]), None
        derived = [d for d in fields.get("derived_from") or [] if d != wanted]
        entry = source_file(
            path,
            file_id=wanted,
            fmt=fmt,
            roles=roles,
            recorded_path=recorded,
            **{**fields, "derived_from": derived},
        )
        if index is None:
            files.append(entry)
        else:
            files[index] = entry
        ids.append(wanted)
    ds.source_files = files
    return ids


def _record_experiments(
    ds: fl.FluxoDataset, names: Iterable[str]
) -> List[fl.LabelingExperiment]:
    wanted = list(names)
    found = []
    for exp in ds.experiments or []:
        name = exp.configuration_name or exp.name or exp.id.rsplit(".", 1)[-1]
        if name in wanted:
            found.append(exp)
    return found


def _network_id(
    ds: fl.FluxoDataset, network_id: Optional[str]
) -> Optional[str]:
    if network_id:
        return network_id
    networks = ds.networks or []
    return networks[0].id if len(networks) == 1 else None


def _planned_run(
    ds: fl.FluxoDataset,
    *,
    run_id: str,
    tool: str,
    fmt: str,
    file_ids: List[str],
    experiment: Optional[fl.LabelingExperiment],
    network_id: Optional[str],
    software: Optional[str],
    software_version: Optional[str],
) -> None:
    """Add or refresh the planned run of one tool on one configuration."""
    runs = list(ds.simulation_runs or [])
    existing = next((r for r in runs if r.id == run_id), None)
    if existing is not None and existing.run_status in (None, "planned"):
        existing.input_files = file_ids
        ds.simulation_runs = runs
        return
    if existing is not None:
        run_id = _free_id(run_id, [r.id for r in runs])
    names = {v.value for v in fl.MFASoftwareEnum}
    name = software or tool_software(tool)
    info = {"id": f"{run_id}.software"}
    if name in names:
        info["software_name"] = name
    else:
        info.update(software_name="other", software_name_other=name)
    if software_version:
        info["version"] = software_version
    fields = {
        "id": run_id,
        "run_status": "planned",
        "software": fl.SoftwareInfo(**info),
        "input_format": fmt,
        "input_files": file_ids,
    }
    if network_id:
        fields["network_model_id"] = network_id
    investigations = ds.investigations or []
    if len(investigations) == 1:
        fields["investigation_id"] = investigations[0].id
    if experiment is not None:
        fields["experiment_id"] = [experiment.id]
        fields["configuration_name"] = (
            experiment.configuration_name or experiment.name
        )
        if experiment.stationary is not None:
            fields["fluxomics_method"] = (
                "IST_MFA" if experiment.stationary else "INST_MFA"
            )
    runs.append(fl.SimulationRun(**fields))
    ds.simulation_runs = runs


def record_tool_input(
    ds: fl.FluxoDataset,
    ti: ToolInput,
    check: InputCheck,
    *,
    root: Optional[Path],
    generated: bool,
    derived_from: Iterable[str] = (),
    aliases: Optional[AliasMatch] = None,
    software: Optional[str] = None,
    software_version: Optional[str] = None,
    network_id: Optional[str] = None,
) -> fl.FluxoDataset:
    """Add a tool input, its check and its planned runs to a record.

    Args:
        ds: The record.
        ti: The input (already checked).
        check: Result of :func:`check_tool_input` for it.
        root: Directory the recorded paths are relative to (normally the
            record's directory).
        generated: Whether the converter wrote the files.
        derived_from: Ids of the files they were generated from.
        aliases: Renames of a hand-written input, stored as aliases.
        software: Tool that will run it (default: the format's own).
        software_version: Version of that tool.
        network_id: Network the input encodes, when the record has several.

    Returns:
        The updated record (a copy).
    """
    ds = ds.model_copy(deep=True)
    extra = {}
    if generated:
        extra = {
            "generated_by": GENERATOR,
            "generator_version": generator_version(),
            "generation_date": _now(),
        }
    file_ids = upsert_files(
        ds,
        ti.files(),
        root=root,
        derived_from=list(derived_from),
        **extra,
    )
    net_id = _network_id(ds, network_id)
    if ti.experiment is not None:
        names = [ti.experiment]
    else:
        names = [e.name for e in ti.read().experiments]
    experiments = _record_experiments(ds, names)

    check_fields = {
        "id": f"{ds.id}.check.{_slug(record_path(ti.path, root))}",
        "files": file_ids,
        "equivalent": check.equivalent,
        "checked_by": GENERATOR,
        "checker_version": generator_version(),
        "check_date": _now(),
    }
    if net_id:
        check_fields["network_model_id"] = net_id
    if experiments:
        check_fields["experiment_id"] = [e.id for e in experiments]
    if check.differences:
        check_fields["differences"] = check.differences
    if check.expected_differences:
        check_fields["expected_differences"] = check.expected_differences
    checks = [
        c for c in ds.equivalence_checks or [] if c.id != check_fields["id"]
    ]
    checks.append(fl.EquivalenceCheck(**check_fields))
    ds.equivalence_checks = checks

    for experiment in experiments or [None]:
        base = experiment.id if experiment is not None else ds.id
        _planned_run(
            ds,
            run_id=f"{base}.{ti.tool}.run",
            tool=ti.tool,
            fmt=ti.fmt,
            file_ids=file_ids,
            experiment=experiment,
            network_id=net_id,
            software=software,
            software_version=software_version,
        )
    if aliases is not None:
        ds = add_aliases(ds, tool_software(ti.tool), aliases, net_id)
    return ds
