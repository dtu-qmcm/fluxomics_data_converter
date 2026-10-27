"""Materialise a converter model as a FluxoLink ``FluxoDataset``.

Entry point::

    from fluxomics_data_converter.io import parse_fluxml_file
    from fluxomics_data_converter.link import save_record, to_fluxo_dataset

    data = parse_fluxml_file("EC.fml")
    ds = to_fluxo_dataset(data, dataset_id="ec", source_path=Path("EC.fml"))
    save_record(ds, "ec.yaml")

Mapping rules
-------------
The mapping copies what the converter model holds and nothing else. Where a
parser fills a field with its own default rather than with source content,
the default is not published:

- MTF, FreeFlux and INCA never carry ``Simulation.type``/``method``,
  ``Tracers.type`` or ``Group.scale``; those are left out for these formats
  (see :func:`.format_conventions.converter_defaults_are_data`).
- Atom-map variant weights become ``AtomTransitionVariant.ratio`` only when
  they are not uniform. The FluxML parser assigns ``1/n`` when the file has
  no ``ratio``; FreeFlux and INCA always assign ``1/n``; single maps carry
  ``1.0``. Uniform weights are the converter's (and the schema's) default
  reading of an absent ratio, so publishing them would turn an assumption
  into data.
- The MTF parser invents flux bounds (``-1000..1000`` for free fluxes,
  ``0..1e6`` for free pool sizes, ``lo = hi = value`` for constrained ones)
  and the FreeFlux parser sets ``lo = hi = value``. Those placeholders are
  dropped; :func:`~.model_from_record.from_fluxo_dataset` restores them.
- ``Metadata.strain`` is published only inside a cultivation ``Condition``
  supplied by the metadata sidecar; the adapter never creates one.

Identifiers
-----------
Native ids are kept for metabolites and reactions. Everything else is
scoped under its owner with ``.`` separators so ids stay unique when one
source has several configurations (``EC.fml`` reuses ``fp_0`` and
``ms_group_1`` across ``a_INST`` and ``b_INST``)::

    <dataset_id>.network                    MetabolicNetworkModel
    <dataset_id>.network.net1               network FluxConstraint
    <dataset_id>.<configuration>            LabelingExperiment
    <dataset_id>.<configuration>.<native>   MID / NMR / flux / pool-size
                                            measurement (native group or
                                            flux id)
    <dataset_id>.<configuration>.input.<metabolite>   LabeledSource
    <dataset_id>.<configuration>.run        SimulationRun
    <dataset_id>.src.<file name>            SourceFile
"""

from __future__ import annotations

import re
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fluxolink.datamodel import fluxolink as fl

from ..core.core import FluxomicsData, LabelingExperiments
from ..experiment.measurement import Datum, Group, Measurement
from ..model.atom_mapping import AtomTransition as ConverterAtomTransition
from ..model.constraint import ConstraintFormula, Constraints
from ..model.reaction import Reaction
from ..output.simulation import Simulation
from ..io import format_conventions as conv
from .files import source_file, tool_files

_STATE_SPACES = {
    "auto": "auto",
    "isotopomer": "isotopomer",
    "cumomer": "cumomer",
    "emu": "EMU",
}

_OPERATORS = {"=": "eq", "<=": "le", ">=": "ge", "<": "lt", ">": "gt"}

_VARIABLE_CODES = {"F": "free", "D": "dependent", "C": "constrained"}

_SIDECAR_KEYS = frozenset(
    {
        "dataset",
        "investigation",
        "investigations",
        "contributors",
        "publication",
        "publication_doi",
        "conditions",
        "experiment_conditions",
    }
)

_INCA_TIME_UNIT_RE = re.compile(r"INCA time unit:\s*([^;]+)")


def to_fluxo_dataset(
    data: FluxomicsData,
    *,
    dataset_id: str,
    source_path: Optional[Path] = None,
    source_format: Optional[str] = None,
    software: Optional[str] = None,
    software_version: Optional[str] = None,
    dataset_type: str = "benchmark",
    data_origin: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    source_root: Optional[Path] = None,
) -> fl.FluxoDataset:
    """Convert a converter model into a FluxoLink dataset.

    Args:
        data: The parsed model.
        dataset_id: Identifier of the dataset; prefixes every generated id.
        source_path: The native file, directory (FreeFlux) or prefix (MTF)
            the model was parsed from. Recorded as ``SourceFile`` entries
            with checksum and size.
        source_format: Native format (``FluxML``, ``MTF``, ``freeflux``,
            ``INCA``; case-insensitive). Inferred from ``source_path`` when
            omitted.
        software: Tool that runs the configurations. Defaults to the
            format's native tool (FluxML: 13CFLUX2, MTF: influx_si,
            FreeFlux: freeflux, INCA: INCA).
        software_version: Version of ``software``.
        dataset_type: ``"benchmark"`` or ``"catalog"``.
        data_origin: ``"experimental"``, ``"synthetic"`` or ``"simulated"``;
            set on every experiment.
        metadata: Optional sidecar with content no native file carries,
            merged verbatim: ``investigation`` (or ``investigations``),
            ``contributors``, ``publication`` (``{"doi": ...}``) or
            ``publication_doi``, ``conditions``, ``experiment_conditions``
            (configuration name -> condition id) and ``dataset`` (extra
            ``FluxoDataset`` slots such as ``name`` or ``creation_date``).
            A single investigation is linked to every experiment and run; a
            single condition to every experiment.
        source_root: Directory that ``SourceFile.path`` is made relative
            to. Defaults to the source's own directory (file names only).

    Returns:
        A validated :class:`fluxolink.datamodel.fluxolink.FluxoDataset`.

    Raises:
        ValueError: If the sidecar has unknown keys, or the model holds
            content the schema cannot represent (a reaction without
            participants at all).
    """
    fmt = conv.canonical_format(source_format, source_path)
    sidecar = dict(metadata or {})
    unknown = set(sidecar) - _SIDECAR_KEYS
    if unknown:
        raise ValueError(
            f"Unknown metadata sidecar key(s) {sorted(unknown)}; expected "
            f"a subset of {sorted(_SIDECAR_KEYS)}"
        )

    builder = _Builder(
        data=data,
        dataset_id=dataset_id,
        fmt=fmt,
        source_path=Path(source_path) if source_path is not None else None,
        source_root=Path(source_root) if source_root is not None else None,
        software=software,
        software_version=software_version,
        data_origin=data_origin,
    )
    investigations = _investigations(sidecar, dataset_id)
    conditions = _conditions(sidecar, data)
    builder.investigation_id = (
        investigations[0].id if len(investigations) == 1 else None
    )
    builder.experiment_conditions = _experiment_conditions(
        sidecar, conditions, data
    )

    source_files = builder.source_files()
    network = builder.network()
    experiments, runs = builder.experiments()

    fields: Dict[str, Any] = {
        "id": dataset_id,
        "schema_version": fl.version,
        "dataset_type": dataset_type,
    }
    if source_files:
        fields["source_files"] = source_files
    if conditions:
        fields["conditions"] = conditions
    if investigations:
        fields["investigations"] = investigations
    fields["networks"] = [network]
    if experiments:
        fields["experiments"] = experiments
    if runs:
        fields["simulation_runs"] = runs
    fields.update(sidecar.get("dataset") or {})
    return fl.FluxoDataset(**fields)


# -- sidecar -----------------------------------------------------------------


def _investigations(
    sidecar: Dict[str, Any], dataset_id: str
) -> List[fl.Investigation]:
    """Build investigations from the sidecar.

    Returns:
        The investigations, possibly empty.

    Raises:
        ValueError: If contributors or a publication are given without
            exactly one investigation to attach them to.
    """
    raw: List[Dict[str, Any]] = [
        dict(item) for item in sidecar.get("investigations") or []
    ]
    if sidecar.get("investigation"):
        raw.append(dict(sidecar["investigation"]))
    extras: Dict[str, Any] = {}
    if sidecar.get("contributors"):
        extras["contributors"] = list(sidecar["contributors"])
    publication = sidecar.get("publication") or {}
    doi = sidecar.get("publication_doi") or publication.get("doi")
    if doi:
        extras["publication_doi"] = doi
    if extras and len(raw) != 1:
        raise ValueError(
            "contributors / publication need exactly one investigation in "
            "the metadata sidecar"
        )
    investigations = []
    for number, item in enumerate(raw, start=1):
        item.setdefault("id", f"{dataset_id}.investigation{number}")
        if extras:
            contributors = list(item.get("contributors") or [])
            contributors.extend(extras.get("contributors", []))
            if contributors:
                item["contributors"] = contributors
            if "publication_doi" in extras:
                item.setdefault("publication_doi", extras["publication_doi"])
        investigations.append(fl.Investigation(**item))
    return investigations


def _conditions(
    sidecar: Dict[str, Any], data: FluxomicsData
) -> List[fl.Condition]:
    """Build conditions from the sidecar, adding the model's strain.

    ``Metadata.strain`` (FluxML ``<info><strain>``) fills ``strain`` of the
    single sidecar condition that lacks one; it never creates a condition.

    Returns:
        The conditions, possibly empty.
    """
    raw = [dict(item) for item in sidecar.get("conditions") or []]
    strain = data.metadata.strain if data.metadata else None
    if strain and len(raw) == 1 and not raw[0].get("strain"):
        raw[0]["strain"] = strain
    return [fl.Condition(**item) for item in raw]


def _experiment_conditions(
    sidecar: Dict[str, Any],
    conditions: List[fl.Condition],
    data: FluxomicsData,
) -> Dict[str, str]:
    """Map configuration names to condition ids.

    Returns:
        Configuration name to condition id.
    """
    mapping = dict(sidecar.get("experiment_conditions") or {})
    if not mapping and len(conditions) == 1:
        mapping = {exp.name: conditions[0].id for exp in data.experiments}
    return mapping


# -- builder -----------------------------------------------------------------


class _Builder:
    """Holds the conversion context and builds each part of the dataset."""

    def __init__(
        self,
        *,
        data: FluxomicsData,
        dataset_id: str,
        fmt: Optional[str],
        source_path: Optional[Path],
        source_root: Optional[Path],
        software: Optional[str],
        software_version: Optional[str],
        data_origin: Optional[str],
    ) -> None:
        """Store the conversion inputs."""
        self.data = data
        self.dataset_id = dataset_id
        self.fmt = fmt
        self.source_path = source_path
        self.source_root = source_root
        self.software = software or conv.NATIVE_SOFTWARE.get(fmt or "")
        self.software_version = software_version
        self.data_origin = data_origin
        self.native_settings = conv.converter_defaults_are_data(fmt)
        self.network_id = f"{dataset_id}.network"
        self.atom_counts = conv.effective_atom_counts(data.model)
        self.skipped_reactions = _unrepresentable_reactions(data)
        self.reaction_ids = frozenset(
            rid
            for r in data.model.reactions
            if r.id not in self.skipped_reactions
            for rid in [r.id, *(r.atom_transition_ids or [])]
        )
        self.investigation_id: Optional[str] = None
        self.experiment_conditions: Dict[str, str] = {}
        self.source_file_ids: List[str] = []
        self.time_units = self._time_units()

    # -- provenance --------------------------------------------------------

    def source_files(self) -> List[fl.SourceFile]:
        """Record the native files the model was parsed from.

        Returns:
            One ``SourceFile`` per existing native file.
        """
        if self.source_path is None:
            return []
        files = []
        for file_path, roles, fmt in tool_files(
            self.source_path, self.fmt, self.data
        ):
            file_id = f"{self.dataset_id}.src.{file_path.name}"
            self.source_file_ids.append(file_id)
            files.append(
                source_file(
                    file_path,
                    file_id=file_id,
                    fmt=fmt,
                    roles=roles,
                    recorded_path=self._relative(file_path),
                )
            )
        return files

    def _relative(self, file_path: Path) -> str:
        """Return the path recorded in ``SourceFile.path``.

        Returns:
            POSIX path relative to ``source_root`` (or the file name).
        """
        if self.source_root is not None:
            try:
                return (
                    file_path.resolve()
                    .relative_to(self.source_root.resolve())
                    .as_posix()
                )
            except ValueError:
                pass
        return file_path.name

    def _time_units(self) -> Dict[str, str]:
        """Find the time unit of each configuration.

        The converter model has no time-unit field, so for FluxML the unit
        is read from ``<configuration><measurement><mlabel><timeunit>`` of
        the source file; for INCA from the parser's ``INCA time unit``
        note in ``Metadata.comment``.

        Returns:
            Configuration name to time unit.
        """
        units: Dict[str, str] = {}
        if self.fmt == conv.INCA and self.data.metadata:
            match = _INCA_TIME_UNIT_RE.search(self.data.metadata.comment or "")
            if match:
                for exp in self.data.experiments:
                    units[exp.name] = match.group(1).strip()
        if (
            self.fmt == conv.FLUXML
            and self.source_path is not None
            and self.source_path.is_file()
        ):
            try:
                root = ET.parse(self.source_path).getroot()
            except ET.ParseError:
                return units
            for config in root.iter():
                if not config.tag.endswith("configuration"):
                    continue
                for elem in config.iter():
                    if elem.tag.endswith("timeunit") and elem.text:
                        units[config.get("name", "")] = elem.text.strip()
                        break
        return units

    # -- network -----------------------------------------------------------

    def network(self) -> fl.MetabolicNetworkModel:
        """Build the metabolic network.

        Returns:
            The network model.
        """
        model = self.data.model
        meta = self.data.metadata
        fields: Dict[str, Any] = {"id": self.network_id}
        if meta is not None:
            for slot, value in (
                ("name", meta.name),
                ("description", meta.comment),
                ("version", meta.version),
                ("modeler", meta.modeler),
                ("model_date", _iso_datetime(meta.date)),
            ):
                if value is not None:
                    fields[slot] = value
        if model.compartments:
            fields["compartments"] = sorted(model.compartments)
        if len(model.metabolites):
            fields["metabolites"] = [
                self._metabolite(m) for m in model.metabolites
            ]
        fields["reactions"] = [
            self._reaction(r)
            for r in model.reactions
            if r.id not in self.skipped_reactions
        ]
        flux, pools = self._constraints(self.data.constraints, self.network_id)
        if flux:
            fields["flux_constraints"] = flux
        if pools:
            fields["pool_size_constraints"] = pools
        return fl.MetabolicNetworkModel(**fields)

    def _metabolite(self, metabolite) -> fl.NetworkMetabolite:
        """Build one metabolite.

        Returns:
            The network metabolite.
        """
        annotations = {
            a.name.lower(): a.content for a in metabolite.annotations
        }
        fields: Dict[str, Any] = {"id": metabolite.id}
        name = metabolite.name or annotations.get("name")
        optional = {
            "name": name,
            "formula": metabolite.formula or None,
            "charge": metabolite.charge,
            "compartment": metabolite.compartment,
            "n_tracked_atoms": self.atom_counts.get(metabolite.id),
            "inchi": annotations.get("inchi"),
            "smiles": annotations.get("smiles"),
        }
        fields.update({k: v for k, v in optional.items() if v is not None})
        return fl.NetworkMetabolite(**fields)

    def _reaction(self, reaction: Reaction) -> fl.NetworkReaction:
        """Build one reaction with participants and atom transitions.

        Returns:
            The network reaction.
        """
        annotations = {a.name.lower(): a.content for a in reaction.annotations}
        fields: Dict[str, Any] = {
            "id": reaction.id,
            "reversibility": reaction.reversibility,
            "reactants": _participants(reaction, "reactants"),
        }
        name = reaction.name or annotations.get("name")
        if name:
            fields["name"] = name
        if annotations.get("pathway"):
            fields["pathway"] = annotations["pathway"]
        if reaction.products:
            fields["products"] = _participants(reaction, "products")
        transition = self.data.model.atom_mappings.get(reaction.id)
        if transition is not None and transition.maps:
            fields["atom_transition_variants"] = _variants(transition)
        return fl.NetworkReaction(**fields)

    def _constraints(
        self,
        constraints: Optional[Constraints],
        owner: str,
        skip: Iterable[Tuple[str, str, Optional[str]]] = (),
    ) -> Tuple[List[fl.FluxConstraint], List[fl.PoolSizeConstraint]]:
        """Build flux and pool-size constraints.

        Args:
            constraints: Converter constraints, or ``None``.
            owner: Id the constraint ids are scoped under.
            skip: ``(kind, expression, name)`` triples to leave out (the
                network-level constraints, which apply to every experiment
                already; the MTF parser attaches one object to both).

        Returns:
            The flux constraints and the pool-size constraints.
        """
        flux: List[fl.FluxConstraint] = []
        pools: List[fl.PoolSizeConstraint] = []
        if constraints is None:
            return flux, pools
        skipped = set(skip)
        for kind, block in _constraint_blocks(constraints):
            for number, formula in enumerate(block, start=1):
                if (kind, formula.expression, formula.name) in skipped:
                    continue
                operator = _operator(formula)
                fields: Dict[str, Any] = {
                    "id": f"{owner}.{kind}{number}",
                    "constraint_type": (
                        "equality" if operator == "eq" else "inequality"
                    ),
                    # MathML is kept verbatim; hydrate recognises its "<".
                    "expression": formula.expression,
                }
                if formula.name:
                    fields["name"] = formula.name
                if kind == "psize":
                    pools.append(fl.PoolSizeConstraint(**fields))
                else:
                    fields["flux_type"] = "net" if kind == "net" else "exchange"
                    flux.append(fl.FluxConstraint(**fields))
        return flux, pools

    # -- experiments -------------------------------------------------------

    def experiments(
        self,
    ) -> Tuple[List[fl.LabelingExperiment], List[fl.SimulationRun]]:
        """Build one experiment (and optionally one run) per configuration.

        Returns:
            The experiments and the simulation runs.
        """
        experiments = []
        runs = []
        global_keys = [
            (kind, f.expression, f.name)
            for kind, block in _constraint_blocks(self.data.constraints)
            for f in block
        ]
        for exp in self.data.experiments:
            exp_id = f"{self.dataset_id}.{exp.name}"
            experiments.append(self._experiment(exp, exp_id, global_keys))
            if exp.simulation is not None:
                runs.append(self._run(exp, exp_id))
        return experiments, runs

    def _experiment(
        self,
        exp: LabelingExperiments,
        exp_id: str,
        global_keys: List[Tuple[str, str, Optional[str]]],
    ) -> fl.LabelingExperiment:
        """Build one labeling experiment.

        An experiment without a tracer (FreeFlux keeps the tracer in its
        driver script, which is not parsed) is written without
        ``labeled_sources`` rather than with an invented one; the benchmark
        gate of ``fluxolink-validate`` reports it.

        Returns:
            The labeling experiment.
        """
        fields: Dict[str, Any] = {
            "id": exp_id,
            "network_model_id": self.network_id,
            "configuration_name": exp.name,
            "stationary": exp.stationary,
        }
        if exp.tracers:
            fields["labeled_sources"] = self._labeled_sources(exp, exp_id)
        else:
            warnings.warn(
                f"Experiment {exp.name!r} has no tracer input; it is written "
                "without labeled_sources (no tracer is invented)",
                stacklevel=3,
            )
        optional = {
            "investigation_id": self.investigation_id,
            "condition": self.experiment_conditions.get(exp.name),
            "data_origin": self.data_origin,
            "comment": exp.comment,
            "nominal_time": exp.time,
            "time_unit": self.time_units.get(exp.name),
        }
        fields.update({k: v for k, v in optional.items() if v is not None})
        if self.source_file_ids:
            fields["source_files"] = list(self.source_file_ids)
        flux, pools = self._constraints(exp.constraints, exp_id, global_keys)
        if flux:
            fields["flux_constraints"] = flux
        if pools:
            fields["pool_size_constraints"] = pools
        if exp.measurement is not None:
            fields.update(self._measurements(exp.measurement, exp_id))
        return fl.LabelingExperiment(**fields)

    def _labeled_sources(
        self, exp: LabelingExperiments, exp_id: str
    ) -> List[fl.LabeledSource]:
        """Build labeled sources from the experiment's tracers.

        Returns:
            One labeled source per tracer input.
        """
        sources = []
        seen: Counter[str] = Counter()
        for tracer in exp.tracers:
            seen[tracer.metabolite] += 1
            suffix = (
                ""
                if seen[tracer.metabolite] == 1
                else (f".{seen[tracer.metabolite]}")
            )
            fields: Dict[str, Any] = {
                "id": f"{exp_id}.input.{tracer.metabolite}{suffix}",
                "metabolite_id": tracer.metabolite,
            }
            if tracer.id:
                fields["name"] = tracer.id
            state_space = _STATE_SPACES.get((tracer.type or "").lower())
            if self.native_settings and state_space:
                fields["input_type"] = state_space
            if tracer.profile:
                fields["profile"] = tracer.profile
            isotopomers = []
            for label in tracer.labels:
                iso: Dict[str, Any] = {
                    "labeling_pattern": label.labeled_pattern
                }
                optional = {
                    "tracer_fraction": label.fraction,
                    "isotopic_purity": label.purity,
                    "cost": label.cost,
                    "profile_values": (
                        label.expression.textual if label.expression else None
                    ),
                }
                iso.update({k: v for k, v in optional.items() if v is not None})
                isotopomers.append(fl.TracerIsotopomer(**iso))
            if isotopomers:
                fields["tracers"] = isotopomers
            sources.append(fl.LabeledSource(**fields))
        return sources

    def _measurements(
        self, measurement: Measurement, exp_id: str
    ) -> Dict[str, Any]:
        """Build MID, NMR, flux and pool-size measurements.

        Returns:
            ``LabelingExperiment`` slot name to list of measurements.
        """
        model = measurement.model
        data = list(measurement.data.data)
        result: Dict[str, List[Any]] = {
            "mid_measurements": [],
            "nmr_measurements": [],
            "flux_measurements": [],
            "pool_size_measurements": [],
        }
        if model.labeling_measurement is not None:
            for group in model.labeling_measurement.groups:
                datums = [d for d in data if conv.datum_belongs(d.id, group.id)]
                if any(d.type for d in datums):
                    result["nmr_measurements"].append(
                        self._nmr(group, datums, exp_id)
                    )
                else:
                    result["mid_measurements"].append(
                        self._mid(group, datums, exp_id)
                    )
        reaction_ids = self.reaction_ids
        if model.flux_measurement is not None:
            for flux_type, items in (
                ("net", model.flux_measurement.net_fluxes),
                ("exchange", model.flux_measurement.xch_fluxes),
            ):
                for item in items:
                    expression = _textual(item.expression)
                    target = (
                        {"reaction_id": expression.strip()}
                        if expression.strip() in reaction_ids
                        else {"flux_expression": expression}
                    )
                    target["flux_type"] = flux_type
                    result["flux_measurements"].extend(
                        _scalar_measurements(
                            fl.FluxMeasurement, item, data, exp_id, target
                        )
                    )
        if model.metabolitesize_measurement is not None:
            metabolite_ids = self.data.model.metabolite_ids
            for item in model.metabolitesize_measurement.metabolite_sizes:
                expression = _textual(item.expression)
                target = (
                    {"metabolite_id": expression.strip()}
                    if expression.strip() in metabolite_ids
                    else {"pool_expression": expression}
                )
                result["pool_size_measurements"].extend(
                    _scalar_measurements(
                        fl.PoolSizeMeasurement, item, data, exp_id, target
                    )
                )
        return {k: v for k, v in result.items() if v}

    def _mid(
        self, group: Group, datums: List[Datum], exp_id: str
    ) -> fl.MIDMeasurement:
        """Build one MID measurement group.

        Returns:
            The MID measurement.
        """
        spec = _textual(group.expression)
        metabolite, positions = conv.parse_measurement_spec(
            spec, self.data, self.atom_counts
        )
        fields: Dict[str, Any] = {
            "id": f"{exp_id}.{group.id}",
            "metabolite_id": metabolite,
            "measurement_spec": spec,
        }
        if positions:
            fields["atom_positions"] = ",".join(str(p) for p in positions)
        times = _parse_times(group.times)
        if times:
            fields["time_points"] = times
        if self.native_settings:
            fields["scale_method"] = group.scale
        if group.errormodel is not None:
            fields["error_model_expression"] = _textual(
                group.errormodel.expression
            )
        points = []
        for datum in datums:
            index = conv.isotopologue_index(datum)
            if index is None:
                warnings.warn(
                    f"Datum {datum.id!r} of group {group.id!r} names no "
                    "isotopologue; MIDDataPoint needs one, datum dropped",
                    stacklevel=2,
                )
                continue
            point: Dict[str, Any] = {
                "isotopologue_index": index,
                "value": datum.value,
                "standard_deviation": datum.stddev,
            }
            if datum.weight is not None and not datum.weight.isdigit():
                point["weight"] = datum.weight
            if datum.time is not None:
                point["time_point"] = datum.time
            points.append(fl.MIDDataPoint(**point))
        if points:
            fields["data_points"] = points
        return fl.MIDMeasurement(**fields)

    def _nmr(
        self, group: Group, datums: List[Datum], exp_id: str
    ) -> fl.NMRIsotopomerMeasurement:
        """Build one 13C-NMR multiplet measurement group.

        Returns:
            The NMR measurement.
        """
        spec = _textual(group.expression)
        metabolite, _ = conv.parse_measurement_spec(
            spec, self.data, self.atom_counts
        )
        fields: Dict[str, Any] = {
            "id": f"{exp_id}.{group.id}",
            "metabolite_id": metabolite,
            "analytical_method": "NMR",
        }
        times = {d.time for d in datums if d.time is not None}
        if len(times) == 1:
            fields["time_point"] = times.pop()
        fields["data_points"] = [
            fl.NMRIsotopomerDataPoint(
                atom_position=d.pos,
                multiplet_type=d.type,
                value=d.value,
                standard_deviation=d.stddev,
            )
            for d in datums
            if d.type is not None and d.pos
        ]
        return fl.NMRIsotopomerMeasurement(**fields)

    # -- simulation runs ---------------------------------------------------

    def _run(self, exp: LabelingExperiments, exp_id: str) -> fl.SimulationRun:
        """Build the planned simulation run of one configuration.

        Returns:
            The simulation run.
        """
        simulation: Simulation = exp.simulation
        run_id = f"{exp_id}.run"
        fields: Dict[str, Any] = {
            "id": run_id,
            "network_model_id": self.network_id,
            "experiment_id": [exp_id],
            "configuration_name": exp.name,
            "fluxomics_method": "IST_MFA" if exp.stationary else "INST_MFA",
            "run_status": "planned",
            "software": self._software(f"{run_id}.software"),
        }
        if self.investigation_id:
            fields["investigation_id"] = self.investigation_id
        if self.fmt:
            fields["input_format"] = self.fmt
        options: List[fl.SolverOption] = []
        if self.fmt == conv.MTF and self.source_path is not None:
            prefix = self.source_path
            name = (prefix.with_suffix("") if prefix.suffix else prefix).name
            options.append(fl.SolverOption(key="--prefix", value=name))
        if self.native_settings:
            # Tool-specific knobs go to solver_options; FluxML's default
            # <simulation type="auto"> carries no information.
            if simulation.type in ("explicit", "full"):
                options.append(
                    fl.SolverOption(
                        key="simulation_type", value=simulation.type
                    )
                )
            state_space = _STATE_SPACES.get((simulation.method or "").lower())
            if state_space:
                fields["isotopic_state_space"] = state_space
        if options:
            fields["solver_options"] = options
        variables = self._variables(simulation)
        if variables:
            fields["variables"] = variables
        if self.source_file_ids:
            fields["input_files"] = list(self.source_file_ids)
        return fl.SimulationRun(**fields)

    def _software(self, software_id: str) -> fl.SoftwareInfo:
        """Build the software record of a run.

        Returns:
            The software information.
        """
        names = {v.value for v in fl.MFASoftwareEnum}
        fields: Dict[str, Any] = {"id": software_id}
        if self.software in names:
            fields["software_name"] = self.software
        else:
            fields["software_name"] = "other"
            fields["software_name_other"] = self.software or "unknown"
        if self.software_version:
            fields["version"] = self.software_version
        return fl.SoftwareInfo(**fields)

    def _variables(
        self, simulation: Simulation
    ) -> List[fl.VariableSpecification]:
        """Build variable specifications from simulation variables.

        FluxML ``<variables>`` lists the free fluxes and pool sizes (type
        ``net``/``xch``), so those rows are ``free``. MTF rows carry
        F/D/C codes. INCA and FreeFlux list every flux; their rows get no
        ``variable_type`` because the files do not classify them.

        Returns:
            The variable specifications.
        """
        if simulation.variables is None:
            return []
        rows = []
        flux_default = "free" if self.native_settings else None
        for value in simulation.variables.flux_values:
            if value.flux.split("___")[0] in self.skipped_reactions:
                continue
            code = value.type
            if code in _VARIABLE_CODES:
                kind, var_type = "net_flux", _VARIABLE_CODES[code]
            elif code in ("net", "xch"):
                kind = "net_flux" if code == "net" else "exchange_flux"
                var_type = flux_default
            else:
                warnings.warn(
                    f"Flux variable {value.flux!r} has unknown type {code!r};"
                    " dropped",
                    stacklevel=2,
                )
                continue
            rows.append(self._variable(value.flux, kind, var_type, value, code))
        for value in simulation.variables.metabolitesize_values:
            var_type = (
                _VARIABLE_CODES.get(value.type)
                if value.type
                else ("free" if self.native_settings else None)
            )
            rows.append(
                self._variable(
                    value.metabolite, "pool_size", var_type, value, value.type
                )
            )
        return rows

    def _variable(
        self,
        variable_id: str,
        kind: str,
        var_type: Optional[str],
        value: Any,
        code: Optional[str],
    ) -> fl.VariableSpecification:
        """Build one variable specification, dropping parser placeholders.

        Returns:
            The variable specification.
        """
        lower, upper = value.lo, value.hi
        if _is_placeholder_bound(self.fmt, kind, code, value):
            lower = upper = None
        fields: Dict[str, Any] = {
            "variable_id": variable_id,
            "variable_kind": kind,
        }
        optional = {
            "variable_type": var_type,
            "value": value.value,
            "lower_bound": lower,
            "upper_bound": upper,
            "increment": value.inc,
            "ed_weight": value.edweight if value.edweight != 1.0 else None,
        }
        fields.update({k: v for k, v in optional.items() if v is not None})
        return fl.VariableSpecification(**fields)


# -- helpers -----------------------------------------------------------------


def _iso_datetime(value: Optional[datetime]) -> Optional[str]:
    """Write a model date the way FluxML ``<info><date>`` does.

    ``IsoDateTime`` keeps the source's (absent) time zone; none is added.

    Returns:
        ``"YYYY-MM-DD HH:MM:SS"`` (with the offset when the value has one).
    """
    if value is None:
        return None
    text = value.strftime("%Y-%m-%d %H:%M:%S")
    offset = value.strftime("%z")
    return f"{text}{offset[:3]}:{offset[3:]}" if offset else text


def _unrepresentable_reactions(data: FluxomicsData) -> frozenset:
    """Find reactions ``NetworkReaction`` cannot hold.

    A reaction needs at least one reactant or product (a pure influx such as
    ``-> X`` is fine). The MTF parser drops participants written with empty
    atom parentheses (``R ()``), so e.g. ``V17: R () -> Q ()`` in the MTF toy
    model arrives with no participants at all. Such reactions are skipped
    with a warning, together with the flux variables that name them, instead
    of being given made-up participants.

    Returns:
        Ids of the reactions to skip.
    """
    skipped = frozenset(
        r.id for r in data.model.reactions if not r.reactants and not r.products
    )
    for rid in sorted(skipped):
        warnings.warn(
            f"Reaction {rid!r} has no participants; it and its flux "
            "variables are left out",
            stacklevel=3,
        )
    return skipped


def _participants(
    reaction: Reaction, side: str
) -> List[fl.ReactionParticipant]:
    """Build the participants of one side of a reaction.

    Each occurrence of a metabolite is its own participant with a 1-based
    ``instance``. The coefficient comes from ``stoichiometry_dict`` (INCA)
    divided over the occurrences on that side; a metabolite on both sides
    keeps coefficient 1 because the dict holds only the net value.

    Returns:
        The participants, in source order.
    """
    ids = getattr(reaction, side)
    other = reaction.products if side == "reactants" else reaction.reactants
    counts = Counter(ids)
    seen: Counter[str] = Counter()
    participants = []
    for metabolite in ids:
        seen[metabolite] += 1
        fields: Dict[str, Any] = {"metabolite_id": metabolite}
        if seen[metabolite] > 1:
            fields["instance"] = seen[metabolite]
        stoich = reaction.stoichiometry_dict
        if stoich and metabolite in stoich and metabolite not in other:
            coefficient = abs(stoich[metabolite]) / counts[metabolite]
            if abs(coefficient - 1.0) > 1e-12:
                fields["stoichiometry"] = coefficient
        participants.append(fl.ReactionParticipant(**fields))
    return participants


def _variants(
    transition: ConverterAtomTransition,
) -> List[fl.AtomTransitionVariant]:
    """Build atom-transition variants from a converter atom transition.

    The converter maps product atom -> reactant atom; the schema maps
    reactant atom -> product atom. Indices are 1-based on both sides.

    Returns:
        One variant per atom map.
    """
    weights = transition.weights or {}
    emit_ratio = (
        len(transition.maps) > 1
        and all(key in weights for key in transition.maps)
        and len({round(weights[key], 12) for key in transition.maps}) > 1
    )
    order = {
        met: i for i, met in reversed(list(enumerate(transition.products)))
    }
    variants = []
    for variant_id, atom_map in transition.maps.items():
        pairs = sorted(
            atom_map.mapping.items(),
            key=lambda item: (
                order.get(item[0].mol, len(order)),
                item[0].mol,
                item[0].instance,
                item[0].index,
            ),
        )
        transitions = []
        for product, source in pairs:
            fields: Dict[str, Any] = {
                "reactant_metabolite_id": source.mol,
                "reactant_atom_index": source.index,
                "product_metabolite_id": product.mol,
                "product_atom_index": product.index,
            }
            if source.instance != 1:
                fields["reactant_instance"] = source.instance
            if product.instance != 1:
                fields["product_instance"] = product.instance
            element = product.element or source.element
            if element:
                fields["element"] = element
            transitions.append(fl.AtomTransition(**fields))
        fields = {"variant_id": variant_id, "transitions": transitions}
        if emit_ratio:
            fields["ratio"] = weights[variant_id]
        variants.append(fl.AtomTransitionVariant(**fields))
    return variants


def _constraint_blocks(
    constraints: Optional[Constraints],
) -> List[Tuple[str, List[ConstraintFormula]]]:
    """List constraint formulas per kind (``net``, ``xch``, ``psize``).

    Returns:
        ``(kind, formulas)`` pairs.
    """
    if constraints is None:
        return []
    blocks = []
    for kind, block in (
        ("net", constraints.net),
        ("xch", constraints.xch),
        ("psize", constraints.metabolitesize),
    ):
        if block is not None and block.formulas:
            blocks.append((kind, list(block.formulas)))
    return blocks


def _operator(formula: ConstraintFormula) -> str:
    """Return the ``ConstraintOperatorEnum`` value of a formula.

    Returns:
        ``eq``, ``le``, ``ge``, ``lt`` or ``gt``.

    Raises:
        ValueError: If the formula has no relational operator.
    """
    if formula.is_mathml:
        operator = conv.mathml_relation(formula.expression)
    else:
        parts = conv.split_relation(formula.expression)
        operator = parts[1] if parts else None
    if operator is None:
        raise ValueError(
            f"Constraint {formula.expression!r} has no relational operator"
        )
    return _OPERATORS[operator]


def _textual(expression) -> str:
    """Return the textual (or else MathML) content of an expression.

    Returns:
        The expression text.
    """
    if expression is None:
        return ""
    return expression.textual if expression.textual else expression.mathml


def _parse_times(times: Optional[str]) -> List[float]:
    """Parse a FluxML ``times`` attribute (``"0.0,0.5,1."``).

    Returns:
        The time points, or an empty list when absent or unparseable.
    """
    if not times:
        return []
    try:
        return [float(t) for t in times.split(",") if t.strip()]
    except ValueError:
        warnings.warn(f"Cannot parse group times {times!r}", stacklevel=2)
        return []


def _scalar_measurements(
    cls: type,
    item: Any,
    data: List[Datum],
    exp_id: str,
    target: Dict[str, Any],
) -> List[Any]:
    """Build flux or pool-size measurements, one per datum.

    A measurement definition without a datum is dropped with a warning:
    ``value`` is required by the schema.

    Returns:
        The measurements.
    """
    datums = [d for d in data if d.id == item.id]
    if not datums:
        warnings.warn(
            f"Measurement {item.id!r} has no datum; FluxoLink requires a "
            "value, so it is dropped",
            stacklevel=3,
        )
    measurements = []
    for number, datum in enumerate(datums, start=1):
        fields: Dict[str, Any] = dict(target)
        fields["id"] = f"{exp_id}.{item.id}" + (
            f"#{number}" if number > 1 else ""
        )
        fields["value"] = datum.value
        fields["standard_deviation"] = datum.stddev
        if item.errormodel is not None:
            fields["error_model_expression"] = _textual(
                item.errormodel.expression
            )
        if datum.time is not None:
            fields["time_point"] = datum.time
        measurements.append(cls(**fields))
    return measurements


def _is_placeholder_bound(
    fmt: Optional[str], kind: str, code: Optional[str], value: Any
) -> bool:
    """Tell whether a variable's bounds are a parser placeholder.

    Returns:
        True for the bounds the MTF and FreeFlux parsers invent.
    """
    lo, hi = value.lo, value.hi
    if lo is None and hi is None:
        return False
    if fmt == conv.MTF:
        if code == "C":
            return lo == hi == value.value
        if code == "F":
            if kind == "pool_size":
                return (lo, hi) == (0.0, 1e6)
            return (lo, hi) == (-1000.0, 1000.0)
    if fmt in conv.FREEFLUX_FORMATS:
        return lo == hi == value.value
    return False
