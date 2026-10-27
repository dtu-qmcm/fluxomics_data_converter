"""Rebuild a converter model from a FluxoLink ``FluxoDataset``.

:func:`from_fluxo_dataset` is the inverse of
:func:`~.record_from_model.to_fluxo_dataset`, so the FluxML, MTF and
FreeFlux writers can be driven from a record::

    ds = load_record("ec.yaml")
    data = from_fluxo_dataset(ds)
    write_fluxml(data, "ec_roundtrip.fml")

Normalisation contract
----------------------
- Atom transitions: schema reactant -> product (with ``reactant_instance``
  and ``product_instance``) becomes the converter's product ->
  reactant :class:`~fluxomics_data_converter.AtomMap`; indices are 1-based
  on both sides. A transition without a product side (an atom leaving the
  system) has no converter representation and is skipped.
- Variants: ``variant_id`` keys the map; ``ratio`` becomes the weight.
  Without ratios a multi-variant reaction gets uniform weights, which is
  what every converter parser assigns.
- Participants: each participant is one entry in ``reactants`` /
  ``products``; ``stoichiometry_dict`` is set when a coefficient is not 1
  or a metabolite occurs more than once on a side.
- Measurements: ``measurement_spec`` is restored verbatim as the group
  expression and the native group / flux id is the measurement id with the
  ``<experiment id>.`` scope removed. Datum ids follow the native
  convention of ``SimulationRun.input_format`` (FluxML: the group id plus
  ``weight``; MTF: ``"<group>:M<i>"``; FreeFlux: ``"<group>:M<i>"`` plus
  ``pos``). Datums come out grouped by measurement, in measurement order.
- Variables: ``VariableSpecification`` rows become ``FluxValue`` /
  ``MetaboliteSizeValue``. MTF datasets get F/D/C codes and the MTF
  parser's placeholder bounds back; FluxML and INCA get ``net``/``xch``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from fluxolink.datamodel import fluxolink as fl

from ..core.common import Annotation, DictList, ErrorModel, TextualOrMath
from ..core.core import (
    FluxomicsData,
    LabelingExperiments,
    MetabolicNetworkModel,
    Metadata,
)
from ..experiment.measurement import (
    Datum,
    ExchangeFlux,
    FluxMeasurement,
    Group,
    LabelingMeasurement,
    Measurement,
    MeasurementData,
    MeasurementModel,
    MetaboliteSize,
    MetaboliteSizeMeasurement,
    NetFlux,
)
from ..experiment.tracer import LabelComposition, Tracers
from ..model.atom_mapping import AtomAddress, AtomMap, AtomTransition
from ..model.constraint import (
    ConstraintFormula,
    Constraints,
    ExchangeConstraints,
    MetaboliteSizeConstraints,
    NetConstraints,
)
from ..model.metabolite import Metabolite
from ..model.reaction import Reaction
from ..output.simulation import (
    FluxValue,
    MetaboliteSizeValue,
    Simulation,
    Variables,
)
from ..io import format_conventions as conv

_STATE_SPACES = {
    "auto": "auto",
    "isotopomer": "isotopomer",
    "cumomer": "cumomer",
    "EMU": "emu",
}
_CODES = {"free": "F", "dependent": "D", "constrained": "C"}


def from_fluxo_dataset(
    ds: fl.FluxoDataset, *, network_id: Optional[str] = None
) -> FluxomicsData:
    """Convert a FluxoLink dataset into a converter model.

    Args:
        ds: The dataset.
        network_id: Network to convert. Required (else ``ValueError``)
            when the dataset holds more than one network; experiments bound
            to other networks are left out.

    Returns:
        The converter model.
    """
    network = _select_network(ds, network_id)
    runs = list(ds.simulation_runs or [])
    fmt = _dataset_format(ds, runs)
    experiments = [
        exp
        for exp in ds.experiments or []
        if exp.network_model_id in (None, network.id)
    ]
    conditions = {c.id: c for c in ds.conditions or []}
    model = _model(network)
    return FluxomicsData(
        model=model,
        metadata=_metadata(network, experiments, conditions),
        constraints=_constraints(
            network.flux_constraints, network.pool_size_constraints
        ),
        experiments=[
            _experiment(exp, _run_for(exp, runs), fmt) for exp in experiments
        ],
    )


# -- network -----------------------------------------------------------------


def _select_network(
    ds: fl.FluxoDataset, network_id: Optional[str]
) -> fl.MetabolicNetworkModel:
    """Pick the network to convert.

    Returns:
        The network.

    Raises:
        ValueError: If no network matches or the choice is ambiguous.
    """
    networks = list(ds.networks or [])
    if network_id is not None:
        for network in networks:
            if network.id == network_id:
                return network
        raise ValueError(f"Dataset has no network {network_id!r}")
    if len(networks) != 1:
        raise ValueError(
            f"Dataset has {len(networks)} networks; pass network_id"
        )
    return networks[0]


def _dataset_format(
    ds: fl.FluxoDataset, runs: List[fl.SimulationRun]
) -> Optional[str]:
    """Return the native format the dataset was materialised from.

    Returns:
        The ``InputFormatEnum`` value of the runs or source files.
    """
    for run in runs:
        if run.input_format:
            return run.input_format
    formats = {f.format for f in ds.source_files or []}
    return formats.pop() if len(formats) == 1 else None


def _metadata(
    network: fl.MetabolicNetworkModel,
    experiments: List[fl.LabelingExperiment],
    conditions: Dict[str, fl.Condition],
) -> Optional[Metadata]:
    """Rebuild the converter metadata block.

    ``strain`` comes back from the single condition the experiments
    reference, when it names one.

    Returns:
        The metadata, or ``None`` when there is nothing to put in it.
    """
    strains = {
        conditions[exp.condition].strain
        for exp in experiments
        if exp.condition in conditions and conditions[exp.condition].strain
    }
    fields = {
        "name": network.name,
        "version": network.version,
        "date": network.model_date,
        "comment": network.description,
        "modeler": network.modeler,
        "strain": strains.pop() if len(strains) == 1 else None,
    }
    if all(value is None for value in fields.values()):
        return None
    return Metadata(**fields)


def _model(network: fl.MetabolicNetworkModel) -> MetabolicNetworkModel:
    """Rebuild the metabolic network model.

    Returns:
        The converter network model.
    """
    metabolites = DictList[Metabolite](
        [_metabolite(m) for m in network.metabolites or []]
    )
    reactions = DictList[Reaction]()
    atom_mappings: Dict[str, AtomTransition] = {}
    for reaction in network.reactions:
        converted, transition = _reaction(reaction)
        reactions.append(converted)
        if transition is not None:
            atom_mappings[reaction.id] = transition
    known = set(metabolites.ids)
    for reaction in reactions:
        for met in reaction.reactants + reaction.products:
            if met not in known:
                metabolites.append(Metabolite(id=met, atoms=None))
                known.add(met)
    return MetabolicNetworkModel(
        metabolites=metabolites,
        reactions=reactions,
        atom_mappings=atom_mappings,
        compartments=list(network.compartments or []),
    )


def _metabolite(metabolite: fl.NetworkMetabolite) -> Metabolite:
    """Rebuild one metabolite.

    ``name``, ``inchi`` and ``smiles`` become annotations, the form the
    FluxML parser produces and the FluxML writer emits.

    Returns:
        The converter metabolite.
    """
    annotations = [
        Annotation(name=key, content=value)
        for key, value in (
            ("name", metabolite.name),
            ("inchi", metabolite.inchi),
            ("smiles", metabolite.smiles),
        )
        if value
    ]
    return Metabolite(
        id=metabolite.id,
        atoms=metabolite.n_tracked_atoms,
        charge=metabolite.charge,
        formula=metabolite.formula,
        compartment=metabolite.compartment,
        annotations=annotations,
    )


def _reaction(
    reaction: fl.NetworkReaction,
) -> Tuple[Reaction, Optional[AtomTransition]]:
    """Rebuild one reaction and its atom transition.

    Returns:
        The converter reaction and its atom transition (or ``None``).
    """
    reactants = [p.metabolite_id for p in reaction.reactants]
    products = [p.metabolite_id for p in reaction.products or []]
    annotations = [
        Annotation(name=key, content=value)
        for key, value in (
            ("pathway", reaction.pathway),
            ("name", reaction.name),
        )
        if value
    ]
    variants = list(reaction.atom_transition_variants or [])
    transition = _atom_transition(reaction.id, reactants, products, variants)
    variant_ids = (
        [v.variant_id for v in variants]
        if transition is not None and len(variants) > 1
        else None
    )
    converted = Reaction(
        id=reaction.id,
        reversibility=reaction.reversibility,
        annotations=annotations,
        reactants=reactants,
        products=products,
        atom_transition_ids=variant_ids,
    )
    stoichiometry = _stoichiometry(reaction)
    if stoichiometry is not None:
        converted = converted.with_stoichiometry(stoichiometry)
    return converted, transition


def _stoichiometry(reaction: fl.NetworkReaction) -> Optional[Dict[str, float]]:
    """Build ``stoichiometry_dict`` when the default one would be wrong.

    The converter's default stoichiometric vector gives -1/+1 per
    metabolite regardless of how often it occurs, so an explicit dict is
    needed for non-unit coefficients and repeated metabolites.

    Returns:
        Metabolite id to signed net coefficient, or ``None``.
    """
    net: Dict[str, float] = defaultdict(float)
    occurrences: Counter[Tuple[str, str]] = Counter()
    explicit = False
    for side, sign in (("reactants", -1.0), ("products", 1.0)):
        for part in getattr(reaction, side) or []:
            coefficient = (
                part.stoichiometry if part.stoichiometry is not None else 1.0
            )
            explicit |= abs(coefficient - 1.0) > 1e-12
            occurrences[(side, part.metabolite_id)] += 1
            net[part.metabolite_id] += sign * coefficient
    repeated = any(count > 1 for count in occurrences.values())
    both = {m for s, m in occurrences if s == "reactants"} & {
        m for s, m in occurrences if s == "products"
    }
    if not (explicit or repeated or both):
        return None
    return dict(net)


def _atom_transition(
    reaction_id: str,
    reactants: List[str],
    products: List[str],
    variants: List[fl.AtomTransitionVariant],
) -> Optional[AtomTransition]:
    """Rebuild the converter atom transition of a reaction.

    Returns:
        The atom transition, or ``None`` when no variant has a mapping.
    """
    maps: Dict[str, AtomMap] = {}
    for variant in variants:
        mapping: Dict[AtomAddress, AtomAddress] = {}
        for t in variant.transitions or []:
            if t.product_metabolite_id is None or t.product_atom_index is None:
                continue
            mapping[
                AtomAddress(
                    t.product_metabolite_id,
                    t.product_atom_index,
                    t.element,
                    t.product_instance or 1,
                )
            ] = AtomAddress(
                t.reactant_metabolite_id,
                t.reactant_atom_index,
                t.element,
                t.reactant_instance or 1,
            )
        if mapping:
            maps[variant.variant_id] = AtomMap(mapping=mapping)
    if not maps:
        return None
    ratios = {v.variant_id: v.ratio for v in variants if v.variant_id in maps}
    if all(r is not None for r in ratios.values()):
        weights: Optional[Dict[str, float]] = dict(ratios)
    elif len(maps) > 1:
        weights = {key: 1.0 / len(maps) for key in maps}
    else:
        weights = None
    return AtomTransition(
        reaction_id=reaction_id,
        reactants=reactants,
        products=products,
        maps=maps,
        weights=weights,
    )


def _constraints(
    flux: Optional[List[fl.FluxConstraint]],
    pools: Optional[List[fl.PoolSizeConstraint]],
) -> Optional[Constraints]:
    """Rebuild converter constraints.

    Returns:
        The constraints, or ``None`` when there are none.
    """
    blocks: Dict[str, List[ConstraintFormula]] = defaultdict(list)
    for item in flux or []:
        kind = "xch" if item.flux_type == "exchange" else "net"
        blocks[kind].append(_formula(item))
    for item in pools or []:
        blocks["psize"].append(_formula(item))
    if not blocks:
        return None
    return Constraints(
        net=NetConstraints(formulas=blocks["net"]) if blocks["net"] else None,
        xch=(
            ExchangeConstraints(formulas=blocks["xch"])
            if blocks["xch"]
            else None
        ),
        metabolitesize=(
            MetaboliteSizeConstraints(formulas=blocks["psize"])
            if blocks["psize"]
            else None
        ),
    )


def _formula(item: Any) -> ConstraintFormula:
    """Rebuild one constraint formula.

    Returns:
        The converter constraint formula.

    Raises:
        ValueError: If the constraint has no expression (a bounds-only
            constraint, which no converter parser produces).
    """
    if item.expression:
        expression = item.expression
    elif getattr(item, "reaction_id", None) or getattr(
        item, "metabolite_id", None
    ):
        expression = _bounds_expression(item)
    else:
        raise ValueError(f"Constraint {item.id!r} has no expression")
    return ConstraintFormula(
        name=item.name,
        expression=expression,
        is_mathml=(item.expression or "").lstrip().startswith("<"),
    )


def _bounds_expression(item: Any) -> str:
    """Write a single-entity constraint as a textual expression.

    Returns:
        A textual constraint such as ``"R1 >= 0"`` or ``"R1 = 2"``.

    Raises:
        ValueError: If the constraint carries no value.
    """
    target = getattr(item, "reaction_id", None) or item.metabolite_id
    if item.fixed_value is not None:
        return f"{target} = {item.fixed_value}"
    parts = []
    if item.lower_bound is not None:
        parts.append(f"{target} >= {item.lower_bound}")
    if item.upper_bound is not None:
        parts.append(f"{target} <= {item.upper_bound}")
    if len(parts) != 1:
        raise ValueError(
            f"Constraint {item.id!r} needs exactly one of fixed_value, "
            "lower_bound or upper_bound to become one converter formula"
        )
    return parts[0]


# -- experiments ---------------------------------------------------------------


def _run_for(
    exp: fl.LabelingExperiment, runs: List[fl.SimulationRun]
) -> Optional[fl.SimulationRun]:
    """Find the single-experiment run that carries an experiment's settings.

    Returns:
        The run, or ``None``.
    """
    for run in runs:
        if list(run.experiment_id or []) == [exp.id]:
            return run
    return None


def _native_id(exp_id: str, measurement_id: str) -> str:
    """Strip the ``<experiment id>.`` scope from a measurement id.

    Returns:
        The native group / flux / pool-size id.
    """
    prefix = f"{exp_id}."
    if measurement_id.startswith(prefix):
        return measurement_id[len(prefix) :]
    return measurement_id


def _experiment(
    exp: fl.LabelingExperiment,
    run: Optional[fl.SimulationRun],
    dataset_fmt: Optional[str],
) -> LabelingExperiments:
    """Rebuild one experiment.

    Returns:
        The converter experiment.
    """
    fmt = (run.input_format if run else None) or dataset_fmt
    return LabelingExperiments(
        name=exp.configuration_name or exp.id,
        stationary=exp.stationary,
        time=exp.nominal_time,
        comment=exp.comment,
        tracers=[_tracer(src) for src in exp.labeled_sources or []],
        constraints=_constraints(
            exp.flux_constraints, exp.pool_size_constraints
        ),
        measurement=_measurement(exp, fmt),
        simulation=_simulation(run, fmt) if run else None,
    )


def _tracer(source: fl.LabeledSource) -> Tracers:
    """Rebuild one tracer input.

    Returns:
        The converter tracer.
    """
    labels = [
        LabelComposition(
            labeled_pattern=iso.labeling_pattern,
            purity=iso.isotopic_purity,
            cost=iso.cost if iso.cost else None,
            fraction=iso.tracer_fraction,
            expression=(
                TextualOrMath(textual=iso.profile_values)
                if iso.profile_values
                else None
            ),
        )
        for iso in source.tracers or []
    ]
    return Tracers(
        id=source.name,
        metabolite=source.metabolite_id,
        type=_STATE_SPACES.get(source.input_type or "", "isotopomer"),
        profile=source.profile,
        labels=labels,
    )


def _measurement(
    exp: fl.LabelingExperiment, fmt: Optional[str]
) -> Optional[Measurement]:
    """Rebuild the measurement model and data of an experiment.

    Returns:
        The converter measurement, or ``None`` when there is none.
    """
    groups: List[Group] = []
    data: List[Datum] = []
    for mid in exp.mid_measurements or []:
        group_id = _native_id(exp.id, mid.id)
        groups.append(
            Group(
                id=group_id,
                times=(
                    ",".join(repr(float(t)) for t in mid.time_points)
                    if mid.time_points
                    else None
                ),
                scale=mid.scale_method or "auto",
                errormodel=_error_model(mid.error_model_expression),
                expression=_expression(mid.measurement_spec or group_id),
            )
        )
        for point in mid.data_points or []:
            if point.value is None:
                continue
            data.append(_mid_datum(group_id, point, fmt))
    for nmr in exp.nmr_measurements or []:
        group_id = _native_id(exp.id, nmr.id)
        spec = ",".join(
            f"{p.multiplet_type}{p.atom_position}"
            for p in nmr.data_points or []
        )
        groups.append(
            Group(
                id=group_id,
                expression=TextualOrMath(
                    textual=f"{nmr.metabolite_id}#{spec}"
                    if spec
                    else nmr.metabolite_id
                ),
            )
        )
        for point in nmr.data_points or []:
            data.append(
                Datum(
                    id=group_id,
                    value=point.value,
                    stddev=point.standard_deviation,
                    time=nmr.time_point,
                    pos=point.atom_position,
                    type=point.multiplet_type,
                )
            )
    net_fluxes, xch_fluxes = _flux_measurements(exp, data)
    pool_sizes = _pool_measurements(exp, data)
    if not (groups or net_fluxes or xch_fluxes or pool_sizes):
        return None
    return Measurement(
        model=MeasurementModel(
            labeling_measurement=(
                LabelingMeasurement(groups=groups) if groups else None
            ),
            flux_measurement=(
                FluxMeasurement(net_fluxes=net_fluxes, xch_fluxes=xch_fluxes)
                if net_fluxes or xch_fluxes
                else None
            ),
            metabolitesize_measurement=(
                MetaboliteSizeMeasurement(metabolite_sizes=pool_sizes)
                if pool_sizes
                else None
            ),
        ),
        data=MeasurementData(data=data),
    )


def _mid_datum(
    group_id: str, point: fl.MIDDataPoint, fmt: Optional[str]
) -> Datum:
    """Rebuild one MID datum in the native id convention.

    Returns:
        The converter datum.
    """
    index = point.isotopologue_index
    fields: Dict[str, Any] = {
        "value": point.value,
        "stddev": point.standard_deviation or 0.0,
        "time": point.time_point,
    }
    if fmt == conv.MTF:
        fields.update(id=f"{group_id}:M{index}", weight=point.weight)
    elif fmt in conv.FREEFLUX_FORMATS:
        fields.update(id=f"{group_id}:M{index}", pos=index, weight=point.weight)
    else:
        fields.update(id=group_id, weight=point.weight or str(index), pos=index)
    return Datum(**fields)


def _flux_measurements(
    exp: fl.LabelingExperiment, data: List[Datum]
) -> Tuple[List[NetFlux], List[ExchangeFlux]]:
    """Rebuild flux measurements and append their datums.

    Returns:
        The net and exchange flux measurement definitions.
    """
    net: Dict[str, NetFlux] = {}
    xch: Dict[str, ExchangeFlux] = {}
    for item in exp.flux_measurements or []:
        native = _native_id(exp.id, item.id).split("#")[0]
        expression = item.reaction_id or item.flux_expression or native
        target, cls = (
            (xch, ExchangeFlux)
            if item.flux_type == "exchange"
            else (net, NetFlux)
        )
        if native not in target:
            target[native] = cls(
                id=native,
                errormodel=_error_model(item.error_model_expression),
                expression=_expression(expression),
            )
        data.append(
            Datum(
                id=native,
                value=item.value,
                stddev=item.standard_deviation or 0.0,
                time=item.time_point,
            )
        )
    return list(net.values()), list(xch.values())


def _pool_measurements(
    exp: fl.LabelingExperiment, data: List[Datum]
) -> List[MetaboliteSize]:
    """Rebuild pool-size measurements and append their datums.

    Returns:
        The pool-size measurement definitions.
    """
    sizes: Dict[str, MetaboliteSize] = {}
    for item in exp.pool_size_measurements or []:
        native = _native_id(exp.id, item.id).split("#")[0]
        expression = item.metabolite_id or item.pool_expression or native
        if native not in sizes:
            sizes[native] = MetaboliteSize(
                id=native,
                errormodel=_error_model(item.error_model_expression),
                expression=_expression(expression),
            )
        data.append(
            Datum(
                id=native,
                value=item.value,
                stddev=item.standard_deviation or 0.0,
                time=item.time_point,
            )
        )
    return list(sizes.values())


def _expression(text: str) -> TextualOrMath:
    """Wrap a stored expression, telling MathML from text.

    Returns:
        The converter expression.
    """
    if text.lstrip().startswith("<"):
        return TextualOrMath(mathml=text)
    return TextualOrMath(textual=text)


def _error_model(text: Optional[str]) -> Optional[ErrorModel]:
    """Wrap an error-model expression.

    Returns:
        The converter error model, or ``None``.
    """
    return ErrorModel(expression=_expression(text)) if text else None


# -- simulation ----------------------------------------------------------------


def _option(run: fl.SimulationRun, key: str) -> Optional[str]:
    """Return the value of one of the run's solver options.

    Returns:
        The option's value, or ``None`` when the run does not set it.
    """
    for option in run.solver_options or []:
        if option.key == key:
            return option.value
    return None


def _simulation(run: fl.SimulationRun, fmt: Optional[str]) -> Simulation:
    """Rebuild the simulation settings of a configuration.

    Returns:
        The converter simulation block.
    """
    rows = list(run.variables or [])
    codes = fmt == conv.MTF and not any(
        r.variable_kind == "exchange_flux" for r in rows
    )
    flux_values = []
    pool_values = []
    for row in rows:
        if row.variable_kind == "pool_size":
            pool_values.append(_pool_value(row, fmt))
        else:
            flux_values.append(_flux_value(row, fmt, codes))
    return Simulation(
        type=_option(run, "simulation_type") or "auto",
        method=_STATE_SPACES.get(run.isotopic_state_space or "", "auto"),
        variables=(
            Variables(
                flux_values=flux_values, metabolitesize_values=pool_values
            )
            if rows
            else None
        ),
    )


def _flux_value(
    row: fl.VariableSpecification, fmt: Optional[str], codes: bool
) -> FluxValue:
    """Rebuild one flux variable.

    Returns:
        The converter flux value.
    """
    lo, hi = row.lower_bound, row.upper_bound
    if codes and row.variable_type in _CODES:
        flux_type = _CODES[row.variable_type]
        if lo is None and hi is None:
            if flux_type == "F" and row.value is not None:
                lo, hi = -1000.0, 1000.0
            elif flux_type == "C":
                lo = hi = row.value
    else:
        flux_type = "xch" if row.variable_kind == "exchange_flux" else "net"
        if fmt in conv.FREEFLUX_FORMATS and lo is None and hi is None:
            lo = hi = row.value
    return FluxValue(
        flux=row.variable_id,
        type=flux_type,
        value=row.value,
        lo=lo,
        hi=hi,
        inc=row.increment,
        edweight=row.ed_weight if row.ed_weight is not None else 1.0,
    )


def _pool_value(
    row: fl.VariableSpecification, fmt: Optional[str]
) -> MetaboliteSizeValue:
    """Rebuild one pool-size variable.

    Returns:
        The converter pool-size value.
    """
    lo, hi = row.lower_bound, row.upper_bound
    code = _CODES.get(row.variable_type or "")
    if fmt in (None, conv.FLUXML) or fmt in conv.FREEFLUX_FORMATS:
        code = None
    if fmt == conv.MTF and lo is None and hi is None:
        if code == "F" and row.value is not None:
            lo, hi = 0.0, 1e6
        elif code == "C":
            lo = hi = row.value
    if fmt in conv.FREEFLUX_FORMATS and lo is None and hi is None:
        lo = hi = row.value
    return MetaboliteSizeValue(
        metabolite=row.variable_id,
        value=row.value,
        lo=lo,
        hi=hi,
        inc=row.increment,
        type=code,
        edweight=row.ed_weight if row.ed_weight is not None else 1.0,
    )
