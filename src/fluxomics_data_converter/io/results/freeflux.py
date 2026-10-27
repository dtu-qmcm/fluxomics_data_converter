"""Turn a FreeFlux fit result object into a FluxoLink run.

FreeFlux writes no result files, so this reader works on the object
``fitter.solve()`` returns (``FitResults``, or ``InstFitResults`` for
``fitter('inst')``). It is duck-typed and never imports :mod:`freeflux`::

    with model.fitter("ss") as fit:
        ...
        res = fit.solve(solver="slsqp")
        ci = res.estimate_confidence_intervals(which="net")
    run = from_freeflux_result(
        res, run_id="sm.freeflux", fitter_meta={"solver": "slsqp"}, ci=ci
    )

Attributes read
---------------
Full-precision fields are preferred over FreeFlux's rounded properties:
``_opt_net_fluxes`` and ``_opt_total_fluxes`` (``pandas.Series``; total ids
``<rxn>_f`` / ``<rxn>_b``), ``_opt_concs`` (INST pool sizes), ``_opt_obj``
(the WSSR), ``simulated_MDVs`` / ``measured_MDVs`` (``{emu: mdv}`` /
``{emu: [means, sds]}``; INST: ``simulated_inst_MDVs`` /
``measured_inst_MDVs`` keyed by time as well), ``simulated_fluxes`` /
``measured_fluxes``, ``n_meas``, ``n_params``, ``dof`` and ``is_success``.

Mapping decisions
-----------------
* Net fluxes are FreeFlux's own; forward and backward fluxes are its total
  fluxes; the exchange flux is derived as ``min(forward, backward)``
  (FreeFlux exposes none) and says so in ``derivation``.
* Simulated MIDs are FreeFlux's normalised model output (``scaled:
  false``; FreeFlux fits no scale factors).
* Residuals are recomputed from the simulated and measured values as
  ``simulated - measured`` and divided by the SD, which is how FreeFlux
  defines ``opt_resids``.
* Goodness of fit: ``wssr`` = ``_opt_obj``; the p-value is recomputed as
  the conventional upper tail ``P(chi2(dof) >= wssr)``; ``fit_accepted`` is
  FreeFlux's own two-sided ``chi2_test`` rule at its default 0.999 level
  (``chi2_confidence_level``).
* Confidence intervals: ``ci`` takes what
  ``estimate_confidence_intervals`` returned (``linearized`` by default,
  FreeFlux's non-standard ``t.ppf((1+cl)/2, dof)/sqrt(dof+1)*sd`` with lower
  bounds of irreversible reactions clipped at 0). ``mc_runs`` takes the
  ``FitMCResults`` of ``solve_with_confidence_intervals`` (its own
  ``estimate_confidence_intervals`` is used) or a list of net-flux Series
  (FreeFlux's algorithm is replicated: 3-sigma outlier filter, joint
  quantile mask, ``[min, max]``). Monte Carlo intervals are kept even when
  they exclude the estimate; a linearised one that does is dropped.

Not recoverable: solver status message, iteration count, runtime, flux
SDs (FreeFlux keeps only the Hessian, whose scaling convention it does not
document), and anything about the run setup that ``fitter_meta`` does not
supply.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from fluxolink.datamodel import fluxolink as fl

from ._common import (
    checked_interval,
    chi2_two_sided_accepts,
    chi2_upper_p_value,
    compact,
    goodness_of_fit,
    iso_timestamp,
    measurement_fit,
    option,
    software,
    solution,
    to_float,
    to_int,
)

__all__ = ["from_freeflux_result"]

#: ``fitter_meta`` keys that fill run slots instead of ``solver_options``.
_META_SLOTS = frozenset(
    {
        "version",
        "software_version",
        "runtime_seconds",
        "start_time",
        "end_time",
        "timepoints",
        "time_unit",
        "random_seed",
        "seed",
        "input_format",
        "command_line",
    }
)


def _first_attr(obj: Any, *names: str) -> Any:
    """Return the first attribute of ``obj`` that exists and is not None."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _as_dict(value: Any) -> Dict[str, Any]:
    """Return a Series or mapping as a plain ``{str: value}`` dict."""
    if value is None:
        return {}
    items = value.items() if hasattr(value, "items") else dict(value).items()
    return {str(k): v for k, v in items}


def _vector(value: Any) -> np.ndarray:
    """Return an MDV-like value as a flat float array."""
    if not isinstance(value, (np.ndarray, list, tuple)) and hasattr(
        value, "value"
    ):
        value = value.value
    return np.asarray(value, dtype=float).reshape(-1)


def _emu_parts(emu_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ``("Glu", "1,2,3,4,5")`` for the EMU id ``"Glu_12345"``."""
    metabolite, sep, atoms = emu_id.rpartition("_")
    if sep and atoms.isdigit():
        return metabolite, ",".join(atoms)
    return None, None


def _detect_version(res: Any, meta: Mapping) -> Optional[str]:
    """Return the FreeFlux version from ``fitter_meta`` or the installation."""
    version = meta.get("software_version") or meta.get("version")
    if version:
        return str(version)
    if type(res).__module__.split(".")[0] == "freeflux":
        try:
            from importlib.metadata import version as dist_version

            return dist_version("freeflux")
        except Exception:  # pragma: no cover - depends on the environment
            return None
    return None


def from_freeflux_result(
    res: Any,
    *,
    run_id: str,
    fitter_meta: Optional[Mapping[str, Any]] = None,
    ci: Optional[Mapping[str, Any]] = None,
    ci_method: str = "linearized",
    mc_runs: Any = None,
    confidence_level: float = 0.95,
    chi2_confidence_level: float = 0.999,
    software_version: Optional[str] = None,
    flux_unit: Optional[str] = None,
    pool_size_unit: Optional[str] = None,
    investigation_id: Optional[str] = None,
) -> fl.SimulationRun:
    """Build a :class:`SimulationRun` from a FreeFlux fit result.

    Args:
        res: ``FitResults`` or ``InstFitResults`` (or any object with the
            same attributes).
        run_id: Id of the run.
        fitter_meta: How the fit was run: ``solve()`` arguments (``solver``,
            ``tol``, ``max_iters``, ``ini_fluxes``, ...) become
            ``solver_options`` verbatim (``solver`` also fills
            ``optimization_method``, ``tol`` ``convergence_tolerance``);
            ``version``, ``runtime_seconds``, ``start_time``, ``end_time``,
            ``timepoints``, ``time_unit`` and ``seed`` fill their slots.
        ci: Intervals as returned by ``estimate_confidence_intervals``:
            ``{id: [lb, ub]}`` keyed by net flux, total flux
            (``<rxn>_f``/``<rxn>_b``) or concentration ids, or
            ``{"net": {...}, "total": {...}, "conc": {...}}``.
        ci_method: Method that produced ``ci``
            (``ConfidenceIntervalMethodEnum``).
        mc_runs: Monte Carlo results: a ``FitMCResults``-like object or a
            list of net-flux Series. Used for intervals when ``ci`` is not
            given; always sets ``monte_carlo_runs``.
        confidence_level: Confidence level of the intervals.
        chi2_confidence_level: Level of FreeFlux's chi-square acceptance
            test (its ``chi2_test`` default is 0.999).
        software_version: FreeFlux version (overrides ``fitter_meta``).
        flux_unit: Unit for every flux result.
        pool_size_unit: Unit for every pool size result.
        investigation_id: Investigation the run belongs to.

    Returns:
        The completed estimation run.

    Raises:
        ValueError: If ``res`` carries no optimal net fluxes.
    """
    meta = dict(fitter_meta or {})
    net = _as_dict(_first_attr(res, "_opt_net_fluxes", "opt_net_fluxes"))
    if not net:
        raise ValueError("the FreeFlux result carries no optimal net fluxes")
    total = _as_dict(_first_attr(res, "_opt_total_fluxes", "opt_total_fluxes"))
    concs = _as_dict(_first_attr(res, "_opt_concs", "opt_concentrations"))
    inst = bool(concs) or hasattr(res, "simulated_inst_MDVs")
    irreversible = {r for r in net if r in total}

    intervals, method, n_mc = _intervals(
        res, ci, ci_method, mc_runs, irreversible, confidence_level
    )
    fluxes = _flux_results(
        net, total, intervals, method, confidence_level, flux_unit
    )
    pools = []
    for name, raw in concs.items():
        value = to_float(raw)
        if value is None:
            continue
        pools.append(
            fl.PoolSizeResult(
                **compact(
                    {
                        "metabolite_id": name,
                        "value": value,
                        "pool_size_unit": pool_size_unit,
                        **_ci_slots(
                            value,
                            intervals.get(("conc", name)),
                            method,
                            confidence_level,
                            f"FreeFlux conc {name}",
                        ),
                    }
                )
            )
        )
    mids, fits = _mdv_results(res, inst)
    fits += _flux_fits(res)

    wssr = to_float(_first_attr(res, "_opt_obj", "opt_objective"))
    n_meas = to_int(getattr(res, "n_meas", None))
    n_params = to_int(getattr(res, "n_params", None))
    dof = to_int(getattr(res, "dof", None))
    if dof is None and n_meas is not None and n_params is not None:
        dof = n_meas - n_params
    gof = goodness_of_fit(
        wssr=wssr,
        number_of_measurements=n_meas,
        number_of_parameters=n_params,
        degrees_of_freedom=dof,
        chi_square_p_value=chi2_upper_p_value(wssr, dof),
        chi_square_confidence_level=chi2_confidence_level,
        fit_accepted=chi2_two_sided_accepts(wssr, dof, chi2_confidence_level),
        reduced_chi_square=wssr / dof if wssr is not None and dof else None,
    )

    modes = ["estimation"]
    if intervals:
        modes.append("confidence_interval_estimation")
    if mc_runs is not None:
        modes.append("monte_carlo")
    success = getattr(res, "is_success", None)
    timepoints = meta.get("timepoints")
    start_time = iso_timestamp(meta.get("start_time"))
    end_time = iso_timestamp(meta.get("end_time"))
    fields: Dict[str, Any] = {
        "id": run_id,
        "investigation_id": investigation_id,
        "fluxomics_method": "INST_MFA" if inst else "IST_MFA",
        "run_mode": modes,
        "run_status": "completed",
        "software": software(
            run_id, "freeflux", software_version or _detect_version(res, meta)
        ),
        "input_format": meta.get("input_format"),
        "command_line": meta.get("command_line"),
        "optimization_method": (
            None if meta.get("solver") is None else str(meta["solver"])
        ),
        "isotopic_state_space": "EMU",
        "monte_carlo_runs": n_mc,
        "random_seed": to_int(meta.get("seed", meta.get("random_seed"))),
        "convergence_tolerance": to_float(meta.get("tol")),
        "integration_timespan": (
            [float(t) for t in timepoints] if timepoints is not None else None
        ),
        "time_unit": meta.get("time_unit"),
        "solver_options": [
            option(str(k), v) for k, v in meta.items() if k not in _META_SLOTS
        ],
        "start_time": start_time,
        "end_time": end_time,
        "runtime_seconds": to_float(meta.get("runtime_seconds")),
        "converged": None if success is None else bool(success),
        "flux_solution": solution(
            run_id,
            "estimated",
            flux_results=fluxes,
            pool_size_results=pools,
            simulated_mids=mids,
            measurement_fits=fits,
        ),
        "goodness_of_fit": gof,
    }
    return fl.SimulationRun(**compact(fields))


# -- intervals ----------------------------------------------------------------


def _key_for(
    name: str, net: Iterable[str], total: Iterable[str], concs: Iterable[str]
) -> Optional[Tuple[str, str]]:
    """Return ``(kind, id)`` for an interval key (net / total / conc)."""
    if name in net:
        return ("net", name)
    if name in total:
        return ("total", name)
    if name in concs:
        return ("conc", name)
    return None


def _intervals(
    res: Any,
    ci: Optional[Mapping[str, Any]],
    ci_method: str,
    mc_runs: Any,
    irreversible: set,
    confidence_level: float,
) -> Tuple[Dict[Tuple[str, str], Any], Optional[str], Optional[int]]:
    """Return intervals keyed by ``(kind, id)``, their method, MC run count."""
    net = _as_dict(_first_attr(res, "_opt_net_fluxes", "opt_net_fluxes"))
    total = _as_dict(_first_attr(res, "_opt_total_fluxes", "opt_total_fluxes"))
    concs = _as_dict(_first_attr(res, "_opt_concs", "opt_concentrations"))
    n_mc = None
    sets: Dict[str, Any] = {}
    if mc_runs is not None:
        if hasattr(mc_runs, "net_fluxes_set"):
            n_mc = len(mc_runs.net_fluxes_set)
        else:
            n_mc = len(list(mc_runs))
    if ci is not None:
        method = ci_method
        if any(
            k in ("net", "total", "conc") and isinstance(v, Mapping)
            for k, v in ci.items()
        ):
            sets = {k: v for k, v in ci.items() if isinstance(v, Mapping)}
        else:
            sets = {"auto": ci}
    elif mc_runs is not None:
        method = "monte_carlo"
        sets = _mc_sets(mc_runs, irreversible, confidence_level)
    else:
        return {}, None, n_mc
    out: Dict[Tuple[str, str], Any] = {}
    for kind, ranges in sets.items():
        for name, bounds in _as_dict(ranges).items():
            key = (
                (kind, name)
                if kind != "auto"
                else _key_for(name, net, total, concs)
            )
            if key is None:
                warnings.warn(
                    f"FreeFlux interval for unknown id {name!r}", stacklevel=3
                )
                continue
            out[key] = bounds
    return out, method if out else None, n_mc


def _mc_sets(
    mc_runs: Any, irreversible: set, confidence_level: float
) -> Dict[str, Any]:
    """Return Monte Carlo intervals per kind (net / total / conc)."""
    sets: Dict[str, Any] = {}
    if hasattr(mc_runs, "estimate_confidence_intervals"):
        kinds = ["net", "total"] + (
            ["conc"] if getattr(mc_runs, "concs_set", None) else []
        )
        for kind in kinds:
            try:
                sets[kind] = mc_runs.estimate_confidence_intervals(
                    which=kind, confidence_level=confidence_level
                )
            except Exception as error:  # FreeFlux raises on too few runs
                warnings.warn(
                    f"FreeFlux MC intervals ({kind}): {error}", stacklevel=4
                )
        return sets
    samples = list(getattr(mc_runs, "net_fluxes_set", mc_runs))
    if samples:
        sets["net"] = _mc_ranges(samples, irreversible, confidence_level)
    return sets


def _mc_ranges(
    samples: List[Any], irreversible: set, confidence_level: float
) -> Dict[str, List[float]]:
    """Replicate FreeFlux's ``_confidence_intervals_mc``.

    Runs with any value more than 3 SD from the mean are dropped, then only
    runs inside the per-flux quantiles for every flux are kept, and the
    interval is their ``[min, max]``; lower bounds of irreversible
    reactions are clipped at 0. A constant column (a fixed flux) does not
    reject runs, where FreeFlux's ``zscore`` would yield NaN.

    Args:
        samples: One net-flux Series (or mapping) per Monte Carlo run.
        irreversible: Ids whose lower bound is clipped at 0.
        confidence_level: Confidence level of the intervals.

    Returns:
        Flux id -> ``[lower, upper]``; empty when too few runs remain.
    """
    frame = pd.DataFrame([_as_dict(s) for s in samples]).astype(float)
    spread = frame.std(ddof=0).replace(0.0, np.nan)
    zscores = ((frame - frame.mean()) / spread).abs().fillna(0.0)
    frame = frame[(zscores < 3).all(axis=1)]
    if frame.empty:
        return {}
    low_q, high_q = (1 - confidence_level) / 2, (1 + confidence_level) / 2
    quantiles = frame.quantile([low_q, high_q])
    inside = frame.apply(
        lambda col: col.between(
            quantiles.loc[low_q, col.name], quantiles.loc[high_q, col.name]
        )
    )
    selected = frame[inside.all(axis=1)]
    if selected.empty:
        warnings.warn(
            "too few FreeFlux Monte Carlo runs for intervals", stacklevel=4
        )
        return {}
    ranges = {}
    for name, col in selected.items():
        low = col.min()
        ranges[name] = [
            max(0.0, low) if name in irreversible else low,
            col.max(),
        ]
    return ranges


def _ci_slots(
    value: float,
    bounds: Any,
    method: Optional[str],
    confidence_level: float,
    label: str,
) -> Dict[str, Any]:
    """Return the CI slots of one result.

    Args:
        value: The estimate.
        bounds: ``[lb, ub]`` as FreeFlux returns it, or ``None``.
        method: ``ConfidenceIntervalMethodEnum`` value of the interval.
        confidence_level: Confidence level of the interval.
        label: Name used in warnings.

    Returns:
        The interval slots, empty when there is no usable interval.
    """
    if bounds is None or method is None:
        return {}
    pair = list(bounds)
    if len(pair) != 2:
        return {}
    lo, hi = checked_interval(
        value, pair[0], pair[1], label=label, method=method
    )
    if lo is None and hi is None:
        return {}
    return {
        "lower_ci": lo,
        "upper_ci": hi,
        "confidence_level": confidence_level,
        "confidence_interval_method": method,
    }


# -- fluxes and measurements --------------------------------------------------


def _flux_results(
    net: Dict[str, Any],
    total: Dict[str, Any],
    intervals: Dict[Tuple[str, str], Any],
    method: Optional[str],
    confidence_level: float,
    flux_unit: Optional[str],
) -> List[fl.FluxResult]:
    """Build net, forward, backward and derived exchange flux results.

    Args:
        net: Net flux id -> value.
        total: Total flux id (``<rxn>_f``/``<rxn>_b`` or ``<rxn>``) ->
            value.
        intervals: ``(kind, id)`` -> ``[lb, ub]``.
        method: Method of the intervals.
        confidence_level: Confidence level of the intervals.
        flux_unit: Unit for every result.

    Returns:
        The flux results.
    """
    out: List[fl.FluxResult] = []

    def add(
        rxn: str,
        flux_type: str,
        value: Optional[float],
        key: Optional[Tuple[str, str]],
        derivation: Optional[str] = None,
    ) -> None:
        if value is None:
            return
        slots = _ci_slots(
            value,
            intervals.get(key) if key else None,
            method,
            confidence_level,
            f"FreeFlux {flux_type} {rxn}",
        )
        out.append(
            fl.FluxResult(
                **compact(
                    {
                        "reaction_id": rxn,
                        "flux_type": flux_type,
                        "value": value,
                        "flux_unit": flux_unit,
                        "derivation": derivation,
                        **slots,
                    }
                )
            )
        )

    for rxn, raw in net.items():
        add(rxn, "net", to_float(raw), ("net", rxn))
        fwd = to_float(total.get(f"{rxn}_f"))
        bwd = to_float(total.get(f"{rxn}_b"))
        if fwd is None or bwd is None:
            continue
        add(rxn, "forward", fwd, ("total", f"{rxn}_f"))
        add(rxn, "backward", bwd, ("total", f"{rxn}_b"))
        add(rxn, "exchange", min(fwd, bwd), None, "xch = min(fwd, bwd)")
    return out


def _mdv_results(
    res: Any, inst: bool
) -> Tuple[List[fl.SimulatedMID], List[fl.MeasurementFitResult]]:
    """Return simulated MIDs and fits from the (INST) MDV dictionaries."""
    if inst:
        simulated = _as_dict(getattr(res, "simulated_inst_MDVs", None))
        measured = _as_dict(getattr(res, "measured_inst_MDVs", None))
    else:
        simulated = {
            emu: {None: mdv}
            for emu, mdv in _as_dict(
                getattr(res, "simulated_MDVs", None)
            ).items()
        }
        measured = {
            emu: {None: pair}
            for emu, pair in _as_dict(
                getattr(res, "measured_MDVs", None)
            ).items()
        }
    mids: List[fl.SimulatedMID] = []
    fits: List[fl.MeasurementFitResult] = []
    for emu, by_time in simulated.items():
        metabolite, atoms = _emu_parts(emu)
        points = []
        for time, mdv in dict(by_time).items():
            t = to_float(time)
            for index, value in enumerate(_vector(mdv)):
                if np.isfinite(value):
                    points.append(
                        fl.SimulatedMIDDataPoint(
                            **compact(
                                {
                                    "isotopologue_index": index,
                                    "time_point": t,
                                    "value": float(value),
                                }
                            )
                        )
                    )
            pair = dict(measured.get(emu, {})).get(time)
            if pair is None:
                continue
            means, sds = _vector(pair[0]), _vector(pair[1])
            sims = _vector(mdv)
            for index in range(min(len(means), len(sims))):
                fits.append(
                    measurement_fit(
                        emu,
                        measured=to_float(means[index]),
                        simulated=to_float(sims[index]),
                        sd=to_float(sds[index]) if index < len(sds) else None,
                        isotopologue_index=index,
                        time_point=t,
                    )
                )
        mids.append(
            fl.SimulatedMID(
                **compact(
                    {
                        "measurement_id": emu if emu in measured else None,
                        "metabolite_id": metabolite,
                        "fragment_name": emu,
                        "atom_positions": atoms,
                        "scaled": False,  # FreeFlux fits no scale factors
                        "data_points": points,
                    }
                )
            )
        )
    return mids, fits


def _flux_fits(res: Any) -> List[fl.MeasurementFitResult]:
    """Return the fits of the measured fluxes."""
    simulated = _as_dict(getattr(res, "simulated_fluxes", None))
    measured = _as_dict(getattr(res, "measured_fluxes", None))
    fits = []
    for name, pair in measured.items():
        values = list(pair) if not np.isscalar(pair) else [pair]
        fits.append(
            measurement_fit(
                name,
                measured=to_float(values[0]),
                simulated=to_float(simulated.get(name)),
                sd=to_float(values[1]) if len(values) > 1 else None,
            )
        )
    return fits
