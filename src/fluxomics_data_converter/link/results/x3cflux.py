"""Turn a 13CFLUX v3 (``x3cflux``) fit into a FluxoLink run.

x3cflux writes no result files: ``run_optimization`` returns
``(optimum, objective)`` and everything else is computed from the live
simulator. This reader is duck-typed and never imports :mod:`x3cflux` at
module import time::

    sim = x3cflux.create_simulator_from_fml("model.fml")
    x, obj = run_optimization(sim, start)
    cis = run_profile_likelihood_cis(sim, x, alpha=0.95)
    run = from_x3cflux_fit(sim, x, obj, run_id="m.x3cflux", cis=cis)

What is read
------------
``simulator.parameter_space``
    ``parameter_names`` (``<rxn>.n`` net, ``<rxn>.x`` exchange, bare names
    pool sizes), ``compute_parameters(optimum)`` for the values, the
    ``*_classification`` objects for free / constrained / dependent.
``simulator.measurement_names`` / ``measurement_data`` /
``measurement_standard_deviations`` / ``measurement_time_stamps``
    Labeling groups (one vector per group and time stamp) and flux / pool
    measurements; ``compute_scaled_measurements(optimum)`` gives the fitted
    values and ``compute_measurements(optimum)`` the unscaled ones.
``simulator.configurations[*].measurements``
    Measurement type, MS weights and the ``auto_scalable`` flag.

Mapping decisions
-----------------
* Exchange fluxes ``<rxn>.x`` are taken as absolute exchange fluxes (the
  FluxML convention x3cflux follows).
* Scale factors are never returned by x3cflux; each is recovered exactly
  as ``scaled / unscaled`` for every auto-scaled group and time stamp with
  more than one value (the cases x3cflux scales), with that ``derivation``.
  Simulated MIDs hold the fitted (scaled) values; ``scaled`` says whether a
  factor was applied to the group.
* Goodness of fit: ``wssr`` is the optimizer's objective (x3cflux's
  ``compute_loss``). x3cflux reports no degrees of freedom, so the reader
  counts ``n_measurements - n_free_parameters - n_auto_scaled_groups``
  (one per recovered scale factor) and records that in
  ``degrees_of_freedom_convention``. The p-value is the upper tail;
  x3cflux has no acceptance verdict, so ``fit_accepted`` is left unset.
* Intervals from ``run_profile_likelihood_cis`` become
  ``profile_likelihood`` intervals at ``confidence_level`` (x3cflux's
  ``alpha``). SDs come from ``free_covariance`` (the result of
  ``compute_free_parameter_covariance``) for free parameters only.

Not recoverable: ipopt status and iteration count (``run_optimization``
discards them), runtime, scale factor SDs.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from fluxolink.datamodel import fluxolink as fl

from ._common import (
    checked_interval,
    chi2_upper_p_value,
    compact,
    goodness_of_fit,
    isotopologue_index,
    measurement_fit,
    option,
    software,
    solution,
    to_float,
)

__all__ = ["from_x3cflux_fit"]

_DOF_CONVENTION = "n_measurements - n_free_parameters - n_auto_scaled_groups"
_SCALE_DERIVATION = "scaled / unscaled simulated measurement"
_LABELING_KINDS = {
    "MSMeasurement": "mid",
    "MIMSMeasurement": "labeling_other",
    "MSMSMeasurement": "labeling_other",
    "CumomerMeasurement": "labeling_other",
    "GenericMeasurement": "labeling_other",
    "HNMRMeasurement": "nmr",
    "CNMRMeasurement": "nmr",
}
_PARAMETER_KINDS = {
    "FluxMeasurement": "flux",
    "PoolSizeMeasurement": "pool_size",
}


def _index(entry: Any) -> Optional[int]:
    """Return the parameter index of a classification entry."""
    if isinstance(entry, (tuple, list)) and entry:
        entry = entry[0]
    try:
        return int(entry)
    except (TypeError, ValueError):
        return None


def _variable_types(space: Any) -> Dict[Tuple[str, str], str]:
    """Return ``(kind, name)`` -> free / constrained / dependent."""
    out: Dict[Tuple[str, str], str] = {}
    for kind, attr in (
        ("n", "net_flux_classification"),
        ("x", "exchange_flux_classification"),
        ("p", "pool_size_classification"),
    ):
        cls = getattr(space, attr, None)
        names = list(getattr(cls, "names", []) or [])
        if not names:
            continue
        for label, field in (
            ("dependent", "dependent"),
            ("constrained", "quasi_constraint"),
            ("constrained", "constraint"),
            ("free", "free"),
        ):
            for entry in getattr(cls, field, []) or []:
                idx = _index(entry)
                if idx is not None and 0 <= idx < len(names):
                    out[(kind, names[idx])] = label
    return out


def _split_name(name: str) -> Tuple[str, str]:
    """Return ``("n", "R1")`` for ``"R1.n"``; bare names are pools."""
    if name.endswith((".n", ".x")):
        return name[-1], name[:-2]
    return "p", name


def _pairs(
    cis: Any, ci_names: Optional[Sequence[str]], simulator: Any
) -> Dict[str, Tuple[Any, Any]]:
    """Normalise profile-likelihood intervals to ``{name: (lb, ub)}``.

    Args:
        cis: Mapping, or list in ``ci_names`` order, or None.
        ci_names: Parameter names of a list.
        simulator: The simulator (for x3cflux's default names).

    Returns:
        Parameter name -> ``(lb, ub)``.

    Raises:
        ValueError: If a list comes without usable names.
    """
    if cis is None:
        return {}
    if isinstance(cis, Mapping):
        return {str(k): tuple(v) for k, v in cis.items()}
    cis = list(cis)
    if ci_names is None:
        try:  # x3cflux's own default order for run_profile_likelihood_cis
            from x3cflux.utilities import get_non_constant_names

            ci_names = list(get_non_constant_names(simulator))
        except Exception as error:
            raise ValueError(
                "cis is a list; pass ci_names (the names given to "
                "run_profile_likelihood_cis) or a {name: (lb, ub)} mapping"
            ) from error
    if len(ci_names) != len(cis):
        raise ValueError(f"{len(cis)} intervals but {len(ci_names)} ci_names")
    return {str(n): tuple(b) for n, b in zip(ci_names, cis)}


def _detect_version(simulator: Any) -> Optional[str]:
    """Return the installed x3cflux version when the simulator is real."""
    if type(simulator).__module__.split(".")[0] != "x3cflux":
        return None
    try:
        from importlib.metadata import version

        return version("x3cflux")
    except Exception:  # pragma: no cover - depends on the environment
        return None


def from_x3cflux_fit(
    simulator: Any,
    optimum: Any,
    objective: float,
    *,
    run_id: str,
    cis: Any = None,
    ci_names: Optional[Sequence[str]] = None,
    confidence_level: float = 0.95,
    free_covariance: Any = None,
    multistart_objectives: Optional[Sequence[float]] = None,
    optimizer_options: Optional[Mapping[str, Any]] = None,
    optimization_method: str = "ipopt",
    auto_scaled: Optional[Sequence[bool]] = None,
    software_version: Optional[str] = None,
    flux_unit: Optional[str] = None,
    pool_size_unit: Optional[str] = None,
    runtime_seconds: Optional[float] = None,
    investigation_id: Optional[str] = None,
) -> fl.SimulationRun:
    """Build a :class:`SimulationRun` from an x3cflux fit.

    Args:
        simulator: The x3cflux simulator the fit used.
        optimum: Free-parameter vector at the optimum.
        objective: Objective value at the optimum (the WSSR).
        run_id: Id of the run.
        cis: ``run_profile_likelihood_cis`` result: a list of ``(lb, ub)``
            (with ``ci_names``) or a ``{name: (lb, ub)}`` mapping.
        ci_names: Parameter names of ``cis`` when it is a list (default:
            x3cflux's ``get_non_constant_names``, which needs x3cflux).
        confidence_level: ``alpha`` given to ``run_profile_likelihood_cis``.
        free_covariance: ``compute_free_parameter_covariance`` result
            ``(covariance, non_identifiable_indices)`` or a bare matrix, in
            ``free_parameter_names`` order.
        multistart_objectives: Objective values of all starts of
            ``run_multi_optimization`` (sets ``multistart_count``).
        optimizer_options: ipopt keyword arguments used
            (``solver_options``, verbatim).
        optimization_method: Optimizer (x3cflux's optimizers use ipopt).
        auto_scaled: Auto-scaling flag per labeling group (default: the
            configurations' ``auto_scalable``, else inferred from the
            scaled and unscaled measurements).
        software_version: x3cflux version (default: installed version).
        flux_unit: Unit for every flux result.
        pool_size_unit: Unit for every pool size result.
        runtime_seconds: Wall-clock runtime, if the caller timed it.
        investigation_id: Investigation the run belongs to.

    Returns:
        The completed estimation run.
    """
    space = simulator.parameter_space
    x = np.asarray(optimum, dtype=float).reshape(-1)
    names = [str(n) for n in space.parameter_names]
    values = np.asarray(space.compute_parameters(x), dtype=float).reshape(-1)
    free_names = [str(n) for n in getattr(space, "free_parameter_names", [])]
    types = _variable_types(space)
    intervals = _pairs(cis, ci_names, simulator)
    sds, nonid = _free_sds(free_covariance, free_names)

    fluxes: List[fl.FluxResult] = []
    pools: List[fl.PoolSizeResult] = []
    for name, raw in zip(names, values):
        value = to_float(raw)
        if value is None:
            continue
        kind, base = _split_name(name)
        slots = {
            "value": value,
            "variable_type": types.get((kind, base)),
            "standard_deviation": sds.get(name),
            "structural_identifiability": (
                ("no" if name in nonid else "yes")
                if name in free_names and free_covariance is not None
                else None
            ),
        }
        if name in intervals:
            lo, hi = checked_interval(
                value,
                *intervals[name],
                label=f"x3cflux {name}",
                method="profile_likelihood",
            )
            if lo is not None or hi is not None:
                slots.update(
                    lower_ci=lo,
                    upper_ci=hi,
                    confidence_level=confidence_level,
                    confidence_interval_method="profile_likelihood",
                )
        if kind == "p":
            pools.append(
                fl.PoolSizeResult(
                    **compact(
                        {
                            "metabolite_id": base,
                            "pool_size_unit": pool_size_unit,
                            **slots,
                        }
                    )
                )
            )
        else:
            fluxes.append(
                fl.FluxResult(
                    **compact(
                        {
                            "reaction_id": base,
                            "flux_type": "net" if kind == "n" else "exchange",
                            "flux_unit": flux_unit,
                            **slots,
                        }
                    )
                )
            )

    stationary = _stationary(simulator)
    mids, fits, scales, wssr_check = _measurements(
        simulator, x, stationary, auto_scaled
    )
    wssr = to_float(objective)
    if (
        wssr is not None
        and wssr_check is not None
        and not np.isclose(wssr, wssr_check, rtol=1e-6, atol=1e-9)
    ):
        warnings.warn(
            f"x3cflux objective {wssr} differs from the WSSR recomputed from "
            f"the measurements ({wssr_check})",
            stacklevel=2,
        )
    n_meas = sum(1 for f in fits if f.measured_value is not None)
    n_params = len(free_names) + len(scales)
    dof = n_meas - n_params
    gof = goodness_of_fit(
        wssr=wssr,
        number_of_measurements=n_meas,
        number_of_parameters=n_params,
        degrees_of_freedom=dof,
        degrees_of_freedom_convention=_DOF_CONVENTION,
        chi_square_p_value=chi2_upper_p_value(wssr, dof),
        reduced_chi_square=wssr / dof if wssr is not None and dof > 0 else None,
    )

    options = [option(str(k), v) for k, v in (optimizer_options or {}).items()]
    configs = list(getattr(simulator, "configurations", []) or [])
    config_names = [str(getattr(c, "name", "")) for c in configs]
    fields: Dict[str, Any] = {
        "id": run_id,
        "investigation_id": investigation_id,
        "configuration_name": config_names[0]
        if len(config_names) == 1 and config_names[0]
        else None,
        "fluxomics_method": "IST_MFA" if stationary else "INST_MFA",
        "run_mode": ["estimation"]
        + (["confidence_interval_estimation"] if intervals else []),
        "run_status": "completed",
        "software": software(
            run_id, "13CFLUX3", software_version or _detect_version(simulator)
        ),
        "optimization_method": optimization_method,
        "isotopic_state_space": _state_space(simulator),
        "multistart_count": len(multistart_objectives)
        if multistart_objectives is not None and len(multistart_objectives)
        else None,
        "solver_options": options,
        "runtime_seconds": to_float(runtime_seconds),
        "flux_solution": solution(
            run_id,
            "estimated",
            flux_results=fluxes,
            pool_size_results=pools,
            simulated_mids=mids,
            measurement_fits=fits,
            scale_factors=scales,
        ),
        "goodness_of_fit": gof,
    }
    return fl.SimulationRun(**compact(fields))


def _free_sds(
    free_covariance: Any, free_names: List[str]
) -> Tuple[Dict[str, float], set]:
    """Return SDs of free parameters and the non-identifiable ones."""
    if free_covariance is None:
        return {}, set()
    nonid_idx: Sequence[int] = []
    cov = free_covariance
    if isinstance(free_covariance, tuple) and len(free_covariance) == 2:
        cov, nonid_idx = free_covariance
    matrix = np.atleast_2d(np.asarray(cov, dtype=float))
    nonid = {
        free_names[int(i)]
        for i in np.atleast_1d(nonid_idx)
        if 0 <= int(i) < len(free_names)
    }
    sds: Dict[str, float] = {}
    for i, name in enumerate(free_names[: matrix.shape[0]]):
        var = matrix[i, i]
        if name not in nonid and np.isfinite(var) and var >= 0:
            sds[name] = float(np.sqrt(var))
    return sds, nonid


def _stationary(simulator: Any) -> bool:
    """Return whether the simulator is isotopically stationary."""
    name = type(simulator).__name__
    if "NonStationary" in name:
        return False
    if "Stationary" in name:
        return True
    configs = list(getattr(simulator, "configurations", []) or [])
    if configs and getattr(configs[0], "stationary", None) is not None:
        return bool(configs[0].stationary)
    return True


def _state_space(simulator: Any) -> Optional[str]:
    """Return the isotopic state space from the simulator class name."""
    name = type(simulator).__name__
    if "Cumomer" in name:
        return "cumomer"
    if "EMU" in name:
        return "EMU"
    return None


def _labeling_measurements(simulator: Any) -> Tuple[List[Any], List[Any]]:
    """Return the configurations' labeling and parameter measurements."""
    labeling, parameter = [], []
    for config in getattr(simulator, "configurations", []) or []:
        for meas in getattr(config, "measurements", []) or []:
            if type(meas).__name__ in _PARAMETER_KINDS:
                parameter.append(meas)
            else:
                labeling.append(meas)
    return labeling, parameter


def _measurements(
    simulator: Any,
    x: np.ndarray,
    stationary: bool,
    auto_scaled: Optional[Sequence[bool]],
) -> Tuple[
    List[fl.SimulatedMID],
    List[fl.MeasurementFitResult],
    List[fl.ScaleFactorResult],
    Optional[float],
]:
    """Build fits, simulated MIDs and recovered scale factors.

    Args:
        simulator: The x3cflux simulator.
        x: Free parameters at the optimum.
        stationary: Whether time points are to be left out.
        auto_scaled: Auto-scaling flag per labeling group, or None.

    Returns:
        The simulated MIDs, the fits, the scale factors and the WSSR
        recomputed from the fits (None when there is no fit).

    Raises:
        ValueError: If the data vectors do not match the groups.
    """
    lab_names, par_names = (list(v) for v in simulator.measurement_names)
    lab_data, par_data = simulator.measurement_data
    lab_sds, par_sds = simulator.measurement_standard_deviations
    raw_lab, raw_par = simulator.compute_measurements(x)
    if hasattr(simulator, "compute_scaled_measurements"):
        fit_lab, fit_par = simulator.compute_scaled_measurements(x)
    else:
        fit_lab, fit_par = raw_lab, raw_par
    stamps = list(getattr(simulator, "measurement_time_stamps", []) or [])
    lab_objs, par_objs = _labeling_measurements(simulator)

    counts = [
        max(1, len(stamps[g])) if g < len(stamps) else 1
        for g in range(len(lab_names))
    ]
    if sum(counts) != len(lab_data):
        counts = [1] * len(lab_names)
    if sum(counts) != len(lab_data):
        raise ValueError(
            f"{len(lab_data)} labeling data vectors for {len(lab_names)} "
            "measurement groups"
        )

    mids: List[fl.SimulatedMID] = []
    fits: List[fl.MeasurementFitResult] = []
    scales: List[fl.ScaleFactorResult] = []
    wssr = 0.0
    k = 0
    for g, group in enumerate(lab_names):
        meas = lab_objs[g] if g < len(lab_objs) else None
        kind = _LABELING_KINDS.get(type(meas).__name__) if meas else None
        weights = _weights(meas)
        flag = _auto_flag(auto_scaled, meas, g)
        points = []
        group_scaled = False
        for t_index in range(counts[g]):
            time = None
            if not stationary and g < len(stamps) and t_index < len(stamps[g]):
                time = to_float(stamps[g][t_index])
            data = np.asarray(lab_data[k], float).reshape(-1)
            sd = np.asarray(lab_sds[k], float).reshape(-1)
            raw = np.asarray(raw_lab[k], float).reshape(-1)
            fitted = np.asarray(fit_lab[k], float).reshape(-1)
            k += 1
            factor = _scale_factor(raw, fitted)
            scaled = (
                flag
                if flag is not None
                else (factor is not None and not np.isclose(factor, 1.0))
            )
            if scaled and len(raw) > 1 and factor is not None:
                group_scaled = True
                scales.append(
                    fl.ScaleFactorResult(
                        **compact(
                            {
                                "measurement_id": group,
                                "time_point": time,
                                "value": factor,
                                "derivation": _SCALE_DERIVATION,
                            }
                        )
                    )
                )
            for i in range(len(data)):
                index = None
                if kind == "mid":
                    index = weights[i] if weights and i < len(weights) else i
                fit = measurement_fit(
                    group,
                    measured=to_float(data[i]),
                    simulated=to_float(fitted[i]) if i < len(fitted) else None,
                    sd=to_float(sd[i]) if i < len(sd) else None,
                    isotopologue_index=index,
                    time_point=time,
                )
                fits.append(fit)
                if fit.weighted_residual is not None:
                    wssr += fit.weighted_residual**2
                if index is not None and fit.simulated_value is not None:
                    points.append(
                        fl.SimulatedMIDDataPoint(
                            **compact(
                                {
                                    "isotopologue_index": index,
                                    "time_point": time,
                                    "value": fit.simulated_value,
                                }
                            )
                        )
                    )
        if points:
            mids.append(
                fl.SimulatedMID(
                    **compact(
                        {
                            "measurement_id": group,
                            "metabolite_id": _metabolite(meas),
                            "fragment_name": group,
                            "scaled": group_scaled,
                            "data_points": points,
                        }
                    )
                )
            )
    for j, name in enumerate(par_names):
        meas = par_objs[j] if j < len(par_objs) else None
        fit = measurement_fit(
            name,
            measured=to_float(par_data[j]),
            simulated=to_float(fit_par[j]),
            sd=to_float(par_sds[j]),
        )
        fits.append(fit)
        if fit.weighted_residual is not None:
            wssr += fit.weighted_residual**2
    return mids, fits, scales, wssr if fits else None


def _weights(meas: Any) -> List[int]:
    """Return the mass weights of an MS measurement specification."""
    spec = getattr(meas, "specification", None)
    weights = getattr(spec, "weights", None)
    if weights is None:
        return []
    out = [isotopologue_index(w) for w in weights]
    return [w for w in out if w is not None]


def _metabolite(meas: Any) -> Optional[str]:
    """Return the metabolite name of a labeling measurement, if exposed."""
    name = getattr(meas, "metabolite_name", None)
    return str(name) if name else None


def _auto_flag(
    auto_scaled: Optional[Sequence[bool]], meas: Any, group: int
) -> Optional[bool]:
    """Return the auto-scaling flag of one labeling group, when known."""
    if auto_scaled is not None and group < len(auto_scaled):
        return bool(auto_scaled[group])
    flag = getattr(meas, "auto_scalable", None)
    return None if flag is None else bool(flag)


def _scale_factor(raw: np.ndarray, fitted: np.ndarray) -> Optional[float]:
    """Return ``s`` from ``fitted = s * raw`` (least squares, exact)."""
    if raw.shape != fitted.shape:
        return None
    denom = float(np.dot(raw, raw))
    if denom == 0.0:
        return None
    return float(np.dot(raw, fitted) / denom)
