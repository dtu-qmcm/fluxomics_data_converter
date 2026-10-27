"""Read INCA results (``.mat``) into a FluxoLink run.

Entry point::

    from fluxomics_data_converter.link.results import read_inca_results
    run = read_inca_results(Path("model.mat"), mc_path=Path("model_mc.mat"))

An INCA results file is a MATLAB v5 file with the fit ``f``, the model ``m``
and the simulation ``s``. It is read with :func:`scipy.io.loadmat`
(``struct_as_record=False, squeeze_me=True``); MATLAB class objects inside
``m`` (``mets``, ``rates``, ``states``) are not readable that way and are not
needed.

Mapping
-------
``f.par`` (one entry per parameter)
    ``type`` ``'Net flux'`` -> net, ``'Exch flux'`` -> exchange (INCA's
    exchange flux is absolute), ``'Fwd flux'``/``'Rev flux'`` -> forward /
    backward, ``'Pool'`` -> ``PoolSizeResult``, ``'Norm'`` (MS scale
    factor) -> ``ScaleFactorResult``. The reaction id is ``id`` without its
    ``" net"``/``" exch"`` suffix. ``std`` is the linearised SD; ``free``
    = 1 -> ``variable_type: free`` (``free`` = 0 is dependent *or* fixed,
    so it is left unset).
Confidence intervals
    ``lb``/``ub`` when non-empty (INCA ran ``continuate``:
    ``parameter_continuation`` at ``1 - alf``); otherwise the ``CI`` of the
    Monte Carlo companion file (``monte_carlo``), whose columns follow
    ``f.par``. Scale factors (``Norm``) get the same bounds. A Monte Carlo
    interval is kept even when it excludes the estimate; a continuation
    interval that does is dropped with a warning.
``f.mnt[].res[]``
    ``MeasurementFitResult`` per value: ``data`` measured, ``fit``
    simulated, ``std``, ``val`` = ``(fit - data) / std`` -> weighted
    residual, ``peak`` -> isotopologue index, ``time`` (non-stationary
    only). MS values also become ``SimulatedMID`` with ``scaled: true``
    (``fit`` includes the ``Norm`` factor; metabolite and atoms from
    ``m.expts.data_ms`` when readable).
``f.chi2``, ``f.dof``, ``f.Echi2``, ``f.alf``
    ``wssr``, ``degrees_of_freedom``; ``fit_accepted`` is INCA's own test
    ``Echi2[0] <= chi2 <= Echi2[1]`` at ``chi_square_confidence_level`` =
    ``1 - alf``. INCA reports no p-value; the upper-tail p-value is computed
    from ``chi2`` and ``dof``.
``m.options``
    Every field -> ``solver_options`` (verbatim); ``sim_ss`` decides
    IST/INST-MFA, ``fit_starts`` -> ``multistart_count``; for
    non-stationary runs ``int_tspan`` and ``sim_tunit`` fill
    ``integration_timespan`` and ``time_unit`` (``int_reltol`` stays a
    solver option).

Not recoverable: INCA version, optimizer iterations, runtime, convergence
flag, and the continuation profiles (``vals``/``chi2s``).
"""

from __future__ import annotations

import math
import re
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    to_int,
)

__all__ = ["read_inca_results"]

_FLUX_TYPES = {
    "net flux": "net",
    "exch flux": "exchange",
    "fwd flux": "forward",
    "forward flux": "forward",
    "rev flux": "backward",
    "reverse flux": "backward",
    "backward flux": "backward",
}
_ID_SUFFIXES = (" net", " exch", " fwd", " rev", " pool", " norm")
_KINDS = {
    "ms": "mid",
    "msms": "labeling_other",
    "nmr": "nmr",
    "flux": "flux",
    "pool": "pool_size",
}


def _loadmat(path: Path) -> Dict[str, Any]:
    """Return a MATLAB v5 file's variables, structs as attribute objects."""
    from scipy.io import loadmat

    return loadmat(str(path), struct_as_record=False, squeeze_me=True)


def _items(value: Any) -> List[Any]:
    """Return a MATLAB struct array (or a single struct) as a list."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value.reshape(-1))
    return [value]


def _get(struct: Any, name: str) -> Any:
    """Return a struct field, ``None`` when missing or an empty array."""
    value = getattr(struct, name, None)
    if isinstance(value, np.ndarray) and value.size == 0:
        return None
    return value


def _text(value: Any) -> Optional[str]:
    """Return a MATLAB char value as ``str`` (``None`` when empty)."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = "".join(str(v) for v in value.reshape(-1))
    text = str(value).strip()
    return text or None


def _base_id(par_id: str) -> str:
    """Return the id without INCA's suffix (``"R2 net"`` -> ``"R2"``)."""
    for suffix in _ID_SUFFIXES:
        if par_id.lower().endswith(suffix):
            return par_id[: -len(suffix)].strip()
    return par_id


def _mc_intervals(
    mc_path: Optional[Path], n_par: int
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Read the Monte Carlo ``CI`` (2 x p) and the number of samples.

    Args:
        mc_path: INCA Monte Carlo file, or ``None``.
        n_par: Number of ``f.par`` entries the columns must match.

    Returns:
        The ``CI`` matrix (``None`` when absent or mismatched) and the
        number of Monte Carlo samples.

    Raises:
        ValueError: If the file holds neither ``CI`` nor ``ci0``.
    """
    if mc_path is None:
        return None, None
    data = _loadmat(Path(mc_path))
    if "CI" in data:
        ci, samples = np.asarray(data["CI"], float), data.get("K")
        n_runs = None
        if samples is not None:
            n_runs = int(np.atleast_2d(samples).shape[0])
    elif "ci0" in data:  # interrupted run ("dump" file)
        ci, samples = np.asarray(data["ci0"], float), data.get("k")
        n_runs = None
        if samples is not None:
            arr = np.asarray(samples)
            n_runs = (
                int(arr.shape[0] * arr.shape[1])
                if arr.ndim == 3
                else (int(arr.shape[0]))
            )
    else:
        raise ValueError(f"{mc_path} holds no INCA Monte Carlo CI")
    ci = np.atleast_2d(ci)
    if ci.shape != (2, n_par):
        warnings.warn(
            f"{Path(mc_path).name}: CI shape {ci.shape} does not match the "
            f"{n_par} parameters of f.par; Monte Carlo intervals ignored",
            stacklevel=3,
        )
        return None, n_runs
    return ci, n_runs


def _ms_fragments(m: Any) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """Return ``(experiment, fragment id)`` -> ``(metabolite, atoms)``."""
    out: Dict[Tuple[str, str], Tuple[str, str]] = {}
    if m is None:
        return out
    for expt in _items(_get(m, "expts")):
        expt_id = _text(_get(expt, "id")) or ""
        try:
            fragments = _items(_get(expt, "data_ms"))
        except Exception:  # pragma: no cover - unreadable MATLAB object
            continue
        for frag in fragments:
            frag_id = _text(getattr(frag, "id", None))
            state = _text(getattr(frag, "state", None))
            atoms = getattr(frag, "atoms", None)
            if not frag_id or not state:
                continue
            if isinstance(atoms, str):  # INCA stores e.g. "1 2 3"
                positions = [int(a) for a in re.findall(r"\d+", atoms)]
            elif atoms is not None:
                positions = [to_int(a) for a in np.atleast_1d(atoms)]
            else:
                positions = []
            text = ",".join(str(p) for p in positions if p is not None)
            out[(expt_id, frag_id)] = (state, text)
    return out


def read_inca_results(
    mat_path: Path | str,
    *,
    mc_path: Optional[Path | str] = None,
    run_id: Optional[str] = None,
    software_version: Optional[str] = None,
    flux_unit: Optional[str] = None,
    pool_size_unit: Optional[str] = None,
    optimization_method: Optional[str] = None,
    investigation_id: Optional[str] = None,
) -> fl.SimulationRun:
    """Read an INCA results ``.mat`` file into a :class:`SimulationRun`.

    Args:
        mat_path: INCA results file holding ``f`` (and usually ``m``).
        mc_path: Companion Monte Carlo file (``K`` samples and ``CI``, or an
            interrupted run's ``k``/``ci0``).
        run_id: Id of the run (default: ``<file stem>.inca``).
        software_version: INCA version (the file does not record it).
        flux_unit: Unit for every flux result when ``f.par.unit`` is empty.
        pool_size_unit: Unit for every pool size when ``f.par.unit`` is
            empty.
        optimization_method: Optimizer to record (the file does not say).
        investigation_id: Investigation the run belongs to.

    Returns:
        The completed estimation run.

    Raises:
        ValueError: If the file has no fit structure ``f``.
    """
    mat_path = Path(mat_path)
    data = _loadmat(mat_path)
    if "f" not in data:
        raise ValueError(f"{mat_path} holds no INCA fit structure 'f'")
    f, m = data["f"], data.get("m")
    run_id = run_id or f"{mat_path.stem}.inca"
    opts = _get(m, "options") if m is not None else None
    stationary = to_int(_get(opts, "sim_ss")) != 0 if opts is not None else True

    pars = _items(_get(f, "par"))
    mc_ci, mc_runs = _mc_intervals(
        Path(mc_path) if mc_path else None, len(pars)
    )
    alf = to_float(_get(f, "alf"))
    res_times: Dict[str, Optional[float]] = {}
    fits, mids, n_meas = _measurements(f, m, stationary, res_times)

    fluxes: List[fl.FluxResult] = []
    pools: List[fl.PoolSizeResult] = []
    scales: List[fl.ScaleFactorResult] = []
    n_free, continuation = 0, False
    for column, par in enumerate(pars):
        par_type = (_text(_get(par, "type")) or "").lower()
        par_id = _text(_get(par, "id")) or f"par{column}"
        value = to_float(_get(par, "val"))
        free = to_int(_get(par, "free")) == 1
        n_free += int(free)
        if value is None:
            warnings.warn(
                f"INCA parameter {par_id!r} has no value; skipped", stacklevel=2
            )
            continue
        ci_fields, used_cont = _interval(par, par_id, value, alf, mc_ci, column)
        continuation |= used_cont
        if par_type == "norm":
            parts = par_id.split()
            res_id = parts[2] if len(parts) > 3 else None
            scales.append(
                fl.ScaleFactorResult(
                    **compact(
                        {
                            "measurement_id": par_id,
                            "value": value,
                            "standard_deviation": to_float(_get(par, "std")),
                            "time_point": res_times.get(res_id)
                            if res_id
                            else None,
                            "lower_ci": ci_fields.get("lower_ci"),
                            "upper_ci": ci_fields.get("upper_ci"),
                        }
                    )
                )
            )
            continue
        unit = _text(_get(par, "unit"))
        common = {
            "value": value,
            "standard_deviation": to_float(_get(par, "std")),
            "variable_type": "free" if free else None,
            **ci_fields,
        }
        if par_type == "pool":
            pools.append(
                fl.PoolSizeResult(
                    **compact(
                        {
                            "metabolite_id": _base_id(par_id),
                            "pool_size_unit": unit or pool_size_unit,
                            **common,
                        }
                    )
                )
            )
        elif par_type in _FLUX_TYPES:
            fluxes.append(
                fl.FluxResult(
                    **compact(
                        {
                            "reaction_id": _base_id(par_id),
                            "flux_type": _FLUX_TYPES[par_type],
                            "flux_unit": unit or flux_unit,
                            **common,
                        }
                    )
                )
            )
        else:
            warnings.warn(
                f"INCA parameter type {par_type!r} ({par_id}) is not mapped",
                stacklevel=2,
            )

    chi2 = to_float(_get(f, "chi2"))
    dof = to_int(_get(f, "dof"))
    echi2 = _get(f, "Echi2")
    accepted = None
    if chi2 is not None and echi2 is not None and np.size(echi2) == 2:
        low, high = (float(v) for v in np.asarray(echi2).reshape(-1))
        accepted = bool(low <= chi2 <= high)
    if dof is not None and n_meas - n_free != dof:
        warnings.warn(
            f"{mat_path.name}: {n_meas} measurements - {n_free} free "
            f"parameters != f.dof {dof}",
            stacklevel=2,
        )
    gof = goodness_of_fit(
        wssr=chi2,
        number_of_measurements=n_meas or None,
        number_of_parameters=n_free or None,
        degrees_of_freedom=dof,
        chi_square_p_value=chi2_upper_p_value(chi2, dof),
        chi_square_confidence_level=None if alf is None else 1 - alf,
        fit_accepted=accepted,
        reduced_chi_square=chi2 / dof if chi2 is not None and dof else None,
    )

    modes = ["estimation"]
    if continuation or mc_ci is not None:
        modes.append("confidence_interval_estimation")
    if mc_path is not None:
        modes.append("monte_carlo")
    fields: Dict[str, Any] = {
        "id": run_id,
        "investigation_id": investigation_id,
        "fluxomics_method": "IST_MFA" if stationary else "INST_MFA",
        "run_mode": modes,
        "run_status": "completed",
        "software": software(run_id, "INCA", software_version),
        "optimization_method": optimization_method,
        "monte_carlo_runs": mc_runs if mc_path is not None else None,
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
    if opts is not None:
        fields.update(_option_slots(opts, stationary))
    return fl.SimulationRun(**compact(fields))


def _interval(
    par: Any,
    par_id: str,
    value: float,
    alf: Optional[float],
    mc_ci: Optional[np.ndarray],
    column: int,
) -> Tuple[Dict[str, Any], bool]:
    """Return the CI slots of one parameter and whether continuation ran.

    Args:
        par: One ``f.par`` entry.
        par_id: Its id (for warnings).
        value: Its estimate.
        alf: ``f.alf``, used when the entry has no ``alf``.
        mc_ci: Monte Carlo ``CI`` matrix (2 x p), or ``None``.
        column: Column of this parameter in ``mc_ci``.

    Returns:
        The interval slots (empty when there is no usable interval) and
        whether the interval came from ``continuate``.
    """
    par_alf = to_float(_get(par, "alf"))
    level = 1 - (par_alf if par_alf is not None else (alf or 0.05))
    lb, ub = _get(par, "lb"), _get(par, "ub")
    if lb is not None or ub is not None:
        method, used = "parameter_continuation", True
        lo, hi = checked_interval(
            value, lb, ub, label=f"INCA {par_id}", method=method
        )
    elif mc_ci is not None:
        method, used = "monte_carlo", False
        lo, hi = checked_interval(
            value,
            mc_ci[0, column],
            mc_ci[1, column],
            label=f"INCA {par_id}",
            method=method,
        )
    else:
        return {}, False
    if lo is None and hi is None:
        return {}, used
    return (
        {
            "lower_ci": lo,
            "upper_ci": hi,
            "confidence_level": level,
            "confidence_interval_method": method,
        },
        used,
    )


def _measurements(
    f: Any, m: Any, stationary: bool, res_times: Dict[str, Optional[float]]
) -> Tuple[List[fl.MeasurementFitResult], List[fl.SimulatedMID], int]:
    """Read ``f.mnt[].res[]`` into fits and simulated MIDs.

    Args:
        f: The INCA fit structure.
        m: The INCA model structure (for fragment metabolites), or None.
        stationary: Whether time points are to be left out.
        res_times: Filled with residual id -> time point.

    Returns:
        The fits, the simulated MIDs and the number of measured values.
    """
    fragments = _ms_fragments(m)
    fits: List[fl.MeasurementFitResult] = []
    groups: "OrderedDict[str, dict]" = OrderedDict()
    n_meas = 0
    for mnt in _items(_get(f, "mnt")):
        mnt_id = _text(_get(mnt, "id")) or ""
        expt = _text(_get(mnt, "expt")) or ""
        for res in _items(_get(mnt, "res")):
            res_id = _text(_get(res, "id")) or mnt_id
            res_type = (
                _text(_get(res, "type")) or _text(_get(mnt, "type")) or ""
            ).lower()
            kind = _KINDS.get(res_type)
            raw_time = to_float(_get(res, "time"))
            time = None if stationary else raw_time
            res_times.setdefault(res_id, time)
            measured = to_float(_get(res, "data"))
            sd = to_float(_get(res, "std"))
            simulated = to_float(_get(res, "fit"))
            if measured is not None and sd is not None:
                n_meas += 1
            peak = _text(_get(res, "peak"))
            index = isotopologue_index(peak) if peak else None
            fits.append(
                measurement_fit(
                    res_id,
                    measured=measured,
                    simulated=simulated,
                    sd=sd,
                    weighted_residual=to_float(_get(res, "val")),
                    isotopologue_index=index,
                    time_point=time,
                )
            )
            if kind != "mid" or index is None or simulated is None:
                continue
            group = groups.setdefault(
                res_id, {"fragment": mnt_id, "expt": expt, "points": []}
            )
            group["points"].append(
                fl.SimulatedMIDDataPoint(
                    **compact(
                        {
                            "isotopologue_index": index,
                            "time_point": time,
                            "value": simulated,
                        }
                    )
                )
            )
    mids = []
    for res_id, group in groups.items():
        metabolite, atoms = fragments.get(
            (group["expt"], group["fragment"]), (None, None)
        )
        mids.append(
            fl.SimulatedMID(
                **compact(
                    {
                        "measurement_id": res_id,
                        "metabolite_id": metabolite,
                        "fragment_name": group["fragment"],
                        "atom_positions": atoms or None,
                        "scaled": True,  # res.fit includes the MS norm
                        "data_points": group["points"],
                    }
                )
            )
        )
    return fits, mids, n_meas


def _option_slots(opts: Any, stationary: bool) -> Dict[str, Any]:
    """Return run slots and verbatim ``solver_options`` from ``m.options``."""
    names = [n for n in getattr(opts, "_fieldnames", []) if n != "base"]
    options = [option(name, _get(opts, name)) for name in names]
    slots: Dict[str, Any] = {"solver_options": options}
    starts = to_int(_get(opts, "fit_starts"))
    if starts is not None and starts >= 1:
        slots["multistart_count"] = starts
    if not stationary:
        tspan = _get(opts, "int_tspan")
        if tspan is not None:
            grid = [to_float(t) for t in np.atleast_1d(tspan)]
            if all(t is not None for t in grid):
                slots["integration_timespan"] = grid
        unit = _text(_get(opts, "sim_tunit"))
        if unit:
            slots["time_unit"] = unit
    return {
        k: v
        for k, v in slots.items()
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    }
