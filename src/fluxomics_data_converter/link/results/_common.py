"""Helpers shared by the FluxoLink results readers.

Every reader returns a ``fluxolink`` :class:`SimulationRun`. The helpers
here keep the readers consistent on the points where tools differ:

* numbers: ``NA``, empty cells, NaN and +/-Inf become ``None`` (JSON and
  the schema have no representation for them);
* confidence intervals: a sampled interval (Monte Carlo, bootstrap) is kept
  even when it excludes its estimate, as quantiles of re-fits may do; an
  analytic interval that excludes its estimate is dropped with a warning
  (``fluxolink-validate`` rejects it, ``ci-order``); an interval with
  ``lower > upper`` is always dropped;
* residuals: ``residual`` is simulated minus measured and
  ``weighted_residual`` is that divided by the SD (the schema convention);
* derived values: whatever a reader computes instead of reading is stated
  in the record's ``derivation`` slot, quoting the tool's own number where
  it differs; the native output files keep every raw value.
"""

from __future__ import annotations

import math
import re
import warnings
from datetime import datetime
from typing import Any, Iterable, Optional, Tuple

from fluxolink.datamodel import fluxolink as fl

_NA_STRINGS = frozenset({"", "na", "nan", "null", "none", "n/a", "-"})
_MASS_RE = re.compile(r"^\s*[Mm]\+?(\d+)\s*$")


def to_float(value: Any) -> Optional[float]:
    """Return ``value`` as a finite float, or ``None``.

    Accepts strings (``"NA"`` and empty cells are missing), Python and numpy
    numbers, and one-element arrays.

    Args:
        value: The value to convert.

    Returns:
        The float, or ``None`` when missing, non-numeric or not finite.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        if value.strip().lower() in _NA_STRINGS:
            return None
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        size = getattr(value, "size", None)
        if size is not None and not isinstance(value, (int, float)):
            if size != 1:
                return None
            value = value.reshape(-1)[0] if hasattr(value, "reshape") else value
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
    return number if math.isfinite(number) else None


def to_int(value: Any) -> Optional[int]:
    """Return ``value`` as an int when it is a whole finite number.

    Args:
        value: The value to convert.

    Returns:
        The integer, or ``None``.
    """
    number = to_float(value)
    if number is None or number != int(number):
        return None
    return int(number)


def format_value(value: Any) -> Optional[str]:
    """Render a native setting as text for :class:`SolverOption.value`.

    Strings are kept verbatim, floats use the shortest round-trip form,
    infinities become ``Inf`` and arrays are space-separated.

    Args:
        value: The native value.

    Returns:
        The text, or ``None`` for an absent value.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        parts = [format_value(v) for v in value]
        return " ".join(p for p in parts if p is not None)
    if isinstance(value, float):
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        if math.isnan(value):
            return "NaN"
        return repr(value)
    return str(value)


def option(
    key: str, value: Any = None, description: Optional[str] = None
) -> fl.SolverOption:
    """Build one :class:`SolverOption`.

    Args:
        key: Native option name, verbatim.
        value: Option value (absent for a flag).
        description: Optional explanation.

    Returns:
        The solver option.
    """
    fields: dict[str, Any] = {"key": key}
    text = format_value(value)
    if text is not None:
        fields["value"] = text
    if description:
        fields["description"] = description
    return fl.SolverOption(**fields)


def iso_timestamp(value: Any) -> Optional[str]:
    """Render a timestamp in ISO 8601 form without inventing a time zone.

    Args:
        value: A :class:`datetime` or a ``"YYYY-MM-DD hh:mm:ss"`` string.

    Returns:
        ``"YYYY-MM-DDThh:mm:ss[.f][zone]"``, or ``None``.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value).strip()
    return text.replace(" ", "T", 1) if text else None


def seconds_between(
    start: Optional[str], end: Optional[str]
) -> Optional[float]:
    """Return the seconds from ``start`` to ``end`` (ISO strings).

    Args:
        start: Start timestamp.
        end: End timestamp.

    Returns:
        The non-negative duration, or ``None`` when it cannot be computed.
    """
    if not start or not end:
        return None
    try:
        delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
    except (TypeError, ValueError):
        return None
    seconds = delta.total_seconds()
    return seconds if seconds >= 0 else None


def software(
    run_id: str, name: str, version: Optional[str] = None
) -> fl.SoftwareInfo:
    """Build the :class:`SoftwareInfo` of a run.

    Args:
        run_id: Id of the run (the record gets ``<run_id>.software``).
        name: ``MFASoftwareEnum`` value.
        version: Software version, when known.

    Returns:
        The software record.
    """
    fields: dict[str, Any] = {"id": f"{run_id}.software", "software_name": name}
    if version:
        fields["version"] = str(version)
    return fl.SoftwareInfo(**fields)


#: Interval methods whose bounds are quantiles of re-fits; such an
#: interval may legitimately exclude the point estimate.
SAMPLED_CI_METHODS = frozenset({"monte_carlo", "bootstrap"})


def checked_interval(
    value: Optional[float],
    lower: Any,
    upper: Any,
    *,
    label: str,
    method: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Return a confidence interval the schema and validator accept.

    Args:
        value: The estimate.
        lower: Lower bound (any number-like, may be missing).
        upper: Upper bound.
        label: Name used in warnings.
        method: ``ConfidenceIntervalMethodEnum`` value. A sampled interval
            (:data:`SAMPLED_CI_METHODS`) is kept even when it excludes the
            estimate; any other interval is then dropped.

    Returns:
        ``(lower, upper)`` as floats, or ``(None, None)`` when both bounds
        are missing, when ``lower > upper``, or when an analytic interval
        excludes the estimate (a warning is issued in the last two cases).
    """
    lo, hi = to_float(lower), to_float(upper)
    if lo is None and hi is None:
        return None, None
    if lo is not None and hi is not None and lo > hi:
        warnings.warn(
            f"{label}: confidence interval [{lo}, {hi}] has lower > upper; "
            "interval dropped",
            stacklevel=3,
        )
        return None, None
    excludes = value is not None and (
        (lo is not None and lo > value) or (hi is not None and hi < value)
    )
    if excludes and method not in SAMPLED_CI_METHODS:
        warnings.warn(
            f"{label}: confidence interval [{lo}, {hi}] does not contain the "
            f"estimate {value}; interval dropped",
            stacklevel=3,
        )
        return None, None
    return lo, hi


def chi2_upper_p_value(
    wssr: Optional[float], dof: Optional[int]
) -> Optional[float]:
    """Return the conventional (upper-tail) chi-square p-value.

    Args:
        wssr: Weighted sum of squared residuals.
        dof: Degrees of freedom.

    Returns:
        ``P(X >= wssr)`` for ``X ~ chi2(dof)``, or ``None`` when undefined.
    """
    if wssr is None or dof is None or dof <= 0:
        return None
    from scipy.stats import chi2

    return float(chi2.sf(wssr, dof))


def chi2_two_sided_accepts(
    wssr: Optional[float], dof: Optional[int], confidence_level: float
) -> Optional[bool]:
    """Apply the two-sided chi-square acceptance test (INCA, FreeFlux).

    Args:
        wssr: Weighted sum of squared residuals.
        dof: Degrees of freedom.
        confidence_level: Confidence level of the test (e.g. 0.95).

    Returns:
        True when ``wssr`` lies in the central ``confidence_level``
        interval of ``chi2(dof)``, or ``None`` when undefined.
    """
    if wssr is None or dof is None or dof <= 0:
        return None
    from scipy.stats import chi2

    low = chi2.ppf((1 - confidence_level) / 2, dof)
    high = chi2.ppf((1 + confidence_level) / 2, dof)
    return bool(low <= wssr <= high)


def goodness_of_fit(**fields: Any) -> Optional[fl.GoodnessOfFit]:
    """Build a :class:`GoodnessOfFit`, dropping missing statistics.

    Args:
        **fields: GoodnessOfFit slots; ``None`` values are left out.

    Returns:
        The record, or ``None`` when no statistic is known.
    """
    kept = {k: v for k, v in fields.items() if v is not None}
    if not kept:
        return None
    return fl.GoodnessOfFit(**kept)


def measurement_fit(
    measurement_id: str,
    *,
    measured: Optional[float] = None,
    simulated: Optional[float] = None,
    sd: Optional[float] = None,
    weighted_residual: Optional[float] = None,
    isotopologue_index: Optional[int] = None,
    time_point: Optional[float] = None,
    p_value: Optional[float] = None,
    residual: Optional[float] = None,
    derivation: Optional[str] = None,
) -> fl.MeasurementFitResult:
    """Build one :class:`MeasurementFitResult` with consistent residuals.

    ``residual`` is the given value, else ``simulated - measured`` when
    both are known, else ``weighted_residual * sd``. ``weighted_residual``
    is the tool's value when given, otherwise ``residual / sd``.

    Args:
        measurement_id: Native measurement name or id.
        measured: Measured value.
        simulated: Simulated (fitted) value.
        sd: Standard deviation used for weighting.
        weighted_residual: The tool's ``(simulated - measured) / sd``.
        isotopologue_index: Mass isotopologue index, for MID values.
        time_point: Sampling time, for non-stationary data.
        p_value: The tool's p-value of the residual.
        residual: Unweighted residual, when it cannot be computed as
            ``simulated - measured``.
        derivation: Which values were derived rather than read, and how.

    Returns:
        The fit result.
    """
    if residual is None and measured is not None and simulated is not None:
        residual = simulated - measured
    elif residual is None and weighted_residual is not None and sd is not None:
        residual = weighted_residual * sd
    if weighted_residual is None and residual is not None and sd:
        weighted_residual = residual / sd
    fields = {
        "measurement_id": measurement_id,
        "isotopologue_index": isotopologue_index,
        "time_point": time_point,
        "measured_value": measured,
        "simulated_value": simulated,
        "standard_deviation": sd,
        "residual": residual,
        "weighted_residual": weighted_residual,
        "p_value": p_value,
        "derivation": derivation,
    }
    return fl.MeasurementFitResult(
        **{k: v for k, v in fields.items() if v is not None}
    )


def isotopologue_index(label: Any) -> Optional[int]:
    """Return the mass shift of an isotopologue label (``"M2"`` -> 2).

    Args:
        label: Label such as ``"M0"``, ``"m+3"`` or an integer.

    Returns:
        The index, or ``None`` when the label is not a mass isotopologue.
    """
    if isinstance(label, int) and not isinstance(label, bool):
        return label if label >= 0 else None
    match = _MASS_RE.match(str(label))
    return int(match.group(1)) if match else None


def compact(fields: dict[str, Any]) -> dict[str, Any]:
    """Drop ``None`` values and empty lists from a slot dictionary.

    Args:
        fields: Slot name to value.

    Returns:
        The non-empty slots.
    """
    return {k: v for k, v in fields.items() if v is not None and v != []}


def solution(
    run_id: str,
    role: str,
    *,
    flux_results: Iterable[fl.FluxResult] = (),
    pool_size_results: Iterable[fl.PoolSizeResult] = (),
    simulated_mids: Iterable[fl.SimulatedMID] = (),
    measurement_fits: Iterable[fl.MeasurementFitResult] = (),
    scale_factors: Iterable[fl.ScaleFactorResult] = (),
) -> fl.FluxSolution:
    """Build the :class:`FluxSolution` of a run (id ``<run_id>.solution``).

    Args:
        run_id: Id of the run.
        role: ``SolutionRoleEnum`` value.
        flux_results: Flux values.
        pool_size_results: Pool sizes.
        simulated_mids: Simulated MIDs.
        measurement_fits: Per-measurement fit results.
        scale_factors: Fitted scale factors.

    Returns:
        The flux solution.
    """
    return fl.FluxSolution(
        **compact(
            {
                "id": f"{run_id}.solution",
                "solution_role": role,
                "flux_results": list(flux_results),
                "pool_size_results": list(pool_size_results),
                "simulated_mids": list(simulated_mids),
                "measurement_fits": list(measurement_fits),
                "scale_factors": list(scale_factors),
            }
        )
    )


def wrap_in_dataset(
    run: fl.SimulationRun,
    *,
    dataset_id: str,
    investigation_id: Optional[str] = None,
    investigation_name: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> fl.FluxoDataset:
    """Place one run into a minimal catalog-tier :class:`FluxoDataset`.

    The dataset holds one :class:`Investigation` and the run; network,
    experiments and source files stay outside it, as the catalog tier
    allows. The run is copied, and its ``investigation_id`` is set to the
    dataset's investigation when it has none.

    Args:
        run: The simulation run.
        dataset_id: Id of the new dataset.
        investigation_id: Id of the investigation (default: the run's own
            ``investigation_id``, else ``<dataset_id>.investigation``).
        investigation_name: Name of the investigation.
        name: Name of the dataset.
        description: Description of the dataset.

    Returns:
        The dataset.
    """
    run = run.model_copy(deep=True)
    inv_id = run.investigation_id or investigation_id
    inv_id = inv_id or f"{dataset_id}.investigation"
    run.investigation_id = inv_id
    investigation = fl.Investigation(
        id=inv_id,
        name=investigation_name or f"{run.id} results",
    )
    fields = {
        "id": dataset_id,
        "name": name,
        "description": description,
        "schema_version": getattr(fl, "version", None),
        "dataset_type": "catalog",
        "investigations": [investigation],
        "simulation_runs": [run],
    }
    return fl.FluxoDataset(**compact(fields))
