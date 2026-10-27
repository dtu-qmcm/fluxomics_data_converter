"""Read influx_si results (``<prefix>_res/``) into a FluxoLink run.

Entry point::

    from fluxomics_data_converter.io.results import (
        read_influx_si_results,
    )
    run = read_influx_si_results(Path("simple_model/MTF/simple_model_res"))

Files read (each only when present)
-----------------------------------
``<prefix>.tvar.sim``
    Fluxes and pools: ``FluxResult`` / ``PoolSizeResult`` with value, SD,
    ``variable_type`` (Type F/D/C) and ``structural_identifiability``. With
    ``--sens mc=N`` the ``Low_mc``/``Up_mc`` columns (influx_si's hard-coded
    2.5 % / 97.5 % Monte Carlo quantiles) become a 0.95 ``monte_carlo``
    interval.
``<experiment>.miso.sim``, ``<prefix>.mflux.sim``, ``<prefix>.mmet.sim``
    ``SimulatedMID`` (MS rows) and ``MeasurementFitResult`` for every row.
``<prefix>.stat``
    ``GoodnessOfFit`` (p-value converted to the upper tail).
``tmp/<prefix>.pres.csv``
    ``ScaleFactorResult`` and the number of starts.
``<prefix>.log`` / ``<prefix>.err``
    ``command_line``, ``solver_options``, ``run_mode``, timestamps and
    warnings.
``tmp/<prefix>.R``, ``tmp/<prefix>.ftbl``
    Effective options merged from the ``.opt`` file, ``emu`` and the
    influx_si version.
``tmp/<prefix>_res.kvh`` (``--wkvh`` only)
    Iteration count, exit message and version.

Mapping decisions
-----------------
* **XCH.** influx_si writes exchange fluxes as ``xch01 = xch / (1 + xch)``
  in ``[0, 1)``. ``FluxTypeEnum.exchange`` is the absolute exchange flux
  ``min(fwd, rev)``, so ``value`` is ``xch01 / (1 - xch01)``, the SD comes
  from the delta method (``SD / (1 - xch01)**2``) and Monte Carlo bounds are
  transformed exactly (the map is monotone). ``derivation`` states the
  conversion and quotes the row as written (xch01 and its SD). A row with
  ``xch01 >= 1`` has no finite exchange flux and is skipped with a warning.
* **Monte Carlo intervals** are kept even when they exclude the estimate
  (quantiles of re-fits may); ``fluxolink-validate`` only warns.
* **Residuals.** influx_si's ``Residual`` column is
  ``(simulated - measured) / SD`` and goes to ``weighted_residual``;
  ``residual`` is ``simulated - measured``. The measured value comes from
  the input file (``.miso``/``.mflux``/``.mmet`` in ``input_dir``) and,
  when the input is unavailable, is recovered as
  ``simulated - Residual * SD`` (stated in ``derivation``). ``Pvalue``
  (influx_si's robust two-sided z-test) goes to ``p_value``.
* **Simulated MIDs.** Stationary ``.miso.sim`` values include the fitted
  scale factor (``scaled: true``); for non-stationary runs influx_si writes
  the unscaled, normalised MID (``scaled: false``), so ``residual`` is then
  taken as ``Residual * SD``.
* **Chi-square.** ``.stat`` reports the *lower* tail ``pchisq(chi2, df)``;
  the schema stores the upper tail, so ``chi_square_p_value`` is ``1 - p``.
  ``fit_accepted`` is influx_si's own conclusion.
* **Provenance.** The command line comes from the first line of the log,
  not from the ``.opt`` file, which the command line can override. Every
  command-line flag, ``--prefix`` included, is a ``SolverOption``; options
  that took effect only through the ``.opt`` file
  are added with a description saying so. Log timestamps are written in
  ISO form without a time zone (the log has none).
* **Run mode.** ``--noopt`` is a forward simulation whose flux values are
  the imposed starting values (``solution_role: reference``); otherwise an
  estimation. ``--sens mc[=N]`` adds ``confidence_interval_estimation``
  and ``monte_carlo``.
"""

from __future__ import annotations

import re
import shlex
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from fluxolink.datamodel import fluxolink as fl

from ._common import (
    checked_interval,
    compact,
    goodness_of_fit,
    isotopologue_index,
    iso_timestamp,
    measurement_fit,
    option,
    seconds_between,
    software,
    solution,
    to_float,
    to_int,
)

__all__ = ["read_influx_si_results", "read_miso_sim", "parse_influx_log"]

#: influx_si options that take no value.
_FLAGS_WITHOUT_VALUE = frozenset(
    {
        "-h",
        "--help",
        "--version",
        "--noopt",
        "--noscale",
        "--fullsys",
        "--emu",
        "--irand",
        "--ln",
        "--tikhreg",
        "--sln",
        "--lim",
        "--ffguess",
        "--fdfit",
        "--addnoise",
        "--TIMEIT",
        "--prof",
        "--wkvh",
        "--parR",
        "--nocalc",
        "--force",
        "--copy_doc",
        "--copy_test",
        "--install_rcran",
    }
)
_TYPE = {"F": "free", "D": "dependent", "C": "constrained"}
_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_TIMESTAMP_RE = re.compile(r"^(code gen|calcul|end)\s*:\s*(\S+ \S+)\s*$")
_VERSION_RE = re.compile(r"influx_si version[:\s]+([0-9][\w.\-]*)", re.I)
_RUN_SUFFIX_RE = re.compile(r"\.V\d+$")
_LEVEL_RE = re.compile(r"level of ([\d.]+)\s*%", re.I)
_MC_CONFIDENCE = 0.95
_XCH_DERIVATION = (
    "xch = xch01 / (1 - xch01); SD by the delta method; MC bounds "
    "transformed exactly"
)
_MEASURED_DERIVATION = (
    "measured_value = simulated_value - weighted_residual * SD (input "
    "measurement file unavailable)"
)
_INST_DERIVATION = (
    "residual = weighted_residual * SD; simulated_value is the unscaled, "
    "normalised MID influx_si writes for non-stationary runs"
)


@dataclass
class InfluxLog:
    """What one influx_si run block of a ``.log`` file records.

    Attributes:
        command_line: The quoted argument line, verbatim.
        argv: The parsed arguments (program first).
        options: ``(flag, value)`` pairs in command-line order.
        timestamps: ``code gen`` / ``calcul`` / ``end`` in ISO form.
        warnings: ``***Warning`` / ``***Error`` lines.
        version: influx_si version, when the log states it.
    """

    command_line: Optional[str] = None
    argv: List[str] = field(default_factory=list)
    options: List[Tuple[str, Optional[str]]] = field(default_factory=list)
    timestamps: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    version: Optional[str] = None

    def flag(self, name: str) -> Optional[Tuple[str, Optional[str]]]:
        """Return the last ``(flag, value)`` pair for ``name``, if any."""
        found = None
        for key, value in self.options:
            if key == name:
                found = (key, value)
        return found


def read_influx_si_results(
    res_dir: Path | str,
    *,
    prefix: Optional[str] = None,
    input_dir: Optional[Path | str] = None,
    run_id: Optional[str] = None,
    software_version: Optional[str] = None,
    start: Optional[str] = None,
    flux_unit: Optional[str] = None,
    pool_size_unit: Optional[str] = None,
    investigation_id: Optional[str] = None,
) -> fl.SimulationRun:
    """Read an influx_si result directory into a :class:`SimulationRun`.

    Args:
        res_dir: The ``<prefix>_res`` directory written by influx_si.
        prefix: File prefix (default: the directory name without ``_res``,
            else the stem of the ``.tvar.sim`` file).
        input_dir: Directory holding the input MTF files, used to join the
            simulated values to the measured ones (default: the parent of
            ``res_dir``, where influx_si writes ``<prefix>_res``).
        run_id: Id of the run (default: ``<prefix>.influx_si``).
        software_version: influx_si version; overrides what the files say.
        start: Read one start of a multistart run (``"V2"``) instead of the
            best one influx_si copies to the unsuffixed files.
        flux_unit: Unit to state on every flux result (influx_si has none).
        pool_size_unit: Unit to state on every pool size result.
        investigation_id: Investigation the run belongs to.

    Returns:
        The completed simulation run.

    Raises:
        FileNotFoundError: If ``res_dir`` holds no influx_si result file.
    """
    res_dir = Path(res_dir)
    if not res_dir.is_dir():
        raise FileNotFoundError(f"{res_dir} is not a directory")
    prefix = prefix or _detect_prefix(res_dir)
    input_dir = Path(input_dir) if input_dir is not None else res_dir.parent
    pres = _read_pres(res_dir / "tmp" / f"{prefix}.pres.csv")
    if start is None and not (res_dir / f"{prefix}.tvar.sim").exists():
        start = pres.best if pres is not None else None
    suffix = f".{start}" if start else ""
    run_id = run_id or f"{prefix}.influx_si"

    log = _read_log(res_dir / f"{prefix}.log")
    r_opts = _read_r_options(res_dir / "tmp" / f"{prefix}.R")
    kvh = _read_kvh(_find_kvh(res_dir, prefix, suffix))
    noopt = log.flag("--noopt") is not None or r_opts.get("noopt") == "TRUE"
    sens = _option_value(log, r_opts, "sens")
    mc_runs = _mc_runs(sens)

    fluxes, pools = _read_tvar_sim(
        res_dir / f"{prefix}{suffix}.tvar.sim",
        flux_unit=flux_unit,
        pool_size_unit=pool_size_unit,
    )
    mids: List[fl.SimulatedMID] = []
    fits: List[fl.MeasurementFitResult] = []
    for path in _miso_sim_files(res_dir, suffix):
        experiment = path.name[: -len(f"{suffix}.miso.sim")]
        sim_mids, sim_fits = read_miso_sim(
            path, input_miso=input_dir / f"{experiment}.miso"
        )
        mids += sim_mids
        fits += sim_fits
    fits += _read_scalar_sim(
        res_dir / f"{prefix}{suffix}.mflux.sim",
        input_dir / f"{prefix}.mflux",
        name_column="Flux",
    )
    fits += _read_scalar_sim(
        res_dir / f"{prefix}{suffix}.mmet.sim",
        input_dir / f"{prefix}.mmet",
        name_column="Specie",
    )
    scale_factors = pres.scale_factors(start) if pres is not None else []
    gof = _read_stat(res_dir / f"{prefix}{suffix}.stat")
    err_text = _read_text(res_dir / f"{prefix}.err").strip()

    if not (fluxes or pools or fits or gof or log.command_line):
        raise FileNotFoundError(f"no influx_si result files in {res_dir}")

    modes = ["forward_simulation"] if noopt else ["estimation"]
    if mc_runs:
        modes += ["confidence_interval_estimation", "monte_carlo"]
    status = (
        "completed"
        if (fluxes or gof)
        else ("failed" if err_text else "unknown")
    )
    program = Path(log.argv[0]).name if log.argv else ""
    stationary = not program.startswith("influx_i") and not any(
        f.time_point is not None for f in fits
    )
    version = (
        software_version
        or log.version
        or kvh.get("influx/version")
        or r_opts.get("vernum")
        or _preamble_version(res_dir / "tmp" / f"{prefix}.ftbl")
    )
    start_time = log.timestamps.get("code gen") or log.timestamps.get("calcul")
    end_time = log.timestamps.get("end")
    emu = r_opts.get("emu")
    if emu is None and (
        log.flag("--emu") or "--emu" in _opt_file_flags(input_dir, prefix)
    ):
        emu = "TRUE"
    status_parts = [kvh.get("optimization process information/exit message")]
    status_parts += log.warnings
    if err_text:
        status_parts.append(f".err: {err_text[:500]}")
    iterations = to_int(
        kvh.get("optimization process information/iteration number")
    )

    fields: Dict[str, Any] = {
        "id": run_id,
        "investigation_id": investigation_id,
        "fluxomics_method": "IST_MFA" if stationary else "INST_MFA",
        "run_mode": modes,
        "run_status": status,
        "software": software(run_id, "influx_si", version),
        "input_format": "MTF",
        "command_line": log.command_line,
        "optimization_method": None
        if noopt
        else (_option_value(log, r_opts, "meth") or "nlsic"),
        "isotopic_state_space": (
            None if emu is None else ("EMU" if emu == "TRUE" else "cumomer")
        ),
        "multistart_count": (
            pres.n_starts if pres and pres.n_starts > 1 else None
        ),
        "monte_carlo_runs": mc_runs,
        "random_seed": to_int(_option_value(log, r_opts, "seed")),
        "solver_options": _solver_options(log, r_opts, input_dir, prefix),
        "start_time": start_time,
        "end_time": end_time,
        "runtime_seconds": seconds_between(start_time, end_time),
        "converged": None if noopt else _converged(kvh),
        "solver_status": " | ".join(p for p in status_parts if p) or None,
        "iteration_count": iterations,
        "flux_solution": solution(
            run_id,
            "reference" if noopt else "estimated",
            flux_results=fluxes,
            pool_size_results=pools,
            simulated_mids=mids,
            measurement_fits=fits,
            scale_factors=scale_factors,
        ),
        "goodness_of_fit": gof,
    }
    return fl.SimulationRun(**compact(fields))


# -- result tables ------------------------------------------------------------


def _read_table(path: Path, *, comment: Optional[str] = None) -> pd.DataFrame:
    """Return an influx_si TSV table with every cell as a stripped string."""
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        comment=comment,
        skip_blank_lines=True,
    )
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame.apply(lambda col: col.str.strip())


def _column_map(frame: pd.DataFrame) -> Dict[str, str]:
    """Return lower-cased canonical names -> the frame's column names."""
    aliases = {"species": "specie", "std": "sd"}
    out: Dict[str, str] = {}
    for col in frame.columns:
        key = col.lower()
        out[aliases.get(key, key)] = col
    return out


def _cell(row: pd.Series, cols: Dict[str, str], name: str) -> str:
    """Return a cell as text ('' when the column is absent)."""
    col = cols.get(name)
    return str(row[col]) if col is not None else ""


def _read_tvar_sim(
    path: Path,
    *,
    flux_unit: Optional[str],
    pool_size_unit: Optional[str],
) -> Tuple[List[fl.FluxResult], List[fl.PoolSizeResult]]:
    """Read ``.tvar.sim`` into flux and pool size results.

    Args:
        path: The ``.tvar.sim`` file (skipped when absent).
        flux_unit: Unit to state on every flux result.
        pool_size_unit: Unit to state on every pool size result.

    Returns:
        The flux results (net and absolute exchange) and pool sizes.
    """
    if not path.exists():
        return [], []
    frame = _read_table(path)
    cols = _column_map(frame)
    has_mc = "low_mc" in cols and "up_mc" in cols
    fluxes: List[fl.FluxResult] = []
    pools: List[fl.PoolSizeResult] = []
    for _, row in frame.iterrows():
        name = _cell(row, cols, "name")
        kind = _cell(row, cols, "kind").upper()
        raw_value = _cell(row, cols, "value")
        value = to_float(raw_value)
        sd = to_float(_cell(row, cols, "sd"))
        low = to_float(_cell(row, cols, "low_mc")) if has_mc else None
        up = to_float(_cell(row, cols, "up_mc")) if has_mc else None
        common: Dict[str, Any] = {
            "standard_deviation": sd,
            "variable_type": _TYPE.get(_cell(row, cols, "type").upper()),
            "structural_identifiability": (
                _cell(row, cols, "struct_identif") or None
            ),
        }
        if not name:
            continue
        if kind == "XCH" and value is not None:
            if value >= 1:
                warnings.warn(
                    f"{path.name}: XCH {name} = {raw_value!r} has no finite "
                    "absolute exchange flux; row skipped",
                    stacklevel=3,
                )
                continue
            native = f"influx_si xch01 = {raw_value}"
            if sd is not None:
                native += f" (SD {sd!r})"
            common["derivation"] = f"{_XCH_DERIVATION}; {native}"
            value, common["standard_deviation"] = _xch_absolute(value, sd)
            low, up = _xch_bound(low), _xch_bound(up)
        if value is None:
            warnings.warn(
                f"{path.name}: {kind} {name} has no value; row skipped",
                stacklevel=3,
            )
            continue
        lo, hi = checked_interval(
            value, low, up, label=f"{kind} {name}", method="monte_carlo"
        )
        if lo is not None or hi is not None:
            common.update(
                lower_ci=lo,
                upper_ci=hi,
                confidence_level=_MC_CONFIDENCE,
                confidence_interval_method="monte_carlo",
            )
        if kind == "METAB":
            pools.append(
                fl.PoolSizeResult(
                    **compact(
                        {
                            "metabolite_id": name,
                            "value": value,
                            "pool_size_unit": pool_size_unit,
                            **common,
                        }
                    )
                )
            )
        elif kind in ("NET", "XCH"):
            fluxes.append(
                fl.FluxResult(
                    **compact(
                        {
                            "reaction_id": name,
                            "flux_type": "net" if kind == "NET" else "exchange",
                            "value": value,
                            "flux_unit": flux_unit,
                            **common,
                        }
                    )
                )
            )
    return fluxes, pools


def _xch_absolute(
    xch01: float, sd: Optional[float]
) -> Tuple[float, Optional[float]]:
    """Convert ``xch01`` and its SD to the absolute exchange flux.

    Args:
        xch01: Normalised exchange flux in ``[0, 1)``.
        sd: Its SD (delta method: divided by ``(1 - xch01)**2``).

    Returns:
        The absolute exchange flux and its SD.
    """
    value = xch01 / (1.0 - xch01)
    return value, None if sd is None else sd / (1.0 - xch01) ** 2


def _xch_bound(bound: Optional[float]) -> Optional[float]:
    """Convert one ``xch01`` interval bound to the absolute scale.

    Args:
        bound: The ``xch01`` bound.

    Returns:
        ``bound / (1 - bound)``, or ``None`` when missing or ``>= 1``
        (unbounded).
    """
    if bound is None or bound >= 1:
        return None
    return bound / (1.0 - bound)


def _miso_sim_files(res_dir: Path, suffix: str) -> List[Path]:
    """Return the ``.miso.sim`` files of one start (one per experiment)."""
    files = []
    for path in sorted(res_dir.glob("*.miso.sim")):
        stem = path.name[: -len(".miso.sim")]
        if suffix:
            if stem.endswith(suffix):
                files.append(path)
        elif not _RUN_SUFFIX_RE.search(stem):
            files.append(path)
    return files


def _positions(fragment: str) -> Optional[Tuple[int, ...]]:
    """Parse a fragment (``"1,2,3"``, ``"1~3"``, ``"1-3"``) to positions.

    Args:
        fragment: The Fragment cell.

    Returns:
        The positions; ``()`` for an empty fragment (the whole molecule)
        and ``None`` when the text is not a list of positions.
    """
    text = fragment.replace(" ", "")
    if not text:
        return ()
    out: List[int] = []
    for part in text.split(","):
        match = re.fullmatch(r"(\d+)(?:[~-](\d+))?", part)
        if not match:
            return None
        lo = int(match.group(1))
        hi = int(match.group(2) or lo)
        out.extend(range(lo, hi + 1))
    return tuple(out)


def _time_key(text: str) -> Optional[float]:
    """Return a Time cell as a rounded float for matching (None if empty)."""
    number = to_float(text)
    return None if number is None else round(number, 9)


def _sim_kind(dataset: str, isospecies: str) -> str:
    """Return the measurement kind of a ``.miso.sim`` row."""
    head = dataset.split("_", 1)[0].upper()
    if head == "MS":
        return "mid"
    if head == "LAB":
        return "labeling_other"
    if head == "PEAK":
        return "nmr"
    return (
        "mid"
        if isotopologue_index(isospecies) is not None
        else ("labeling_other")
    )


class _InputMiso:
    """Rows of the input ``.miso`` file, consumed as they are matched."""

    def __init__(self, path: Optional[Path]) -> None:
        self.rows: Dict[Tuple[str, str, Optional[float]], List[dict]] = (
            defaultdict(list)
        )
        if path is None or not path.exists():
            return
        frame = _read_table(path, comment="#")
        cols = _column_map(frame)
        for _, row in frame.iterrows():
            specie = _cell(row, cols, "specie")
            if not specie:
                continue
            fragment = _cell(row, cols, "fragment")
            key = (
                specie,
                _cell(row, cols, "isospecies"),
                _time_key(_cell(row, cols, "time")),
            )
            self.rows[key].append(
                {
                    "fragment": fragment,
                    "positions": _positions(fragment),
                    "value": to_float(_cell(row, cols, "value")),
                    "dataset": _cell(row, cols, "dataset"),
                }
            )

    def take(
        self,
        specie: str,
        isospecies: str,
        time: Optional[float],
        fragment: str,
        kind: str,
    ) -> Optional[dict]:
        """Return and consume the input row a simulated row belongs to.

        influx_si rewrites the fragment: a whole-molecule fragment
        (``1,2,3`` for a three-carbon species) is written as ``""``. An
        exact position match wins; an empty simulated fragment otherwise
        matches the longest ``1..k`` input fragment. Labeling (LAB) and
        peak rows match on species, isospecies and time alone.
        """
        candidates = self.rows.get((specie, isospecies, time))
        if not candidates:
            return None
        if kind != "mid":
            return candidates.pop(0)
        wanted = _positions(fragment)
        for i, cand in enumerate(candidates):
            if cand["positions"] == wanted:
                return candidates.pop(i)
        if wanted == ():
            best, best_len = None, -1
            for i, cand in enumerate(candidates):
                pos = cand["positions"]
                if pos is not None and pos == tuple(range(1, len(pos) + 1)):
                    if len(pos) > best_len:
                        best, best_len = i, len(pos)
            if best is not None:
                return candidates.pop(best)
        return None


def read_miso_sim(
    path: Path | str, *, input_miso: Optional[Path | str] = None
) -> Tuple[List[fl.SimulatedMID], List[fl.MeasurementFitResult]]:
    """Read one influx_si ``.miso.sim`` file.

    Args:
        path: The ``.miso.sim`` file (columns ``Specie Fragment Dataset
            Isospecies Value SD Time Residual Pvalue``).
        input_miso: The input ``.miso`` file the run used, joined on
            species, fragment, isospecies and time for the measured values.

    Returns:
        One :class:`SimulatedMID` per MS dataset and one
        :class:`MeasurementFitResult` per row. ``measurement_id`` and
        ``fragment_name`` are influx_si's output ``Dataset`` name
        (``MS_F_``, ``MS_C_1``, ``LAB_E_1xx``). Without a matching input
        row the measured value of a stationary row is recovered as
        ``Value - Residual * SD`` and the fit says so in ``derivation``.
    """
    path = Path(path)
    frame = _read_table(path)
    cols = _column_map(frame)
    inputs = _InputMiso(Path(input_miso) if input_miso else None)
    groups: Dict[str, dict] = {}
    fits: List[fl.MeasurementFitResult] = []
    for _, row in frame.iterrows():
        specie = _cell(row, cols, "specie")
        if not specie:
            continue
        fragment = _cell(row, cols, "fragment")
        dataset = _cell(row, cols, "dataset") or f"{specie}:{fragment}"
        iso = _cell(row, cols, "isospecies")
        time = _time_key(_cell(row, cols, "time"))
        kind = _sim_kind(dataset, iso)
        simulated = to_float(_cell(row, cols, "value"))
        sd = to_float(_cell(row, cols, "sd"))
        weighted = to_float(_cell(row, cols, "residual"))
        match = inputs.take(specie, iso, time, fragment, kind)
        inst = time is not None  # values are unscaled for INST runs
        measured, residual, notes = None, None, []
        if match is not None:
            measured = match["value"]
        elif not inst and None not in (simulated, weighted, sd):
            measured = simulated - weighted * sd
            notes.append(_MEASURED_DERIVATION)
        if inst and weighted is not None and sd is not None:
            residual = weighted * sd
            notes.append(_INST_DERIVATION)
        index = isotopologue_index(iso) if kind == "mid" else None
        fits.append(
            measurement_fit(
                dataset,
                measured=measured,
                simulated=simulated,
                sd=sd,
                weighted_residual=weighted,
                isotopologue_index=index,
                time_point=time,
                p_value=to_float(_cell(row, cols, "pvalue")),
                residual=residual,
                derivation="; ".join(notes) or None,
            )
        )
        if kind != "mid" or index is None or simulated is None:
            continue
        group = groups.setdefault(
            dataset,
            {
                "specie": specie,
                "fragment": fragment,
                "atoms": None,
                "scaled": not inst,
                "points": [],
            },
        )
        if match is not None and match["fragment"] and not group["atoms"]:
            group["atoms"] = match["fragment"]
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
    for dataset, group in groups.items():
        specie = group["specie"]
        pooled = "+" in specie
        mids.append(
            fl.SimulatedMID(
                **compact(
                    {
                        "metabolite_id": None if pooled else specie,
                        "measurement_spec": specie if pooled else None,
                        "fragment_name": dataset,
                        "atom_positions": group["atoms"]
                        or group["fragment"]
                        or None,
                        "scaled": group["scaled"],
                        "data_points": group["points"],
                    }
                )
            )
        )
    return mids, fits


def _read_scalar_sim(
    path: Path, input_path: Path, *, name_column: str
) -> List[fl.MeasurementFitResult]:
    """Read ``.mflux.sim`` / ``.mmet.sim`` (Value = simulated).

    Args:
        path: The result file (skipped when absent).
        input_path: The matching input ``.mflux`` / ``.mmet`` file.
        name_column: Column holding the flux or species name.

    Returns:
        One fit per row.
    """
    if not path.exists():
        return []
    measured_by_name: Dict[str, List[Optional[float]]] = defaultdict(list)
    if input_path.exists():
        source = _read_table(input_path, comment="#")
        src_cols = _column_map(source)
        for _, row in source.iterrows():
            name = _cell(row, src_cols, name_column.lower())
            if name:
                measured_by_name[name].append(
                    to_float(_cell(row, src_cols, "value"))
                )
    frame = _read_table(path)
    cols = _column_map(frame)
    fits = []
    for _, row in frame.iterrows():
        name = _cell(row, cols, name_column.lower())
        if not name:
            continue
        simulated = to_float(_cell(row, cols, "value"))
        sd = to_float(_cell(row, cols, "sd"))
        weighted = to_float(_cell(row, cols, "residual"))
        measured, derivation = None, None
        if measured_by_name.get(name):
            measured = measured_by_name[name].pop(0)
        elif None not in (simulated, weighted, sd):
            measured = simulated - weighted * sd
            derivation = _MEASURED_DERIVATION
        fits.append(
            measurement_fit(
                name,
                measured=measured,
                simulated=simulated,
                sd=sd,
                weighted_residual=weighted,
                p_value=to_float(_cell(row, cols, "pvalue")),
                derivation=derivation,
            )
        )
    return fits


def _read_stat(path: Path) -> Optional[fl.GoodnessOfFit]:
    """Return ``.stat`` as GoodnessOfFit (influx_si's lower-tail test)."""
    if not path.exists():
        return None
    values: Dict[str, str] = {}
    for line in _read_text(path).splitlines():
        key, _, value = line.partition("\t")
        if key.strip():
            values[key.strip()] = value.strip()
    conclusion = values.get("conclusion", "")
    accepted = None
    if conclusion:
        accepted = "does not fit" not in conclusion.lower()
    level = _LEVEL_RE.search(conclusion)
    return goodness_of_fit(
        wssr=to_float(values.get("chi2_value")),
        reduced_chi_square=to_float(values.get("chi2/df")),
        number_of_measurements=to_int(values.get("number_of_measurements")),
        number_of_parameters=to_int(values.get("number_of_parameters")),
        degrees_of_freedom=to_int(values.get("degrees_of_freedom")),
        chi_square_p_value=_upper_tail(to_float(values.get("p-value"))),
        chi_square_confidence_level=(
            float(level.group(1)) / 100 if level else None
        ),
        fit_accepted=accepted,
    )


def _upper_tail(p_lower: Optional[float]) -> Optional[float]:
    """Convert influx_si's lower-tail chi-square p-value to the upper tail.

    influx_si writes ``pchisq(chi2, df)``; the schema stores P(X >= chi2).

    Returns:
        ``1 - p_lower``, or ``None`` when no p-value was reported.
    """
    return None if p_lower is None else 1.0 - p_lower


@dataclass
class _Pres:
    """``tmp/<prefix>.pres.csv``: cost and parameters per start."""

    frame: pd.DataFrame

    @property
    def n_starts(self) -> int:
        """Number of starts (columns)."""
        return len(self.frame.columns)

    @property
    def best(self) -> Optional[str]:
        """Column of the lowest cost, when there are several starts."""
        if self.n_starts < 2 or "cost" not in self.frame.index:
            return None
        costs = {
            col: to_float(self.frame.loc["cost", col])
            for col in self.frame.columns
        }
        finite = {k: v for k, v in costs.items() if v is not None}
        return min(finite, key=finite.__getitem__) if finite else None

    def scale_factors(self, start: Optional[str]) -> List[fl.ScaleFactorResult]:
        """Return the scale factor rows of one start (default: best or only)."""
        column = start or self.best or self.frame.columns[0]
        if column not in self.frame.columns:
            return []
        out = []
        for name, value in self.frame[column].items():
            name = str(name)
            if name == "cost" or name.startswith(("f.", "pf:")):
                continue
            number = to_float(value)
            if number is not None:
                out.append(
                    fl.ScaleFactorResult(measurement_id=name, value=number)
                )
        return out


def _read_pres(path: Path) -> Optional[_Pres]:
    """Return ``.pres.csv`` (tab-separated, row names first), or None."""
    if not path.exists():
        return None
    frame = _read_table(path)
    if frame.empty:
        return None
    frame = frame.set_index(frame.columns[0])
    return _Pres(frame)


# -- provenance ---------------------------------------------------------------


def _read_text(path: Path) -> str:
    """Return a text file's content ('' when absent)."""
    return (
        path.read_text(encoding="utf-8", errors="replace")
        if (path.exists())
        else ""
    )


def _detect_prefix(res_dir: Path) -> str:
    """Infer the influx_si prefix of a result directory.

    Args:
        res_dir: The result directory.

    Returns:
        The prefix.

    Raises:
        FileNotFoundError: If no influx_si file reveals the prefix.
    """
    name = res_dir.name
    if name.endswith("_res"):
        candidate = name[: -len("_res")]
        if (
            any(res_dir.glob(f"{candidate}*.sim"))
            or (res_dir / f"{candidate}.log").exists()
        ):
            return candidate
    for pattern in ("*.tvar.sim", "*.log", "*.stat"):
        for path in sorted(res_dir.glob(pattern)):
            stem = path.name.split(".")[0]
            if stem:
                return stem
    raise FileNotFoundError(f"cannot find an influx_si prefix in {res_dir}")


def parse_influx_log(text: str) -> InfluxLog:
    """Parse the last run block of an influx_si ``.log``.

    influx_si opens the log in append mode, so one file can hold several
    runs; each starts with the quoted command line.

    Args:
        text: The log content.

    Returns:
        The parsed run block.
    """
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith('"')]
    block = lines[starts[-1] :] if starts else lines
    log = InfluxLog()
    if starts:
        log.command_line = block[0].strip()
        try:
            log.argv = shlex.split(log.command_line)
        except ValueError:
            log.argv = log.command_line.replace('"', "").split()
        log.options = _parse_argv(log.argv[1:])
    for line in block[1:]:
        stamp = _TIMESTAMP_RE.match(line.strip())
        if stamp:
            log.timestamps[stamp.group(1)] = iso_timestamp(stamp.group(2))
        if "***Warning" in line or "***Error" in line:
            log.warnings.append(line.strip())
        version = _VERSION_RE.search(line)
        if version and not log.version:
            log.version = version.group(1)
    return log


def _read_log(path: Path) -> InfluxLog:
    """Return the parsed last run block of an influx_si ``.log`` file."""
    return parse_influx_log(_read_text(path))


def _looks_like_value(token: str) -> bool:
    """Return whether a token can be an option value."""
    return not token.startswith("-") or bool(_NUMBER_RE.match(token))


def _parse_argv(args: Sequence[str]) -> List[Tuple[str, Optional[str]]]:
    """Split influx_si arguments into ``(flag, value)`` pairs.

    Positional arguments (FTBL names) are left out; they stay in
    ``command_line``.

    Args:
        args: The arguments after the program name.

    Returns:
        ``(flag, value)`` pairs in order; ``value`` is None for a flag.
    """
    out: List[Tuple[str, Optional[str]]] = []
    i = 0
    while i < len(args):
        token = args[i]
        i += 1
        if not token.startswith("-") or _NUMBER_RE.match(token):
            continue
        if "=" in token:
            key, _, value = token.partition("=")
            out.append((key, value))
            continue
        if (
            token not in _FLAGS_WITHOUT_VALUE
            and i < len(args)
            and (_looks_like_value(args[i]))
        ):
            out.append((token, args[i]))
            i += 1
        else:
            out.append((token, None))
    return out


def _r_block(lines: List[str], header: str) -> Dict[str, str]:
    """Return the ``name=value`` lines after a header, up to a blank line."""
    out: Dict[str, str] = {}
    try:
        start = lines.index(header) + 1
    except ValueError:
        return out
    for line in lines[start:]:
        if not line.strip():
            break
        name, sep, value = line.partition("=")
        if sep and re.fullmatch(r"[A-Za-z_][\w.]*", name.strip()):
            out[name.strip()] = value.strip().strip('"')
    return out


def _read_r_options(path: Path) -> Dict[str, str]:
    """Read the effective options of the generated ``<prefix>.R``.

    Args:
        path: The generated R program (``tmp/<prefix>.R``).

    Returns:
        The defaults merged with the runtime arguments, plus
        ``runtime:<name>`` entries for the runtime arguments alone and
        ``vernum`` for the influx_si version; empty when absent.
    """
    text = _read_text(path)
    if not text:
        return {}
    lines = [line.rstrip() for line in text.splitlines()]
    defaults = _r_block(lines, "# default options")
    runtime = _r_block(lines, "# get runtime arguments")
    merged = {**defaults, **runtime}
    merged.update({f"runtime:{k}": v for k, v in runtime.items()})
    version = re.search(r'^vernum="([^"]+)"', text, re.M)
    if version:
        merged["vernum"] = version.group(1)
    return merged


def _opt_file_flags(input_dir: Path, prefix: str) -> Dict[str, Optional[str]]:
    """Return the ``commandArgs`` flags of the input ``<prefix>.opt``."""
    path = input_dir / f"{prefix}.opt"
    if not path.exists():
        return {}
    frame = _read_table(path, comment="#")
    cols = _column_map(frame)
    flags: Dict[str, Optional[str]] = {}
    for _, row in frame.iterrows():
        if _cell(row, cols, "name") == "commandArgs":
            try:
                tokens = shlex.split(_cell(row, cols, "value"))
            except ValueError:
                tokens = _cell(row, cols, "value").split()
            flags.update(dict(_parse_argv(tokens)))
    return flags


def _option_value(
    log: InfluxLog, r_opts: Dict[str, str], name: str
) -> Optional[str]:
    """Return an option's effective value (command line, else ``.R``)."""
    found = log.flag(f"--{name}")
    if found is not None and found[1] is not None:
        return found[1]
    value = r_opts.get(f"runtime:{name}")
    return value or None


def _mc_runs(sens: Optional[str]) -> Optional[int]:
    """Return the Monte Carlo count of ``--sens mc[=N]`` (default 10)."""
    if not sens:
        return None
    sens = sens.strip().strip('"')
    if sens == "mc":
        return 10
    if sens.startswith("mc="):
        return to_int(sens[3:])
    return None


def _solver_options(
    log: InfluxLog, r_opts: Dict[str, str], input_dir: Path, prefix: str
) -> List[fl.SolverOption]:
    """Build ``solver_options``: command-line flags, then ``.opt`` ones.

    Options set only through the ``.opt`` file are taken from the runtime
    block of the generated ``.R`` (the effective values) or, when that is
    missing, from the input ``.opt`` file.

    Args:
        log: The parsed log.
        r_opts: Options of the generated ``.R``.
        input_dir: Directory of the input files.
        prefix: The influx_si prefix.

    Returns:
        The solver options.
    """
    options = [option(key, value) for key, value in log.options]
    if not any(key == "--prefix" for key, _ in log.options):
        options.insert(0, option("--prefix", prefix))
    on_command_line = {key.lstrip("-") for key, _ in log.options}
    runtime = {
        k[len("runtime:") :]: v
        for k, v in r_opts.items()
        if k.startswith("runtime:")
    }
    if runtime:
        for name, value in runtime.items():
            if name in on_command_line or value == "FALSE":
                continue
            options.append(
                option(
                    f"--{name}",
                    None if value == "TRUE" else value,
                    f"Not on the command line; effective value from the "
                    f"generated {prefix}.R (set by {prefix}.opt).",
                )
            )
    else:
        for key, value in _opt_file_flags(input_dir, prefix).items():
            if key.lstrip("-") in on_command_line:
                continue
            options.append(
                option(key, value, f"From {prefix}.opt commandArgs.")
            )
    return options


def _find_kvh(res_dir: Path, prefix: str, suffix: str) -> Optional[Path]:
    """Return the ``--wkvh`` result file of one start, or None."""
    for path in (
        res_dir / "tmp" / f"{prefix}_res{suffix}.kvh",
        res_dir / f"{prefix}{suffix}.kvh",
        res_dir / f"{prefix}_res{suffix}.kvh",
    ):
        if path.exists():
            return path
    return None


def _read_kvh(path: Optional[Path]) -> Dict[str, str]:
    """Flatten a KVH file to ``"section/sub/key" -> value``.

    KVH nests keys by leading tabs; a key and its value are tab-separated.

    Args:
        path: The KVH file, or None.

    Returns:
        Path-like key -> value (empty when there is no file).
    """
    out: Dict[str, str] = {}
    if path is None:
        return out
    stack: List[str] = []
    for line in _read_text(path).splitlines():
        if not line.strip():
            continue
        depth = len(line) - len(line.lstrip("\t"))
        key, _, value = line.lstrip("\t").partition("\t")
        stack = stack[:depth] + [key.strip()]
        out["/".join(stack)] = value.strip()
    return out


def _converged(kvh: Dict[str, str]) -> Optional[bool]:
    """Return convergence inferred from nlsic's exit message (``.kvh``)."""
    key = "optimization process information/exit message"
    if key not in kvh:
        return None
    message = kvh[key].lower()
    if any(word in message for word in ("maxim", "fail", "error", "not ")):
        return False
    return True


def _preamble_version(path: Path) -> Optional[str]:
    """Return ``X`` of ``influx_si version: X`` in a file's preamble."""
    for line in _read_text(path).splitlines()[:10]:
        match = _VERSION_RE.search(line)
        if match:
            return match.group(1)
    return None
