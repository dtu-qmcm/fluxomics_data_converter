"""Attach a tool's results to the record.

:func:`record_tool_input` leaves one planned simulation run per tool and
configuration, pointing at the input files. :func:`add_results` fills that
run with what the tool produced (read by :mod:`.results`), records the
result files, and marks the run with the results' status, so each result
stays tied to the exact input it came from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from fluxolink.datamodel import fluxolink as fl

from .files import FileEntry
from .tool_inputs import upsert_files

#: Run slots that belong to the planned run, not to the results.
_KEEP = (
    "id",
    "input_files",
    "network_model_id",
    "experiment_id",
    "configuration_name",
    "investigation_id",
    "input_format",
    "fluxomics_method",
)


def _software_name(run: fl.SimulationRun) -> Optional[str]:
    software = run.software
    if software is None:
        return None
    name = getattr(software.software_name, "value", software.software_name)
    if name == "other":
        return software.software_name_other
    return name


def _reid(value: Any, old: str, new: str) -> Any:
    """Rename nested ids that were derived from the results run's id.

    Returns:
        The value with every ``id`` starting with ``old`` re-prefixed.
    """
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "id" and isinstance(item, str) and item.startswith(old):
                out[key] = new + item[len(old) :]
            else:
                out[key] = _reid(item, old, new)
        return out
    if isinstance(value, list):
        return [_reid(item, old, new) for item in value]
    return value


def result_files(path: Path, tool: str) -> List[FileEntry]:
    """List the result files a reader takes from a results path.

    Args:
        path: An influx_si ``<prefix>_res`` directory or an INCA ``.mat``.
        tool: ``influx_si`` or ``INCA``.

    Returns:
        The existing files, with role ``results``.
    """
    path = Path(path)
    if path.is_file():
        fmt = "INCA_mat" if path.suffix == ".mat" else "other"
        return [(path, ["results"], fmt)]
    patterns = ("*.sim", "*.stat", "*.log", "*.kvh", "tmp/*.pres.csv")
    found = sorted({p for pattern in patterns for p in path.glob(pattern)})
    return [(p, ["results"], f"{tool} results") for p in found if p.is_file()]


def add_results(
    ds: fl.FluxoDataset,
    run: fl.SimulationRun,
    *,
    output_files: List[FileEntry] = (),
    root: Optional[Path] = None,
) -> fl.FluxoDataset:
    """Put a tool's results into the record.

    The results go into the planned run of the same software (and
    configuration, when the results name one) if there is exactly one;
    otherwise they are added as a new run.

    Args:
        ds: The record.
        run: The results, as read by :mod:`.results`.
        output_files: The files the results were read from.
        root: Directory the recorded paths are relative to.

    Returns:
        The updated record (a copy).
    """
    ds = ds.model_copy(deep=True)
    output_ids = upsert_files(ds, list(output_files), root=root)
    runs = list(ds.simulation_runs or [])
    name = _software_name(run)
    planned = [
        r
        for r in runs
        if r.run_status in (None, "planned")
        and _software_name(r) == name
        and (
            run.configuration_name is None
            or r.configuration_name in (None, run.configuration_name)
        )
    ]
    new = run.model_dump(mode="json", exclude_unset=True, exclude_none=True)
    if len(planned) == 1:
        target = planned[0]
        kept = target.model_dump(
            mode="json", exclude_unset=True, exclude_none=True
        )
        merged = _reid(new, run.id, target.id)
        merged.update({k: kept[k] for k in _KEEP if k in kept})
        if output_ids:
            merged["output_files"] = output_ids
        runs[runs.index(target)] = fl.SimulationRun.model_validate(merged)
    else:
        networks = ds.networks or []
        if len(networks) == 1:
            new.setdefault("network_model_id", networks[0].id)
        investigations = ds.investigations or []
        if len(investigations) == 1:
            new.setdefault("investigation_id", investigations[0].id)
        if output_ids:
            new["output_files"] = output_ids
        runs.append(fl.SimulationRun.model_validate(new))
    ds.simulation_runs = runs
    return ds
