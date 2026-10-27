"""Describe tool files in a record: one ``SourceFile`` per physical file.

A tool input is often several files: an MTF input is ``.netw``, ``.linp``,
``.miso`` and more next to one prefix, a FreeFlux input is a directory of
tables. :func:`tool_files` lists the physical files of one input with the
role each plays, and :func:`source_file` describes one of them with its
checksum, so a record can point at the exact bytes a tool was given.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from fluxolink.datamodel import fluxolink as fl

from ..io import format_conventions as conv
from ..io.freeflux_parser import FreefluxParser

if TYPE_CHECKING:
    from ..core.core import FluxomicsData

#: File suffix -> IANA media type.
MEDIA_TYPES = {
    ".fml": "application/xml",
    ".xml": "application/xml",
    ".m": "text/x-matlab",
    ".mat": "application/x-matlab-data",
    ".tsv": "text/tab-separated-values",
    ".csv": "text/csv",
    ".xlsx": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ),
}

#: MTF file suffix -> the role the file plays.
MTF_ROLES = {
    ".netw": "network",
    ".linp": "tracer",
    ".miso": "mid_measurements",
    ".mflux": "flux_measurements",
    ".mmet": "pool_size_measurements",
    ".cnstr": "constraints",
    ".tvar": "variable_specification",
    ".opt": "configuration",
}

#: FreeFlux table kind -> the role the table plays.
FREEFLUX_ROLES = {
    "reactions": "network",
    "fluxes": "variable_specification",
    "concentrations": "variable_specification",
    "label_input": "tracer",
    "flux_bounds": "constraints",
    "measured_MDVs": "mid_measurements",
    "measured_fluxes": "flux_measurements",
    "measured_inst_MDVs": "mid_measurements",
}

#: One physical file of a tool input: (path, roles, ``InputFormatEnum``).
FileEntry = Tuple[Path, List[str], str]


def single_file_roles(data: FluxomicsData, fmt: Optional[str]) -> List[str]:
    """List the roles a single-file input (FluxML, INCA) plays.

    Args:
        data: The model parsed from the file.
        fmt: Its ``InputFormatEnum`` value.

    Returns:
        File roles derived from the content the model holds.
    """
    roles = ["network"]
    if data.constraints or any(e.constraints for e in data.experiments):
        roles.append("constraints")
    if any(e.tracers for e in data.experiments):
        roles.append("tracer")
    models = [e.measurement.model for e in data.experiments if e.measurement]
    if any(m.labeling_measurement for m in models):
        roles.append("mid_measurements")
    if any(m.flux_measurement for m in models):
        roles.append("flux_measurements")
    if any(m.metabolitesize_measurement for m in models):
        roles.append("pool_size_measurements")
    if any(e.simulation for e in data.experiments):
        roles.append("simulation")
        if any(
            e.simulation and e.simulation.variables for e in data.experiments
        ):
            roles.append("variable_specification")
    if fmt == conv.INCA:
        roles.append("driver_script")
    return roles


def tool_files(
    path: Path, fmt: Optional[str], data: Optional[FluxomicsData] = None
) -> List[FileEntry]:
    """List the physical files of one tool input.

    Args:
        path: A FluxML or INCA file, an MTF prefix (or its ``.netw``), or a
            FreeFlux directory (or one of its tables).
        fmt: The input's ``InputFormatEnum`` value.
        data: The model parsed from the input; gives the roles of a
            single-file input. Without it such a file is recorded as
            ``network`` only.

    Returns:
        The existing files, each with its roles and format.
    """
    path = Path(path)
    entries: List[FileEntry] = []
    if fmt == conv.MTF:
        prefix = path.with_suffix("") if path.suffix in MTF_ROLES else path
        for suffix in conv.MTF_EXTENSIONS:
            candidate = prefix.with_suffix(suffix)
            if candidate.exists():
                entries.append((candidate, [MTF_ROLES[suffix]], conv.MTF))
    elif fmt in conv.FREEFLUX_FORMATS:
        directory = path.parent if path.is_file() else path
        finder = FreefluxParser()
        for kind, role in FREEFLUX_ROLES.items():
            found = finder._find_file(directory, kind)
            if found is not None:
                entries.append(
                    (found, [role], conv.freeflux_format(found) or fmt)
                )
    elif path.is_file() and fmt is not None:
        roles = (
            single_file_roles(data, fmt) if data is not None else ["network"]
        )
        entries.append((path, roles, fmt))
    return entries


def checksum(content: bytes) -> str:
    """Return the ``SourceFile.checksum`` of some bytes.

    Returns:
        ``"sha256:<hex digest>"``.
    """
    return "sha256:" + hashlib.sha256(content).hexdigest()


def file_checksum(path: Path) -> str:
    """Return the ``SourceFile.checksum`` of a file.

    Returns:
        ``"sha256:<hex digest>"``.
    """
    return checksum(Path(path).read_bytes())


def record_path(path: Path, root: Optional[Path]) -> str:
    """Return the path a record stores for a file.

    Args:
        path: The file.
        root: Directory the record's paths are relative to, normally the
            record's own directory; ``None`` keeps the file name only.

    Returns:
        A POSIX path relative to ``root`` (it may start with ``../``).
    """
    path = Path(path)
    if root is None:
        return path.name
    return Path(
        os.path.relpath(path.resolve(), Path(root).resolve())
    ).as_posix()


def source_file(
    path: Path,
    *,
    file_id: str,
    fmt: str,
    roles: List[str],
    recorded_path: str,
    **fields: Any,
) -> fl.SourceFile:
    """Describe one physical file.

    Args:
        path: The file on disk (read for the checksum and size).
        file_id: Id of the record entry.
        fmt: Its ``InputFormatEnum`` value.
        roles: The roles it plays.
        recorded_path: The path stored in the record.
        **fields: Further ``SourceFile`` slots, such as ``generated_by``,
            ``derived_from`` or ``description``.

    Returns:
        The ``SourceFile``.
    """
    content = Path(path).read_bytes()
    entry: Dict[str, Any] = {
        "id": file_id,
        "path": recorded_path,
        "format": fmt,
        "role": roles,
        "checksum": checksum(content),
        "file_size_bytes": len(content),
    }
    if fmt not in {e.value for e in fl.InputFormatEnum}:
        entry["format"] = "other"
        entry["format_other"] = fmt
    media = MEDIA_TYPES.get(Path(path).suffix.lower())
    if media:
        entry["media_type"] = media
    entry.update({k: v for k, v in fields.items() if v not in (None, [], "")})
    return fl.SourceFile(**entry)
