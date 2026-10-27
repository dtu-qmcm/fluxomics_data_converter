"""Read and write FluxoLink records (``FluxoDataset`` YAML files).

A record is not one more model format: the tools never read it. It keeps
the reference copy of a study's model, points at the real tool files with
their checksums, and collects each tool's results (see
:mod:`fluxomics_data_converter.link`)::

    ds = load_record("ec.yaml")
    ...
    save_record(ds, "ec.yaml")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fluxolink.datamodel import fluxolink as fl


class _IndentedDumper(yaml.SafeDumper):
    """Safe dumper that indents block sequences under their parent key.

    PyYAML writes list items flush with their parent key by default; the
    FluxoLink repository's yamllint configuration (``indent-sequences:
    true``) expects them indented by two spaces.
    """

    def increase_indent(
        self, flow: bool = False, indentless: bool = False
    ) -> None:
        """Never emit indentless sequences."""
        super().increase_indent(flow, False)


def record_document(ds: fl.FluxoDataset) -> Dict[str, Any]:
    """Return a record as plain data, keeping only the slots that were set.

    ``exclude_unset`` keeps schema ``ifabsent`` defaults (``is_pseudo:
    false``, ``instance: 1``) out of the file, so it states only what the
    source said.

    Args:
        ds: The record.

    Returns:
        The record as a JSON-compatible dict.
    """
    return ds.model_dump(mode="json", exclude_unset=True, exclude_none=True)


def record_text(document: Dict[str, Any], header: Optional[str] = None) -> str:
    """Render a record document as YAML text.

    Args:
        document: The record as plain data.
        header: Optional comment block written above the document; each
            line is prefixed with ``#``.

    Returns:
        The YAML text.
    """
    text = yaml.dump(
        document,
        Dumper=_IndentedDumper,
        sort_keys=False,
        allow_unicode=True,
        width=88,
    )
    if header:
        comment = "".join(
            f"# {line}".rstrip() + "\n" for line in header.splitlines()
        )
        text = comment + text
    return text


def save_record(
    ds: fl.FluxoDataset, path: Path | str, *, header: Optional[str] = None
) -> Path:
    """Write a record as YAML.

    Args:
        ds: The record.
        path: Output file; parent directories are created.
        header: Optional comment block written above the document; each
            line is prefixed with ``#``.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record_text(record_document(ds), header), encoding="utf-8")
    return path


def load_record(path: Path | str) -> fl.FluxoDataset:
    """Load a record into the FluxoLink Pydantic classes.

    Args:
        path: The YAML file.

    Returns:
        The validated record.
    """
    with open(path, encoding="utf-8") as handle:
        return fl.FluxoDataset.model_validate(yaml.safe_load(handle))
