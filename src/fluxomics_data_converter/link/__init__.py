"""Link tool inputs, runs and results of one study in a FluxoLink record.

The converter translates directly between tools (:mod:`..io`): FluxML,
MTF, FreeFlux and INCA files go through
:class:`~fluxomics_data_converter.FluxomicsData` and nothing else. A
FluxoLink record is not one more format in that chain; no tool reads it.
It ties the tool files together:

- it keeps the reference copy of the model (network, atom maps, tracers,
  measurements), which every tool input is checked against
  (:mod:`.record_from_model`, :mod:`.model_from_record`);
- it points at each tool's actual input files with checksums, says which
  files were generated from which, and stores the result of checking that
  each encodes the same problem (:mod:`.tool_inputs`, :mod:`.equivalence`);
- it maps each tool's ids onto the canonical ones (:mod:`.aliases`);
- it holds each tool's results, tied to the input they came from
  (:mod:`.runs`, :mod:`.results`).

Typical use::

    from fluxomics_data_converter.io import parse_fluxml_file
    from fluxomics_data_converter.link import (
        check_tool_input, record_tool_input, save_record, to_fluxo_dataset,
        write_tool_input,
    )

    data = parse_fluxml_file("EC.fml")
    ds = to_fluxo_dataset(data, dataset_id="ec", source_path=Path("EC.fml"))
    mtf = write_tool_input(data, "mtf", Path("runs/EC"), experiment="a_STAT")
    check = check_tool_input(data, mtf)  # parse back, compare with data
    ds = record_tool_input(ds, mtf, check, root=Path("."), generated=True)
    save_record(ds, "ec.yaml")

The ``fluxo`` command (:mod:`fluxomics_data_converter.cli`) wraps these.
Everything except :func:`compare` needs the optional ``linkml`` extra
(``uv sync --extra linkml``).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .equivalence import Difference, EquivalenceReport, compare

#: Public name -> defining submodule; imported on first use because these
#: need the optional ``fluxolink`` package.
_LAZY = {
    "load_record": "record",
    "save_record": "record",
    "record_document": "record",
    "record_text": "record",
    "to_fluxo_dataset": "record_from_model",
    "from_fluxo_dataset": "model_from_record",
    "ToolInput": "tool_inputs",
    "InputCheck": "tool_inputs",
    "detect_tool": "tool_inputs",
    "read_tool_input": "tool_inputs",
    "write_tool_input": "tool_inputs",
    "check_tool_input": "tool_inputs",
    "record_tool_input": "tool_inputs",
    "AliasMatch": "aliases",
    "infer_aliases": "aliases",
    "add_aliases": "aliases",
    "alias_map": "aliases",
    "add_results": "runs",
    "result_files": "runs",
}


def __getattr__(name: str) -> Any:
    """Import the FluxoLink-dependent names on first use.

    Returns:
        The requested object.

    Raises:
        ImportError: If the optional ``fluxolink`` package is missing.
        AttributeError: If *name* is not exported.
    """
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        import fluxolink  # noqa: F401
    except ImportError as error:  # pragma: no cover - depends on env
        raise ImportError(
            f"{name} needs the optional 'fluxolink' package; install the "
            "converter with the 'linkml' extra (uv sync --extra linkml)"
        ) from error
    return getattr(import_module(f".{_LAZY[name]}", __name__), name)


__all__ = ["Difference", "EquivalenceReport", "compare", *_LAZY]
