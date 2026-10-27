"""Fluxomics I/O operations package.

Contains parsers and writers for the fluxomics data formats:

- **FluxML** — hierarchical XML format used by 13CFlux/13CFlux2
- **MTF** — multi-file text format used by influx_si
- **Freeflux** — tabular format (tsv/csv/xlsx) used by Freeflux
- **INCA** — MATLAB model scripts (``.m``) used by INCA (read-only)
- **FluxoLink** — LinkML dataset YAML (``.yaml``); needs the optional
  ``linkml`` extra (``uv sync --extra linkml``)

:mod:`.results` reads what the tools write after a run (influx_si, INCA,
FreeFlux, 13CFLUX v3) into FluxoLink records; it also needs the extra.

Quick start
-----------
Auto-detect format from file extension and parse::

    from fluxomics_data_converter.io import parse
    model = parse("models/ecoli.fml")          # FluxML
    model = parse("models/ecoli.netw")         # MTF  (pass the .netw file)
    model = parse("models/reactions.tsv")      # Freeflux (pass reactions file)
    model = parse("models/EC.m")               # INCA
    model = parse("study/ec.yaml")             # FluxoLink dataset

Or use format-specific convenience functions::

    from fluxomics_data_converter.io import (
        parse_fluxml_file,
        parse_mtf,
        parse_freeflux,
        parse_inca,
    )
    model = parse_fluxml_file("models/ecoli.fml")
    model = parse_mtf("models/ecoli")  # base path without extension
    model = parse_freeflux("models/ecoli/")  # dir with reactions.*
    model = parse_inca("models/EC.m")  # INCA MATLAB script

Format protocols
----------------
All parsers satisfy
:class:`~fluxomics_data_converter.io.base.FluxomicsParser` and all writers
satisfy :class:`~fluxomics_data_converter.io.base.FluxomicsWriter`.  See
those protocols for the expected interface when writing format-agnostic
code.
"""

from pathlib import Path

from .base import FluxomicsParser, FluxomicsWriter
from .fluxml_parser import FluxMLParser, parse_fluxml_file
from .fluxml_writer import FluxMLWriter, write_fluxml
from .mtf_parser import MTFParser, parse_mtf
from .mtf_writer import MTFWriter, write_mtf
from .freeflux_parser import FreefluxParser, parse_freeflux
from .freeflux_writer import FreefluxWriter, write_freeflux
from .inca_parser import INCAParser, parse_inca

# FluxoLink names, imported on first use because they need the optional
# ``fluxolink`` package: name -> defining module
_FLUXOLINK = {
    "FluxoLinkParser": "fluxolink_parser",
    "parse_fluxolink": "fluxolink_parser",
    "from_fluxo_dataset": "fluxolink_parser",
    "load_yaml": "fluxolink_parser",
    "FluxoLinkWriter": "fluxolink_writer",
    "write_fluxolink": "fluxolink_writer",
    "to_fluxo_dataset": "fluxolink_writer",
    "dump_yaml": "fluxolink_writer",
}

# Extension → parser class mapping used by parse()
_PARSERS: dict[str, type] = {
    ".fml": FluxMLParser,
    ".xml": FluxMLParser,
    ".netw": MTFParser,
    ".tsv": FreefluxParser,
    ".csv": FreefluxParser,
    ".xlsx": FreefluxParser,
    ".m": INCAParser,
}
# Extension → name of a parser in _FLUXOLINK
_LAZY_PARSERS: dict[str, str] = {
    ".yaml": "FluxoLinkParser",
    ".yml": "FluxoLinkParser",
}


def __getattr__(name: str):
    """Import the FluxoLink parser and writer on first use.

    Returns:
        The requested class or function.

    Raises:
        ImportError: If the optional ``fluxolink`` package is missing.
        AttributeError: If *name* is not exported.
    """
    if name not in _FLUXOLINK:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        import fluxolink  # noqa: F401
    except ImportError as error:  # pragma: no cover - depends on env
        raise ImportError(
            f"{name} needs the optional 'fluxolink' package; install the "
            "converter with the 'linkml' extra (uv sync --extra linkml)"
        ) from error
    from importlib import import_module

    return getattr(import_module(f".{_FLUXOLINK[name]}", __name__), name)


def parse(path: str | Path):
    """Auto-detect format from file extension and parse.

    This is the single entry point for reading any supported fluxomics
    format.  The format is inferred from the file extension:

    - ``.fml`` / ``.xml``             → :class:`FluxMLParser`
    - ``.netw``                        → :class:`MTFParser`
      (pass the ``.netw`` file; all associated ``.linp``, ``.miso`` etc.
      files are resolved automatically from the same directory)
    - ``.tsv`` / ``.csv`` / ``.xlsx`` → :class:`FreefluxParser`
      (pass the ``reactions.*`` file or the directory containing it)
    - ``.m``                           → :class:`INCAParser`
      (an INCA MATLAB model script; read-only)
    - ``.yaml`` / ``.yml``             → :class:`FluxoLinkParser`
      (a FluxoLink dataset; needs the ``linkml`` extra)

    Args:
        path: Path to the model file (or base path for multi-file formats).

    Returns:
        A fully validated :class:`~fluxomics_data_converter.FluxomicsData`.

    Raises:
        ValueError: If the file extension is not recognised.

    Example:
        from fluxomics_data_converter.io import parse

        model = parse("data/ecoli.fml")
        model = parse("data/ecoli.netw")
        model = parse("data/reactions.tsv")
        model = parse("data/EC.m")
        model = parse("study/ec.yaml")
    """
    path = Path(path)
    suffix = path.suffix.lower()
    parser_cls = _PARSERS.get(suffix)
    if parser_cls is None and suffix in _LAZY_PARSERS:
        parser_cls = __getattr__(_LAZY_PARSERS[suffix])
    if parser_cls is None:
        supported = sorted([*_PARSERS, *_LAZY_PARSERS])
        raise ValueError(
            f"Unrecognised file extension '{path.suffix}'. "
            f"Supported extensions: {supported}"
        )
    return parser_cls().parse(path)


__all__ = [
    # Protocols
    "FluxomicsParser",
    "FluxomicsWriter",
    # Unified entry point
    "parse",
    # FluxML format
    "FluxMLParser",
    "parse_fluxml_file",
    "FluxMLWriter",
    "write_fluxml",
    # MTF format (influx_si)
    "MTFParser",
    "parse_mtf",
    "MTFWriter",
    "write_mtf",
    # Freeflux format (tabular)
    "FreefluxParser",
    "parse_freeflux",
    "FreefluxWriter",
    "write_freeflux",
    # INCA format (MATLAB model scripts, read-only)
    "INCAParser",
    "parse_inca",
]
