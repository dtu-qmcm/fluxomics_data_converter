"""Fluxomics I/O operations package.

Contains parsers and writers for the input formats of the 13C-MFA tools:

- **FluxML** — hierarchical XML format used by 13CFlux/13CFlux2
- **MTF** — multi-file text format used by influx_si
- **Freeflux** — tabular format (tsv/csv/xlsx) used by Freeflux
- **INCA** — MATLAB model scripts (``.m``) used by INCA (read-only)

Every format translates through
:class:`~fluxomics_data_converter.FluxomicsData`, so any parser can feed
any writer directly. FluxoLink records are not a format here: they link
these files together and live in :mod:`fluxomics_data_converter.link`.

Quick start
-----------
Auto-detect format from file extension and parse::

    from fluxomics_data_converter.io import parse
    model = parse("models/ecoli.fml")          # FluxML
    model = parse("models/ecoli.netw")         # MTF  (pass the .netw file)
    model = parse("models/reactions.tsv")      # Freeflux (pass reactions file)
    model = parse("models/EC.m")               # INCA

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
    """
    path = Path(path)
    suffix = path.suffix.lower()
    parser_cls = _PARSERS.get(suffix)
    if parser_cls is None:
        supported = sorted(_PARSERS)
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
