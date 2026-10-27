"""Base protocols for fluxomics format parsers and writers.

All format-specific parsers and writers in this package are expected to
conform to :class:`FluxomicsParser` and :class:`FluxomicsWriter`.  Using
``Protocol`` rather than an abstract base class means existing classes do
not need to inherit from anything — they satisfy the protocol as long as
their method signatures match.

Adding a new format
-------------------
1. Create ``<format>_parser.py`` with a class whose ``parse`` method
   matches :meth:`FluxomicsParser.parse`.
2. Create ``<format>_writer.py`` with a class whose ``write`` method
   matches :meth:`FluxomicsWriter.write`.
3. Register the file-extension mapping in
   :func:`fluxomics_data_converter.io.parse`.
4. Export the new class and convenience function from ``io/__init__.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..core.core import FluxomicsData


@runtime_checkable
class FluxomicsParser(Protocol):
    """Protocol satisfied by every format-specific parser.

    A parser converts one file (or a directory/base-path containing
    several related files) into a validated
    :class:`~fluxomics_data_converter.FluxomicsData`.

    Implementors
    ------------
    - :class:`~fluxomics_data_converter.io.FluxMLParser`
      (``parse_file`` alias kept for backward compatibility)
    - :class:`~fluxomics_data_converter.io.MTFParser`
    - :class:`~fluxomics_data_converter.io.FreefluxParser`
    - :class:`~fluxomics_data_converter.io.INCAParser`
    - :class:`~fluxomics_data_converter.io.FluxoLinkParser`
    """

    def parse(self, path: Path | str) -> FluxomicsData:
        """Parse a file or directory and return the data model.

        Args:
            path: Path to the file to parse, or to a base path / directory
                used by multi-file formats (MTF, Freeflux).

        Returns:
            A fully validated :class:`~fluxomics_data_converter.FluxomicsData`.

        Raises:
            FileNotFoundError: If a required file is not present at *path*.
            ValueError: If the file content is structurally invalid or
                internally inconsistent (e.g. a reaction references an
                unknown metabolite).
        """
        ...


@runtime_checkable
class FluxomicsWriter(Protocol):
    """Protocol satisfied by every format-specific writer.

    A writer serialises a :class:`~fluxomics_data_converter.FluxomicsData`
    to a particular file format.

    Implementors
    ------------
    - :class:`~fluxomics_data_converter.io.FluxMLWriter`
    - :class:`~fluxomics_data_converter.io.MTFWriter`
    - :class:`~fluxomics_data_converter.io.FreefluxWriter`
    - :class:`~fluxomics_data_converter.io.FluxoLinkWriter`

    Notes on the ``experiment_name`` parameter
    ------------------------------------------
    Some formats (MTF, Freeflux) write a single experiment per file set and
    therefore require the caller to nominate which experiment to use when
    the model contains more than one.  The FluxML format writes all
    experiments in one file and ignores this parameter.  Writers that do not
    need it should accept (and silently ignore) ``experiment_name=None``.
    """

    def write(
        self,
        model: FluxomicsData,
        path: Path | str,
        experiment_name: str | None = None,
    ) -> None:
        """Serialise *model* to the target format at *path*.

        Args:
            model: The data model to serialise.
            path: Destination file path (for single-file formats like
                FluxML) or base path / output directory (for multi-file
                formats like MTF and Freeflux).
            experiment_name: Name of the experiment to write.  Required
                when the model contains more than one experiment and the
                target format supports only a single experiment per file
                set.  Ignored by formats that write all experiments (e.g.
                FluxML).

        Raises:
            ValueError: If *experiment_name* is needed but not provided,
                or if *model* contains data that cannot be represented in
                this format.
        """
        ...
