"""Results readers: 13C-MFA tool output -> FluxoLink ``SimulationRun``.

Each reader turns what one tool emits after a run into a ``fluxolink``
:class:`SimulationRun` (status ``completed``) with its ``FluxSolution`` and
``GoodnessOfFit``:

- :func:`read_influx_si_results` -- an influx_si ``<prefix>_res/``
  directory (``.tvar.sim``, ``.miso.sim``, ``.mflux.sim``, ``.mmet.sim``,
  ``.stat``, ``.log``, ``tmp/.pres.csv``, optional ``.kvh``);
- :func:`read_inca_results` -- an INCA results ``.mat`` file and, optionally,
  its Monte Carlo companion;
- :func:`from_freeflux_result` -- a FreeFlux ``FitResults`` /
  ``InstFitResults`` object (FreeFlux writes no result files);
- :func:`from_x3cflux_fit` -- a 13CFLUX v3 (``x3cflux``) simulator and an
  optimum (x3cflux writes no result files either).

The FreeFlux and x3cflux readers are duck-typed: neither package is
imported. :func:`wrap_in_dataset` places a run in a minimal catalog-tier
``FluxoDataset`` and :func:`dump_dataset` writes it for
``fluxolink-validate``.

These functions need the optional ``linkml`` extra (``fluxolink``).
"""

from __future__ import annotations

from ._common import dump_dataset, wrap_in_dataset
from .freeflux import from_freeflux_result
from .inca import read_inca_results
from .influx_si import parse_influx_log, read_influx_si_results, read_miso_sim
from .x3cflux import from_x3cflux_fit

__all__ = [
    "dump_dataset",
    "from_freeflux_result",
    "from_x3cflux_fit",
    "parse_influx_log",
    "read_inca_results",
    "read_influx_si_results",
    "read_miso_sim",
    "wrap_in_dataset",
]
