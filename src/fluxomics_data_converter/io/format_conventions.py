"""Format conventions shared by the tool readers/writers and the link layer.

Nothing here imports :mod:`fluxolink`, so the equivalence check can use these
helpers without the optional ``linkml`` extra installed.

The helpers answer questions every converter consumer has to answer the same
way, for example "which isotopologue does this datum hold?" (FluxML stores it
in ``weight``, FreeFlux in ``pos``, MTF only in the datum id ``"G:M2"``) or
"how many atoms does this metabolite track?" (the MTF and FreeFlux parsers
leave ``Metabolite.atoms`` at its default of 0 and keep the count only in the
atom maps).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .mtf_writer import MTFWriter

if TYPE_CHECKING:
    from ..core.core import FluxomicsData, MetabolicNetworkModel
    from ..experiment.measurement import Datum

#: Canonical ``InputFormatEnum`` values understood by the adapter.
FLUXML = "FluxML"
MTF = "MTF"
FREEFLUX_TSV = "freeflux_tsv"
FREEFLUX_XLSX = "freeflux_xlsx"
FREEFLUX_CSV = "freeflux_csv"
INCA = "INCA_script"

FREEFLUX_FORMATS = frozenset({FREEFLUX_TSV, FREEFLUX_XLSX, FREEFLUX_CSV})

_FORMAT_ALIASES: Dict[str, str] = {
    "fluxml": FLUXML,
    "fml": FLUXML,
    "13cflux2": FLUXML,
    "mtf": MTF,
    "influx_si": MTF,
    "freeflux": FREEFLUX_TSV,
    "freeflux_tsv": FREEFLUX_TSV,
    "freeflux_xlsx": FREEFLUX_XLSX,
    "freeflux_csv": FREEFLUX_CSV,
    "inca": INCA,
    "inca_script": INCA,
}

#: Native tool of each format, used when no ``software`` is given.
NATIVE_SOFTWARE: Dict[str, str] = {
    FLUXML: "13CFLUX2",
    MTF: "influx_si",
    FREEFLUX_TSV: "freeflux",
    FREEFLUX_XLSX: "freeflux",
    FREEFLUX_CSV: "freeflux",
    INCA: "INCA",
}

#: MTF file extensions the converter reads (``.opt`` is read but unused).
MTF_EXTENSIONS: Tuple[str, ...] = (
    ".netw",
    ".linp",
    ".miso",
    ".mflux",
    ".mmet",
    ".cnstr",
    ".tvar",
    ".opt",
)

_MID_SUFFIX_RE = re.compile(r":M(\d+)$")
_IDENT_RE = re.compile(r"^\s*([A-Za-z_][\w.]*)(?:\[([^\]]*)\])?")
_RELATION_RE = re.compile(r"(<=|>=|==|=<|=>|=|<|>)")


def canonical_format(
    source_format: Optional[str], source_path: Optional[Path]
) -> Optional[str]:
    """Return the ``InputFormatEnum`` value for a source.

    Args:
        source_format: Format name given by the caller (case-insensitive;
            aliases such as ``"influx_si"`` or ``"freeflux"`` are accepted).
        source_path: Path of the source file, directory or MTF prefix, used
            to infer the format when ``source_format`` is not given.

    Returns:
        The canonical format, or ``None`` when it cannot be determined.

    Raises:
        ValueError: If ``source_format`` is not a known format.
    """
    if source_format:
        key = source_format.strip().lower()
        if key not in _FORMAT_ALIASES:
            raise ValueError(
                f"Unknown source_format {source_format!r}; expected one of "
                f"{sorted(set(_FORMAT_ALIASES.values()))}"
            )
        fmt = _FORMAT_ALIASES[key]
        if fmt == FREEFLUX_TSV and source_path is not None:
            return freeflux_format(source_path) or fmt
        return fmt
    if source_path is None:
        return None
    path = Path(source_path)
    suffix = path.suffix.lower()
    if suffix in (".fml", ".xml"):
        return FLUXML
    if suffix == ".m":
        return INCA
    if suffix in MTF_EXTENSIONS or path.with_suffix(".netw").exists():
        return MTF
    return freeflux_format(path)


def freeflux_format(source_path: Path) -> Optional[str]:
    """Return the FreeFlux table format of a directory or reactions file.

    Args:
        source_path: A FreeFlux directory or one of its table files.

    Returns:
        ``freeflux_tsv``, ``freeflux_xlsx`` or ``freeflux_csv`` according
        to the reactions table's extension, or ``None`` if there is none.
    """
    path = Path(source_path)
    directory = path.parent if path.is_file() else path
    for name in ("reactions", "reaction"):
        for suffix, fmt in (
            (".tsv", FREEFLUX_TSV),
            (".csv", FREEFLUX_CSV),
            (".xlsx", FREEFLUX_XLSX),
        ):
            if (directory / f"{name}{suffix}").exists():
                return fmt
    return None


def converter_defaults_are_data(fmt: Optional[str]) -> bool:
    """Tell whether converter defaults for settings reflect the source.

    The FluxML parser fills ``Simulation.type``/``method``, ``Tracers.type``
    and ``Group.scale`` from attributes whose FluxML defaults equal the
    converter defaults, so the values are source data. The MTF, FreeFlux and
    INCA parsers never read those settings; the values they carry are
    converter defaults and must not be published as data.

    Args:
        fmt: Canonical source format (``None`` when unknown).

    Returns:
        True for FluxML and for an unknown format.
    """
    return fmt in (None, FLUXML)


def effective_atom_counts(model: MetabolicNetworkModel) -> Dict[str, int]:
    """Return the number of tracked atoms per metabolite.

    ``Metabolite.atoms`` is used when positive. Otherwise the count is the
    highest atom index the metabolite reaches in any atom map, which is how
    the converter's own writers recover it for MTF and FreeFlux input.

    Args:
        model: The metabolic network model.

    Returns:
        Metabolite id to atom count, for metabolites whose count is known.
    """
    derived: Dict[str, int] = {}
    for transition in model.atom_mappings.values():
        for atom_map in transition.maps.values():
            for product, source in atom_map.mapping.items():
                for address in (product, source):
                    derived[address.mol] = max(
                        derived.get(address.mol, 0), address.index
                    )
    counts: Dict[str, int] = {}
    for metabolite in model.metabolites:
        if metabolite.atoms:
            counts[metabolite.id] = metabolite.atoms
        elif derived.get(metabolite.id):
            counts[metabolite.id] = derived[metabolite.id]
    return counts


def parse_measurement_spec(
    spec: str, data: FluxomicsData, atom_counts: Dict[str, int]
) -> Tuple[str, Optional[List[int]]]:
    """Split a labeling measurement spec into metabolite and positions.

    Uses the converter's own parser (``MTFWriter._parse_ms_expression``),
    which understands ``Ala[2,3]#M0,1,2``, ``Asp[2-4]#M0,1,2,3`` and the
    FreeFlux ``Glu[1+2+3]`` form. Specs it cannot read (metabolite ids with
    dots such as INCA ``Gluc.ext``, cumomer specs such as ``F[1-3]:010``)
    fall back to the leading identifier and bracket. A spec without a
    bracket measures the whole molecule, so its positions are
    ``1..n_atoms`` when the atom count is known.

    Args:
        spec: The textual measurement spec.
        data: The model the spec belongs to.
        atom_counts: Tracked atoms per metabolite
            (see :func:`effective_atom_counts`).

    Returns:
        The metabolite id and the 1-based atom positions (``None`` when
        they cannot be determined).
    """
    writer = MTFWriter()
    writer._model = data
    metabolite, fragment = writer._parse_ms_expression(spec)
    bracket: Optional[str] = None
    if metabolite not in data.model.metabolite_ids:
        match = _IDENT_RE.match(spec)
        metabolite = match.group(1) if match else spec.strip()
        bracket = match.group(2) if match else None
        fragment = MTFWriter._expand_fragment_spec(bracket) if bracket else ""
    elif "[" in spec.split("#")[0]:
        bracket = fragment
    positions = [int(p) for p in fragment.split(",") if p.strip().isdigit()]
    if not positions and bracket is None:
        n_atoms = atom_counts.get(metabolite)
        positions = list(range(1, n_atoms + 1)) if n_atoms else []
    return metabolite, positions or None


def datum_belongs(datum_id: str, native_id: str) -> bool:
    """Tell whether a datum belongs to a measurement group or flux.

    FluxML datums repeat the group id; the MTF and FreeFlux parsers append
    the isospecies (``"F_123:M0"``).

    Args:
        datum_id: The datum's id.
        native_id: The group, flux or pool-size id.

    Returns:
        True when the datum belongs to ``native_id``.
    """
    return datum_id == native_id or datum_id.startswith(native_id + ":")


def isotopologue_index(datum: Datum) -> Optional[int]:
    """Return the mass-isotopologue index a labeling datum holds.

    Args:
        datum: A labeling datum.

    Returns:
        The index from ``weight`` (FluxML), ``pos`` (FreeFlux) or the
        ``":M<i>"`` suffix of the id (MTF), or ``None``.
    """
    if datum.weight is not None and datum.weight.strip().isdigit():
        return int(datum.weight)
    if datum.pos is not None and datum.type is None:
        return datum.pos
    match = _MID_SUFFIX_RE.search(datum.id)
    return int(match.group(1)) if match else None


def split_relation(expression: str) -> Optional[Tuple[str, str, str]]:
    """Split a textual constraint at its relational operator.

    Args:
        expression: A textual constraint such as ``"0.488*R67=bmAla"``.

    Returns:
        ``(lhs, operator, rhs)`` with the operator normalised to one of
        ``"="``, ``"<="``, ``">="``, ``"<"``, ``">"``, or ``None`` when the
        expression has no operator.
    """
    match = _RELATION_RE.search(expression)
    if match is None:
        return None
    operator = {"==": "=", "=<": "<=", "=>": ">="}.get(
        match.group(1), match.group(1)
    )
    return (
        expression[: match.start()].strip(),
        operator,
        expression[match.end() :].strip(),
    )


def mathml_relation(expression: str) -> Optional[str]:
    """Return the relational operator of a MathML constraint.

    Args:
        expression: MathML markup.

    Returns:
        ``"="``, ``"<="``, ``">="``, ``"<"`` or ``">"``, or ``None``.
    """
    for tag, operator in (
        ("eq", "="),
        ("leq", "<="),
        ("geq", ">="),
        ("lt", "<"),
        ("gt", ">"),
    ):
        if re.search(rf"<(?:\w+:)?{tag}\s*/>", expression):
            return operator
    return None
