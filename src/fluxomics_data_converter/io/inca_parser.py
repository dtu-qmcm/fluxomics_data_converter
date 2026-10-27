"""INCA MATLAB model-script parser.

INCA (Isotopomer Network Compartmental Analysis) models are distributed as
MATLAB ``.m`` scripts that build the model through documented constructor
calls and then mutate it through property assignments::

    r = reaction({'G6P (abcdef) <-> F6P (abcdef)'; ...});
    t = tracer({'1-13C_Gluc: Gluc.ext @ 1'});
    d = msdata({'Ala2: Ala @ 2 3'});
    x = experiment(t);  x.data_ms = d;
    m = model(r);       m.expts = x;
    m.rates.flx.val = [ ... ];
    m.states.val    = [ ... ];

This parser replays that subset — see :mod:`._matlab` for the MATLAB layer —
and maps the result onto :class:`~fluxomics_data_converter.FluxomicsData`.

Derived identifiers
-------------------
INCA scripts contain no reaction or metabolite identifiers; INCA derives
them, and so does this parser:

- Reactions are named ``R1 … Rn`` in source order.
- ``m.rates.flx.val`` holds one entry per irreversible reaction and two
  (forward then reverse) per reversible reaction, in reaction order.
- ``m.states.val`` holds one entry per metabolite in ASCII order of the
  metabolite identifier.

The two vector lengths are asserted while parsing: the values are unlabeled,
so a wrong ordering rule would silently attach every flux and pool size to
the wrong reaction or metabolite.

Representation choices
----------------------
- Metabolite identifiers are kept verbatim; the ``.ext`` / ``.pre`` suffixes
  are an INCA naming convention, not compartments, so ``compartment`` is
  left unset.
- Forward and reverse fluxes are converted to net and exchange
  (``net = f - b``, ``xch = min(f, b)``) for comparability with the FluxML
  and MTF readers.  The raw directional values are preserved as reaction
  annotations so nothing is lost.
- Symmetric metabolites declared via ``m.mets{'X'}.sym`` are expanded at
  parse time into atom-map variants named ``<rxn>___1``, ``<rxn>___2``, …
  matching the convention used by the FluxML parser.

Not recovered
-------------
Values produced by opaque calls (``mod2stoich(m)'``, ``normrnd(...)``),
external data files read with ``readtable``, most of the 33 solver options,
and trailing benchmark harness code.  Statements the parser does not
recognise are collected in :attr:`INCAParser.skipped` rather than dropped
silently.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.common import (
    Annotation,
    AtomTransitionsNetwork,
    DictList,
    TextualOrMath,
)
from ..core.core import (
    FluxomicsData,
    LabelingExperiments,
    MetabolicNetworkModel,
    Metadata,
)
from ..experiment.measurement import (
    Datum,
    FluxMeasurement,
    Group,
    LabelingMeasurement,
    Measurement,
    MeasurementData,
    MeasurementModel,
    NetFlux,
)
from ..experiment.tracer import LabelComposition, Tracers
from ..model.atom_mapping import AtomMap, AtomTransition
from ..model.metabolite import Metabolite
from ..model.reaction import Reaction
from ..output.simulation import (
    FluxValue,
    MetaboliteSizeValue,
    Simulation,
    Variables,
)
from ._matlab import UNKNOWN, Statement, evaluate, read_statements, split_call

# Constructors that produce a record this parser tracks.
_CONSTRUCTORS = frozenset(
    {
        "reaction",
        "tracer",
        "msdata",
        "msmsdata",
        "nmrdata",
        "pooldata",
        "data",
        "experiment",
        "model",
        "idv",
    }
)

_LHS_RE = re.compile(
    r"^(?P<lhs>[A-Za-z_]\w*(?:\.\w+|\{[^{}]*\})*)\s*=(?!=)\s*(?P<rhs>.+)$",
    re.DOTALL,
)

# "0.715 R5P (abcdef)" / "CO2(f)" / "Dummy.ext"
_TERM_RE = re.compile(
    r"""^\s*
        (?:(?P<coeff>\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\*?\s+)?
        (?P<metabolite>[A-Za-z_][\w.]*)
        \s*
        (?:\(\s*(?P<atoms>[A-Za-z]+)\s*\))?
        \s*$""",
    re.VERBOSE,
)

# "1-13C_Gluc: Gluc.ext @ 1 2 3" / "Ala2: Ala @ 2 3"
_SPEC_RE = re.compile(
    r"^\s*(?P<id>[^:]+?)\s*:\s*(?P<state>\S+)\s*@\s*(?P<atoms>.*?)\s*$"
)

_ATOMMAP_RE = re.compile(r"atommap\s*\(\s*'([^']*)'\s*\)")


@dataclass
class _Term:
    """One side-term of an INCA reaction equation.

    Attributes:
        metabolite: Metabolite identifier, verbatim from the equation.
        coefficient: Stoichiometric coefficient, defaulting to 1.
        atoms: Letter-notation atom string, or ``None`` when the term
            carries no atom mapping.
    """

    metabolite: str
    coefficient: float = 1.0
    atoms: Optional[str] = None


@dataclass
class _RawReaction:
    """A parsed INCA reaction equation before model construction.

    Attributes:
        id: Derived identifier (``R1``, ``R2``, …).
        equation: Original equation text.
        reversible: Whether the equation used ``<->``.
        reactants: Left-hand side terms, in order.
        products: Right-hand side terms, in order.
        forward: Forward flux value, when supplied by the script.
        backward: Reverse flux value, when supplied by the script.
    """

    id: str
    equation: str
    reversible: bool
    reactants: List[_Term] = field(default_factory=list)
    products: List[_Term] = field(default_factory=list)
    forward: Optional[float] = None
    backward: Optional[float] = None


def _format_atoms(positions: List[int]) -> str:
    """Format 1-based atom positions as a measurement-expression suffix.

    Mirrors the convention used by the MTF parser so that downstream
    writers need no INCA-specific branch.

    Args:
        positions: 1-based atom positions in the fragment.

    Returns:
        A bracketed suffix such as ``"[1-3]"`` or ``"[2,4]"``, or an empty
        string when *positions* is empty.
    """
    if not positions:
        return ""
    ordered = sorted(positions)
    contiguous = ordered == list(range(ordered[0], ordered[-1] + 1))
    if contiguous and len(ordered) > 2:
        return f"[{ordered[0]}-{ordered[-1]}]"
    return f"[{','.join(str(p) for p in ordered)}]"


def _map_key(atom_map: AtomMap) -> Tuple[Tuple[str, int, str, int], ...]:
    """Build a hashable canonical key for an atom map.

    Used to drop duplicate symmetry variants, which arise whenever a
    declared symmetry does not actually change a given reaction's mapping.

    Args:
        atom_map: The map to key.

    Returns:
        A sorted tuple of ``(product_mol, product_index, source_mol,
        source_index)`` entries.
    """
    return tuple(
        sorted(
            (dst.mol, dst.index, src.mol, src.index)
            for dst, src in atom_map.mapping.items()
        )
    )


class INCAParser:
    """Parser for INCA MATLAB model scripts.

    The parser is single-use per file but may be reused: :meth:`parse`
    resets all internal state.  After a successful parse, :attr:`warnings`
    holds recoverable problems found in the source model and
    :attr:`skipped` holds every statement the parser did not recognise.

    Example::

        parser = INCAParser()
        model = parser.parse("EC.m")
        for message in parser.warnings:
            print(message)
    """

    def __init__(self) -> None:
        """Initialise an empty parser."""
        self.warnings: List[str] = []
        self.skipped: List[Statement] = []
        self._env: Dict[str, Any] = {}

    # -- public API --------------------------------------------------

    def parse(self, path: Path | str) -> FluxomicsData:
        """Parse an INCA ``.m`` model script.

        Args:
            path: Path to the script.

        Returns:
            A fully validated
            :class:`~fluxomics_data_converter.FluxomicsData`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the script contains no ``model(...)`` call, if a
                reaction equation is malformed, or if a value vector length
                contradicts the network (which would mean values attach to
                the wrong reactions or metabolites).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"INCA script not found: {path}")

        self.warnings = []
        self.skipped = []
        self._env = {}

        source = path.read_text(encoding="utf-8", errors="replace")
        for statement in read_statements(source):
            self._execute(statement)

        model_record = self._find_model()
        return self._build(model_record, path)

    # -- statement interpretation ------------------------------------

    def _execute(self, statement: Statement) -> None:
        """Interpret one statement, recording it if unrecognised.

        Args:
            statement: The statement to interpret.
        """
        match = _LHS_RE.match(statement.text)
        if match is None:
            self.skipped.append(statement)
            return

        parts = re.findall(r"\{[^{}]*\}|\w+", match.group("lhs"))
        if not parts:
            self.skipped.append(statement)
            return

        root, shape, selectors = parts[0], [], []
        for part in parts[1:]:
            if part.startswith("{"):
                selectors.append(part[1:-1].strip().strip("'\""))
                shape.append("{}")
            else:
                shape.append(part)
        key = ".".join(shape)

        if not key:
            value = self._rhs(match.group("rhs"))
            if value is UNKNOWN and root in self._env:
                return
            self._env[root] = value
            return

        target = self._env.get(root)
        if not isinstance(target, dict):
            self.warnings.append(
                f"line {statement.line}: assignment to undefined variable "
                f"'{root}' ({statement.text[:60]})"
            )
            return

        if not self._assign(
            target, key, selectors, match.group("rhs"), statement
        ):
            self.skipped.append(statement)

    def _assign(
        self,
        target: Dict[str, Any],
        key: str,
        selectors: List[str],
        rhs: str,
        statement: Statement,
    ) -> bool:
        """Apply a property assignment to a tracked record.

        Args:
            target: The record being mutated.
            key: Dotted property path relative to the record, with ``{}``
                standing in for a brace selector.
            selectors: Brace-selector contents, in order of appearance.
            rhs: Right-hand side source.
            statement: The originating statement, for diagnostics.

        Returns:
            ``True`` if the assignment was recognised and applied.
        """
        kind = target.get("__class__")

        if kind == "model":
            return self._assign_model(target, key, selectors, rhs, statement)

        value = self._rhs(rhs)

        if kind == "tracer" and key == "frac":
            target["frac"] = value if isinstance(value, list) else [value]
            return True
        if kind == "data" and key in {"val", "std"}:
            # normrnd(...) and friends evaluate to UNKNOWN; keep the
            # literal values the script assigned beforehand.
            if value is UNKNOWN:
                self.warnings.append(
                    f"line {statement.line}: cannot evaluate "
                    f"'{rhs.strip()[:50]}', keeping previous {key}"
                )
                return True
            target[key] = value if isinstance(value, list) else [value]
            return True
        if kind == "experiment" and key.startswith("data_"):
            target[key] = value
            return True
        if kind == "experiment" and key == "tracers":
            target["tracers"] = value
            return True
        if kind in {"msdata", "pooldata"} and key.startswith("idvs"):
            target.setdefault("idvs", value)
            return True

        return False

    def _assign_model(
        self,
        model: Dict[str, Any],
        key: str,
        selectors: List[str],
        rhs: str,
        statement: Statement,
    ) -> bool:
        """Apply an assignment whose target is the model object.

        Args:
            model: The model record.
            key: Dotted property path relative to the model.
            selectors: Brace-selector contents, in order of appearance.
            rhs: Right-hand side source.
            statement: The originating statement, for diagnostics.

        Returns:
            ``True`` if the assignment was recognised and applied.
        """
        if key == "mets.{}.sym" and selectors:
            found = _ATOMMAP_RE.search(rhs)
            if found is None:
                return False
            model["sym"][selectors[0]] = self._parse_permutation(
                found.group(1), statement
            )
            return True

        if key == "states.{}.bal" and selectors:
            model["bal"][selectors[0]] = bool(self._rhs(rhs))
            return True

        if key == "expts":
            model["expts"] = self._rhs(rhs)
            return True

        if key.startswith("options."):
            model["options"][key.split(".", 1)[1]] = self._rhs(rhs)
            return True

        if key.startswith("rates.flx.") or key.startswith("states."):
            field_name = key.rsplit(".", 1)[1]
            if field_name not in {"val", "lb", "ub"}:
                return False
            value = self._rhs(rhs)
            if value is UNKNOWN:
                self.warnings.append(
                    f"line {statement.line}: cannot evaluate "
                    f"'{rhs.strip()[:50]}', keeping previous {key}"
                )
                return True
            bucket = "flx" if key.startswith("rates.flx.") else "states"
            model[bucket][field_name] = value
            return True

        return False

    def _rhs(self, rhs: str) -> Any:
        """Evaluate a right-hand side, building records for constructors.

        Args:
            rhs: Right-hand side source.

        Returns:
            A record dictionary, a literal value, or
            :data:`~._matlab.UNKNOWN`.
        """
        text = rhs.strip()
        if text == "idv":
            return {"__class__": "idv"}

        call = split_call(text)
        if call is not None and call[0] in _CONSTRUCTORS:
            return self._construct(call[0], call[1])

        return evaluate(text, self._env)

    def _construct(self, name: str, args: str) -> Any:
        """Build a record for a recognised INCA constructor call.

        Args:
            name: Constructor name.
            args: Argument source.

        Returns:
            A record dictionary, or :data:`~._matlab.UNKNOWN` if the
            arguments fall outside the supported grammar.
        """
        if name == "reaction":
            return {"__class__": "reaction", "equations": self._strings(args)}
        if name == "tracer":
            return {
                "__class__": "tracer",
                "specs": self._strings(args),
                "frac": None,
            }
        if name in {"msdata", "msmsdata", "nmrdata"}:
            return {
                "__class__": "msdata",
                "kind": name,
                "specs": self._strings(args),
            }
        if name == "pooldata":
            return {"__class__": "pooldata", "specs": self._strings(args)}
        if name == "data":
            ids = self._strings(args)
            joined = " ".join(ids)
            return {
                "__class__": "data",
                "ids": joined.split(),
                "val": None,
                "std": None,
            }
        if name == "experiment":
            return {
                "__class__": "experiment",
                "tracers": evaluate(args, self._env) if args.strip() else None,
                "data_ms": None,
                "data_flx": None,
                "data_nmr": None,
                "data_cxn": None,
            }
        if name == "model":
            return {
                "__class__": "model",
                "rates": evaluate(args, self._env) if args.strip() else None,
                "expts": None,
                "options": {},
                "flx": {},
                "states": {},
                "sym": {},
                "bal": {},
            }
        if name == "idv":
            return {"__class__": "idv"}
        return UNKNOWN

    def _strings(self, args: str) -> List[str]:
        """Evaluate constructor arguments to a flat list of strings.

        Args:
            args: Argument source, typically a cell array or one string.

        Returns:
            The string elements, in order.
        """
        value = evaluate(args, self._env)
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str)]
        return []

    def _parse_permutation(
        self, spec: str, statement: Statement
    ) -> Dict[int, int]:
        """Parse an ``atommap`` permutation such as ``'1:4 2:3 3:2 4:1'``.

        Args:
            spec: Permutation source.
            statement: The originating statement, for diagnostics.

        Returns:
            A mapping from source atom position to permuted position.
        """
        permutation: Dict[int, int] = {}
        for pair in spec.split():
            if ":" not in pair:
                continue
            left, right = pair.split(":", 1)
            try:
                permutation[int(left)] = int(right)
            except ValueError:
                self.warnings.append(
                    f"line {statement.line}: bad atom map entry '{pair}'"
                )
        return permutation

    def _find_model(self) -> Dict[str, Any]:
        """Locate the model record produced by the script.

        Returns:
            The model record.

        Raises:
            ValueError: If the script never calls ``model(...)``.
        """
        for value in self._env.values():
            if isinstance(value, dict) and value.get("__class__") == "model":
                return value
        raise ValueError(
            "No model(...) call found; this does not look like an INCA "
            "model script"
        )

    # -- equation parsing --------------------------------------------

    def _parse_equation(self, rxn_id: str, equation: str) -> _RawReaction:
        """Parse one INCA reaction equation.

        Args:
            rxn_id: Derived reaction identifier.
            equation: Equation source.

        Returns:
            The parsed reaction.

        Raises:
            ValueError: If the equation has no arrow or a term cannot be
                parsed.
        """
        reversible = "<->" in equation
        arrow = "<->" if reversible else "->"
        if arrow not in equation:
            raise ValueError(
                f"{rxn_id}: reaction equation has no arrow: {equation!r}"
            )
        left, right = equation.split(arrow, 1)
        return _RawReaction(
            id=rxn_id,
            equation=equation.strip(),
            reversible=reversible,
            reactants=self._parse_side(rxn_id, left),
            products=self._parse_side(rxn_id, right),
        )

    def _parse_side(self, rxn_id: str, side: str) -> List[_Term]:
        """Parse one side of a reaction equation.

        Args:
            rxn_id: Reaction identifier, for error messages.
            side: Source for one side of the arrow.

        Returns:
            The terms, in order.

        Raises:
            ValueError: If a term cannot be parsed.
        """
        terms: List[_Term] = []
        for chunk in side.split("+"):
            if not chunk.strip():
                continue
            match = _TERM_RE.match(chunk)
            if match is None:
                raise ValueError(
                    f"{rxn_id}: cannot parse term {chunk.strip()!r}"
                )
            terms.append(
                _Term(
                    metabolite=match.group("metabolite"),
                    coefficient=float(match.group("coeff") or 1.0),
                    atoms=match.group("atoms"),
                )
            )
        return terms

    # -- model construction ------------------------------------------

    def _build(self, record: Dict[str, Any], path: Path) -> FluxomicsData:
        """Build the data model from the interpreted script.

        Args:
            record: The model record.
            path: Source path, used for naming and metadata.

        Returns:
            The assembled data model.

        Raises:
            ValueError: If the script has no reactions, or if a value
                vector length contradicts the network.
        """
        rates = record.get("rates")
        equations = (
            rates.get("equations", []) if isinstance(rates, dict) else []
        )
        if not equations:
            raise ValueError("INCA model contains no reactions")

        parsed = [
            self._parse_equation(f"R{index}", equation)
            for index, equation in enumerate(equations, start=1)
        ]
        self._attach_fluxes(record, parsed)

        metabolites = self._build_metabolites(parsed, record)
        atom_maps, variant_ids = self._build_atom_maps(parsed, record)
        reactions = self._build_reactions(parsed, variant_ids)

        network = MetabolicNetworkModel(
            metabolites=metabolites,
            reactions=reactions,
            atom_mappings=atom_maps,
        )
        experiment = self._build_experiment(record, parsed, metabolites, path)

        options = record.get("options", {})
        notes = [f"Imported from INCA model script: {path.name}"]
        if isinstance(options.get("sim_tunit"), str):
            notes.append(f"INCA time unit: {options['sim_tunit']}")
        if self.skipped:
            notes.append(f"{len(self.skipped)} statement(s) not interpreted")

        return FluxomicsData(
            model=network,
            metadata=Metadata(name=path.stem, comment="; ".join(notes)),
            experiments=[experiment],
        )

    def _attach_fluxes(
        self, record: Dict[str, Any], parsed: List[_RawReaction]
    ) -> None:
        """Distribute ``m.rates.flx.val`` across the parsed reactions.

        Args:
            record: The model record.
            parsed: Parsed reactions, in source order.

        Raises:
            ValueError: If the vector length does not equal the number of
                reactions plus the number of reversible reactions.
        """
        values = record.get("flx", {}).get("val")
        if not isinstance(values, list):
            return

        reversible = sum(1 for raw in parsed if raw.reversible)
        expected = len(parsed) + reversible
        if len(values) != expected:
            raise ValueError(
                f"m.rates.flx.val has {len(values)} entries but the network "
                f"needs {expected} ({len(parsed)} reactions, {reversible} "
                f"reversible). Flux values would attach to the wrong "
                f"reactions."
            )

        cursor = 0
        for raw in parsed:
            raw.forward = float(values[cursor])
            cursor += 1
            if raw.reversible:
                raw.backward = float(values[cursor])
                cursor += 1

    def _build_metabolites(
        self, parsed: List[_RawReaction], record: Dict[str, Any]
    ) -> DictList[Metabolite]:
        """Collect metabolites from the reaction equations.

        Metabolites are inserted in ASCII order of their identifier, which
        is the order INCA uses for ``m.states.val``.

        Args:
            parsed: Parsed reactions.
            record: The model record, for symmetry declarations.

        Returns:
            The metabolites, in ASCII order.

        Raises:
            ValueError: If one metabolite is given conflicting atom counts.
        """
        atom_counts: Dict[str, int] = {}
        seen: set[str] = set()
        for raw in parsed:
            for term in raw.reactants + raw.products:
                seen.add(term.metabolite)
                if term.atoms is None:
                    continue
                count = len(term.atoms)
                previous = atom_counts.get(term.metabolite)
                if previous is not None and previous != count:
                    raise ValueError(
                        f"Metabolite '{term.metabolite}' has {previous} atoms "
                        f"in one reaction and {count} in {raw.id}"
                    )
                atom_counts[term.metabolite] = count

        symmetric = set(record.get("sym", {}))
        return DictList[Metabolite](
            [
                Metabolite(
                    id=met_id,
                    atoms=atom_counts.get(met_id),
                    involved_in_variants=met_id in symmetric,
                )
                for met_id in sorted(seen)
            ]
        )

    def _build_reactions(
        self,
        parsed: List[_RawReaction],
        variant_ids: Dict[str, List[str]],
    ) -> DictList[Reaction]:
        """Build reaction objects with stoichiometry and flux annotations.

        Args:
            parsed: Parsed reactions, in source order.
            variant_ids: Atom-map variant identifiers, keyed by reaction id,
                for reactions expanded by a declared symmetry.

        Returns:
            The reactions, in source order.
        """
        reactions: List[Reaction] = []
        for raw in parsed:
            stoichiometry: Dict[str, float] = {}
            for term in raw.reactants:
                stoichiometry[term.metabolite] = (
                    stoichiometry.get(term.metabolite, 0.0) - term.coefficient
                )
            for term in raw.products:
                stoichiometry[term.metabolite] = (
                    stoichiometry.get(term.metabolite, 0.0) + term.coefficient
                )

            annotations = [
                Annotation(name="inca_equation", content=raw.equation)
            ]
            if raw.forward is not None:
                annotations.append(
                    Annotation(
                        name="inca_flux_forward", content=repr(raw.forward)
                    )
                )
            if raw.backward is not None:
                annotations.append(
                    Annotation(
                        name="inca_flux_backward", content=repr(raw.backward)
                    )
                )

            reaction = Reaction(
                id=raw.id,
                reversibility=raw.reversible,
                reactants=[term.metabolite for term in raw.reactants],
                products=[term.metabolite for term in raw.products],
                annotations=annotations,
                atom_transition_ids=variant_ids.get(raw.id),
            )
            reactions.append(reaction.with_stoichiometry(stoichiometry))
        return DictList[Reaction](reactions)

    def _build_atom_maps(
        self, parsed: List[_RawReaction], record: Dict[str, Any]
    ) -> Tuple[AtomTransitionsNetwork, Dict[str, List[str]]]:
        """Build atom transitions, expanding declared symmetries.

        Args:
            parsed: Parsed reactions.
            record: The model record, for symmetry declarations.

        Returns:
            A tuple of the atom transitions keyed by reaction identifier and
            the variant identifiers for reactions a symmetry expanded.
        """
        symmetries: Dict[str, Dict[int, int]] = record.get("sym", {})
        transitions = AtomTransitionsNetwork()
        variant_ids: Dict[str, List[str]] = {}

        for raw in parsed:
            reactant_atoms = [
                (term.metabolite, term.atoms)
                for term in raw.reactants
                if term.atoms
            ]
            product_atoms = [
                (term.metabolite, term.atoms)
                for term in raw.products
                if term.atoms
            ]
            if not reactant_atoms or not product_atoms:
                continue

            try:
                base = AtomTransition.parse_letter_notation(
                    reactant_atoms, product_atoms
                )
            except ValueError as error:
                raise ValueError(f"{raw.id}: {error}") from error

            involved = {
                met
                for met, _ in reactant_atoms + product_atoms
                if met in symmetries
            }
            variants = self._expand_symmetry(base, involved, symmetries)

            reactants = [term.metabolite for term in raw.reactants]
            products = [term.metabolite for term in raw.products]

            if len(variants) == 1:
                maps = {raw.id: variants[0]}
                weights = {raw.id: 1.0}
            else:
                ids = [
                    f"{raw.id}___{number}"
                    for number in range(1, len(variants) + 1)
                ]
                maps = dict(zip(ids, variants))
                weights = {key: 1.0 / len(variants) for key in ids}
                variant_ids[raw.id] = ids

            transitions[raw.id] = AtomTransition(
                reaction_id=raw.id,
                reactants=reactants,
                products=products,
                maps=maps,
                weights=weights,
            )

        return transitions, variant_ids

    def _expand_symmetry(
        self,
        base: AtomMap,
        involved: set[str],
        symmetries: Dict[str, Dict[int, int]],
    ) -> List[AtomMap]:
        """Expand an atom map into one variant per symmetry combination.

        Args:
            base: The map read from the equation.
            involved: Symmetric metabolites appearing in this reaction.
            symmetries: Declared permutations, keyed by metabolite.

        Returns:
            Distinct atom maps, always including *base* first.
        """
        variants = [base]
        for metabolite in sorted(involved):
            permutation = symmetries[metabolite]
            if not permutation:
                continue
            variants = variants + [
                self._permute(variant, metabolite, permutation)
                for variant in variants
            ]

        unique: List[AtomMap] = []
        seen: set[Tuple[Tuple[str, int, str, int], ...]] = set()
        for variant in variants:
            key = _map_key(variant)
            if key in seen:
                continue
            seen.add(key)
            unique.append(variant)
        return unique

    @staticmethod
    def _permute(
        atom_map: AtomMap, metabolite: str, permutation: Dict[int, int]
    ) -> AtomMap:
        """Apply a symmetry permutation to one metabolite's atom positions.

        Args:
            atom_map: The map to transform.
            metabolite: The symmetric metabolite.
            permutation: Mapping from atom position to permuted position.

        Returns:
            A new atom map with the metabolite's positions permuted on both
            the product and source sides.
        """
        result = AtomMap()
        for destination, source in atom_map.mapping.items():
            dst_index = destination.index
            if destination.mol == metabolite:
                dst_index = permutation.get(dst_index, dst_index)
            src_index = source.index
            if source.mol == metabolite:
                src_index = permutation.get(src_index, src_index)
            result = result.add(
                destination.mol,
                dst_index,
                source.mol,
                src_index,
                destination.element,
                destination.instance,
                source.instance,
            )
        return result

    # -- experiment construction -------------------------------------

    def _build_experiment(
        self,
        record: Dict[str, Any],
        parsed: List[_RawReaction],
        metabolites: DictList[Metabolite],
        path: Path,
    ) -> LabelingExperiments:
        """Build the labeling experiment from tracers, data and options.

        Args:
            record: The model record.
            parsed: Parsed reactions, carrying flux values.
            metabolites: The network metabolites, in ASCII order.
            path: Source path, used for the experiment name.

        Returns:
            The assembled experiment.
        """
        experiment = record.get("expts")
        experiment = experiment if isinstance(experiment, dict) else {}
        options = record.get("options", {})

        time_points = options.get("int_tspan")
        time_points = time_points if isinstance(time_points, list) else []
        stationary = bool(options.get("sim_ss", True)) or not time_points

        tracers, tracer_names = self._build_tracers(experiment, metabolites)
        groups = self._build_groups(experiment, metabolites, time_points)
        net_fluxes, data = self._build_flux_data(experiment, parsed)

        measurement_model = MeasurementModel(
            labeling_measurement=(
                LabelingMeasurement(groups=groups) if groups else None
            ),
            flux_measurement=(
                FluxMeasurement(net_fluxes=net_fluxes) if net_fluxes else None
            ),
        )
        measurement = Measurement(
            model=measurement_model,
            data=MeasurementData(data=data),
        )

        simulation = Simulation(
            variables=Variables(
                flux_values=self._build_flux_values(record, parsed),
                metabolitesize_values=self._build_pool_values(
                    record, metabolites
                ),
            )
        )

        comment = None
        if tracer_names:
            comment = "INCA tracers: " + ", ".join(tracer_names)

        return LabelingExperiments(
            name=str(experiment.get("id") or path.stem),
            stationary=stationary,
            time=float(time_points[-1]) if time_points else None,
            comment=comment,
            tracers=tracers,
            measurement=measurement,
            simulation=simulation,
        )

    def _build_tracers(
        self, experiment: Dict[str, Any], metabolites: DictList[Metabolite]
    ) -> Tuple[List[Tracers], List[str]]:
        """Build tracer specifications, grouped by labeled metabolite.

        INCA gives each tracer its own fractional enrichment via ``t.frac``;
        several tracers may share one substrate pool.  Those become one
        :class:`Tracers` with several
        :class:`~fluxomics_data_converter.experiment.tracer.LabelComposition`
        entries.  A remainder below 1.0 is left implicit and treated
        downstream as natural abundance, matching the MTF reader.

        Args:
            experiment: The experiment record.
            metabolites: The network metabolites.

        Returns:
            A tuple of the tracer specifications and the original INCA
            tracer names, in source order.
        """
        record = experiment.get("tracers")
        if not isinstance(record, dict):
            return [], []

        specs = record.get("specs", [])
        fractions = record.get("frac") or []
        grouped: Dict[str, List[LabelComposition]] = {}
        names: List[str] = []

        for index, spec in enumerate(specs):
            match = _SPEC_RE.match(spec)
            if match is None:
                self.warnings.append(f"Cannot parse tracer spec {spec!r}")
                continue
            state = match.group("state")
            names.append(match.group("id").strip())

            if state not in metabolites:
                self.warnings.append(
                    f"Tracer '{match.group('id').strip()}' labels "
                    f"'{state}', which appears in no reaction"
                )
                continue

            atoms = metabolites[state].atoms
            if not atoms:
                self.warnings.append(
                    f"Tracer '{match.group('id').strip()}' labels '{state}', "
                    f"which has no atom count"
                )
                continue

            positions = {
                int(token)
                for token in re.findall(r"\d+", match.group("atoms") or "")
            }
            pattern = "".join(
                "1" if position in positions else "0"
                for position in range(1, atoms + 1)
            )
            fraction = (
                float(fractions[index]) if index < len(fractions) else 1.0
            )
            grouped.setdefault(state, []).append(
                LabelComposition(labeled_pattern=pattern, fraction=fraction)
            )

        tracers: List[Tracers] = []
        for state, labels in grouped.items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                tracers.append(
                    Tracers(id=state, metabolite=state, labels=labels)
                )
            for entry in caught:
                self.warnings.append(f"Tracer '{state}': {entry.message}")
        return tracers, names

    def _build_groups(
        self,
        experiment: Dict[str, Any],
        metabolites: DictList[Metabolite],
        time_points: List[float],
    ) -> List[Group]:
        """Build MS measurement groups from ``msdata`` fragment definitions.

        Args:
            experiment: The experiment record.
            metabolites: The network metabolites.
            time_points: Simulation time points, attached to each group for
                non-stationary models.

        Returns:
            The measurement groups, in source order.
        """
        record = experiment.get("data_ms")
        if not isinstance(record, dict):
            return []

        times = (
            ",".join(f"{point:g}" for point in time_points)
            if time_points
            else None
        )
        groups: List[Group] = []
        seen: set[str] = set()

        for spec in record.get("specs", []):
            match = _SPEC_RE.match(spec)
            if match is None:
                self.warnings.append(f"Cannot parse MS spec {spec!r}")
                continue
            group_id = match.group("id").strip()
            state = match.group("state")

            if group_id in seen:
                self.warnings.append(
                    f"Duplicate MS fragment id '{group_id}', keeping first"
                )
                continue
            seen.add(group_id)

            if state not in metabolites:
                self.warnings.append(
                    f"MS fragment '{group_id}' measures '{state}', which "
                    f"appears in no reaction"
                )
                continue

            positions = [
                int(token)
                for token in re.findall(r"\d+", match.group("atoms") or "")
            ]
            expression = f"{state}{_format_atoms(positions)}"
            groups.append(
                Group(
                    id=group_id,
                    times=times,
                    scale="auto",
                    expression=TextualOrMath(textual=expression),
                )
            )
        return groups

    def _build_flux_data(
        self, experiment: Dict[str, Any], parsed: List[_RawReaction]
    ) -> Tuple[List[NetFlux], List[Datum]]:
        """Build flux measurements from a ``data(...)`` record.

        Args:
            experiment: The experiment record.
            parsed: Parsed reactions, used to validate the referenced ids.

        Returns:
            A tuple of net-flux measurement definitions and their data.
        """
        record = experiment.get("data_flx")
        if not isinstance(record, dict):
            return [], []

        ids = record.get("ids", [])
        values = record.get("val") or []
        deviations = record.get("std") or []
        known = {raw.id for raw in parsed}

        net_fluxes: List[NetFlux] = []
        data: List[Datum] = []
        for index, flux_id in enumerate(ids):
            if flux_id not in known:
                self.warnings.append(
                    f"Flux measurement references unknown reaction '{flux_id}'"
                )
                continue
            net_fluxes.append(
                NetFlux(
                    id=flux_id,
                    expression=TextualOrMath(textual=flux_id),
                )
            )
            if index < len(values):
                stddev = (
                    float(deviations[index])
                    if index < len(deviations)
                    else abs(float(values[index])) * 0.05
                )
                data.append(
                    Datum(
                        id=flux_id,
                        value=float(values[index]),
                        stddev=stddev,
                    )
                )
        return net_fluxes, data

    def _build_flux_values(
        self, record: Dict[str, Any], parsed: List[_RawReaction]
    ) -> List[FluxValue]:
        """Convert directional fluxes to net and exchange flux values.

        INCA stores a forward and, for reversible reactions, a reverse flux.
        These are converted with ``net = f - b`` and ``xch = min(f, b)``.
        Directional bounds from ``m.rates.flx.lb`` / ``.ub`` are propagated:
        an irreversible net flux inherits them directly, a reversible net
        flux spans the difference of the two directional ranges, and an
        exchange flux inherits them as-is.

        Args:
            record: The model record, for the directional bounds.
            parsed: Parsed reactions carrying the directional values.

        Returns:
            Net and exchange flux values, in reaction order.
        """
        bounds = record.get("flx", {})
        lower = bounds.get("lb")
        upper = bounds.get("ub")
        lower = float(lower) if isinstance(lower, (int, float)) else None
        upper = float(upper) if isinstance(upper, (int, float)) else None
        span = (
            (upper - lower) if lower is not None and upper is not None else None
        )

        values: List[FluxValue] = []
        for raw in parsed:
            if raw.forward is None:
                continue
            backward = raw.backward or 0.0
            if raw.reversible:
                net_lo, net_hi = (None, None) if span is None else (-span, span)
            else:
                net_lo, net_hi = lower, upper
            values.append(
                FluxValue(
                    flux=raw.id,
                    type="net",
                    value=raw.forward - backward,
                    lo=net_lo,
                    hi=net_hi,
                )
            )
            if raw.reversible:
                values.append(
                    FluxValue(
                        flux=raw.id,
                        type="xch",
                        value=min(raw.forward, backward),
                        lo=lower,
                        hi=upper,
                    )
                )
        return values

    def _build_pool_values(
        self, record: Dict[str, Any], metabolites: DictList[Metabolite]
    ) -> List[MetaboliteSizeValue]:
        """Map ``m.states.val`` onto metabolite pool sizes.

        Args:
            record: The model record.
            metabolites: The network metabolites, in ASCII order — the order
                INCA uses for the state vector.

        Returns:
            Pool-size values, one per metabolite.

        Raises:
            ValueError: If the vector length does not match the number of
                metabolites.
        """
        states = record.get("states", {})
        values = states.get("val")
        if not isinstance(values, list):
            return []

        if len(values) != len(metabolites):
            raise ValueError(
                f"m.states.val has {len(values)} entries but the network has "
                f"{len(metabolites)} metabolites. Pool sizes would attach to "
                f"the wrong metabolites."
            )

        lower = states.get("lb")
        upper = states.get("ub")
        lower = float(lower) if isinstance(lower, (int, float)) else None
        upper = float(upper) if isinstance(upper, (int, float)) else None

        return [
            MetaboliteSizeValue(
                metabolite=metabolite.id,
                value=float(value),
                lo=lower,
                hi=upper,
                type="F",
            )
            for metabolite, value in zip(metabolites, values)
        ]


def parse_inca(path: Path | str) -> FluxomicsData:
    """Parse an INCA MATLAB model script.

    Args:
        path: Path to the ``.m`` script.

    Returns:
        A fully validated
        :class:`~fluxomics_data_converter.FluxomicsData`.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the script is not an INCA model script or its value
            vectors contradict the network.

    Example::

        from fluxomics_data_converter.io import parse_inca

        model = parse_inca("data/benchmark/INCA/EC.m")
    """
    return INCAParser().parse(path)
