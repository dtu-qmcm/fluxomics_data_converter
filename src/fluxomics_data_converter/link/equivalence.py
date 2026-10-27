"""Semantic equivalence of two converter models.

:func:`compare` decides whether two
:class:`~fluxomics_data_converter.FluxomicsData` objects describe the same
13C-MFA problem, independent of how a format happened to write it down::

    report = compare(parse_fluxml_file("EC.fml"), round_tripped)
    assert report.equivalent, report.summary()

What is compared, after normalisation
-------------------------------------
``metabolite`` / ``tracked_atoms``
    Metabolite ids; tracked-atom counts (``Metabolite.atoms``, else the
    highest index in the atom maps, else 0).
``reaction`` / ``reversibility`` / ``stoichiometry``
    Reaction ids; reversibility; net coefficient per metabolite (from
    ``stoichiometry_dict``, else -1/+1 per occurrence).
``atom_transition``
    Per reaction, the set of variant mappings, each a set of
    ``(reactant, instance, index, product, instance, index, element)``
    tuples, with the summed variant weight (absent weights read as
    uniform). Variant names and duplicate variants do not matter.
``tracer``
    Per experiment and input metabolite: labeling pattern -> fraction and
    purity.
``constraint``
    Per experiment, the effective constraint set (network-level plus
    experiment-level), each written in a canonical form: ``a <= b`` becomes
    ``b - a >= 0``, sides are expanded and an equality's sign is fixed.
``mid_measurement`` / ``nmr_measurement`` / ``flux_measurement`` /
``pool_size_measurement``
    Measured values and standard deviations, matched by (metabolite, atom
    positions, isotopologue, time) for MIDs, (metabolite, position,
    multiplet, time) for NMR and (canonical expression, flux type, time)
    for fluxes and pool sizes. Values must agree within ``rtol``/``atol``.
``experiment``
    Experiments are paired by name; when exactly one experiment on each
    side is left unpaired, those two are paired anyway.

Not compared: the free/dependent variable partition and starting values (a
choice of basis), group scaling and error models, names, annotations and
metadata.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Tuple,
)

import sympy as sp

from ..core.core import FluxomicsData, LabelingExperiments
from ..io import format_conventions as conv

KINDS: Tuple[str, ...] = (
    "metabolite",
    "tracked_atoms",
    "reaction",
    "reversibility",
    "stoichiometry",
    "atom_transition",
    "experiment",
    "tracer",
    "constraint",
    "mid_measurement",
    "nmr_measurement",
    "flux_measurement",
    "pool_size_measurement",
)

_IDENT_RE = re.compile(r"(?<![\w.])[A-Za-z_][\w.]*")

Transition = Tuple[str, int, int, str, int, int, str]
Point = Tuple[Optional[float], float, float]


@dataclass(frozen=True)
class Difference:
    """One way in which the two models differ.

    Attributes:
        kind: Category, one of :data:`KINDS`.
        key: What differs (a metabolite or reaction id, or
            ``"<experiment>/<item>"``).
        message: Human-readable explanation.
        a: The value in the first model, when there is one.
        b: The value in the second model, when there is one.
    """

    kind: str
    key: str
    message: str
    a: Any = None
    b: Any = None

    def __str__(self) -> str:
        """Return ``"[kind] key: message"``."""
        return f"[{self.kind}] {self.key}: {self.message}"


@dataclass
class EquivalenceReport:
    """Result of :func:`compare`.

    Attributes:
        differences: Every difference found, in comparison order.
    """

    differences: List[Difference] = field(default_factory=list)

    @property
    def equivalent(self) -> bool:
        """True when no difference was found."""
        return not self.differences

    def counts(self) -> Dict[str, int]:
        """Count the differences per kind.

        Returns:
            Kind to number of differences, for kinds that occur.
        """
        return dict(Counter(d.kind for d in self.differences))

    def of_kind(self, kind: str) -> List[Difference]:
        """Return the differences of one kind.

        Returns:
            The matching differences.
        """
        return [d for d in self.differences if d.kind == kind]

    def summary(self, limit: int = 10) -> str:
        """Describe the report in a few lines.

        Args:
            limit: Number of differences listed per kind.

        Returns:
            A multi-line summary.
        """
        if self.equivalent:
            return "equivalent"
        lines = [f"{len(self.differences)} difference(s)"]
        for kind, count in self.counts().items():
            lines.append(f"  {kind}: {count}")
            for diff in self.of_kind(kind)[:limit]:
                lines.append(f"    - {diff.key}: {diff.message}")
            if count > limit:
                lines.append(f"    ... and {count - limit} more")
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return :meth:`summary`."""
        return self.summary()


def compare(
    a: FluxomicsData,
    b: FluxomicsData,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    id_map: Optional[Dict[str, str]] = None,
    ignore: Iterable[str] = (),
) -> EquivalenceReport:
    """Compare two converter models after normalisation.

    Args:
        a: First model.
        b: Second model.
        rtol: Relative tolerance for numeric values.
        atol: Absolute tolerance for numeric values.
        id_map: Renames applied to identifiers on both sides before
            comparing (metabolite, reaction and experiment ids, and the
            identifiers inside constraint and measurement expressions), to
            align tool aliases such as ``{"Gluc.ext": "Gluc_ext"}``.
        ignore: Difference kinds (see :data:`KINDS`) to skip.

    Returns:
        The equivalence report.

    Raises:
        ValueError: If ``ignore`` names an unknown kind.
    """
    skipped = set(ignore)
    unknown = skipped - set(KINDS)
    if unknown:
        raise ValueError(f"Unknown difference kind(s) {sorted(unknown)}")
    rename = _renamer(id_map or {})
    differ = _Differ(rtol=rtol, atol=atol, skipped=skipped)
    view_a = _View(a, rename)
    view_b = _View(b, rename)
    differ.network(view_a, view_b)
    differ.experiments(view_a, view_b)
    return EquivalenceReport(differences=differ.differences)


def _renamer(id_map: Dict[str, str]) -> Callable[[str], str]:
    """Build the identifier renaming function.

    Returns:
        A function mapping an identifier to its canonical name.
    """
    return lambda name: id_map.get(name, name)


# -- normalised views ----------------------------------------------------------


class _View:
    """Normalised, renamed view of one model."""

    def __init__(self, data: FluxomicsData, rename: Callable[[str], str]):
        """Build every normalised table the comparison needs."""
        self.data = data
        self.rename = rename
        model = data.model
        self.atom_counts = conv.effective_atom_counts(model)
        self.metabolites = {rename(m.id) for m in model.metabolites}
        self.atoms = {
            rename(m.id): self.atom_counts.get(m.id, 0)
            for m in model.metabolites
        }
        self.reactions: Dict[str, Any] = {}
        for reaction in model.reactions:
            self.reactions[rename(reaction.id)] = reaction
        self.transitions = {
            rename(rid): _variant_weights(at, rename)
            for rid, at in model.atom_mappings.items()
            if at.maps
        }

    def stoichiometry(self, reaction) -> Dict[str, float]:
        """Return the net coefficient per (renamed) metabolite.

        Returns:
            Metabolite id to signed coefficient, zeros removed.
        """
        net: Dict[str, float] = defaultdict(float)
        if reaction.stoichiometry_dict:
            for met, coeff in reaction.stoichiometry_dict.items():
                net[self.rename(met)] += coeff
        else:
            for met in reaction.reactants:
                net[self.rename(met)] -= 1.0
            for met in reaction.products:
                net[self.rename(met)] += 1.0
        return {m: c for m, c in net.items() if c != 0.0}

    def experiments(self) -> Dict[str, LabelingExperiments]:
        """Return the experiments keyed by renamed name.

        Returns:
            Name to experiment.
        """
        return {self.rename(e.name): e for e in self.data.experiments}

    def constraints(self, exp: Optional[LabelingExperiments]) -> FrozenSet[str]:
        """Return the effective canonical constraint set of an experiment.

        Args:
            exp: The experiment, or ``None`` for the network level alone.

        Returns:
            Canonical constraint strings.
        """
        blocks = [self.data.constraints]
        if exp is not None:
            blocks.append(exp.constraints)
        result = set()
        for constraints in blocks:
            if constraints is None:
                continue
            for kind, block in (
                ("net", constraints.net),
                ("xch", constraints.xch),
                ("psize", constraints.metabolitesize),
            ):
                for formula in block.formulas if block else []:
                    result.add(
                        f"{kind}:"
                        + canonical_constraint(
                            formula.expression, formula.is_mathml, self.rename
                        )
                    )
        return frozenset(result)

    def tracers(
        self, exp: LabelingExperiments
    ) -> Dict[str, List[Tuple[str, Optional[float], Optional[float]]]]:
        """Return the tracer composition per input metabolite.

        Returns:
            Metabolite to sorted ``(pattern, fraction, purity)`` triples.
        """
        result: Dict[str, List[Tuple[str, Any, Any]]] = defaultdict(list)
        for tracer in exp.tracers:
            entries = result[self.rename(tracer.metabolite)]
            for label in tracer.labels:
                entries.append(
                    (label.labeled_pattern, label.fraction, label.purity)
                )
        return {
            k: sorted(v, key=lambda e: (e[0], e[1] or 0.0, e[2] or 0.0))
            for k, v in result.items()
        }

    def measurements(
        self, exp: LabelingExperiments
    ) -> Dict[str, Dict[Tuple[Any, ...], List[Point]]]:
        """Collect measured values per measurement kind.

        Returns:
            Kind to (match key without time) to ``(time, value, sd)``.
        """
        tables: Dict[str, Dict[Tuple[Any, ...], List[Point]]] = {
            kind: defaultdict(list)
            for kind in (
                "mid_measurement",
                "nmr_measurement",
                "flux_measurement",
                "pool_size_measurement",
            )
        }
        if exp.measurement is None:
            return tables
        model = exp.measurement.model
        data = exp.measurement.data.data
        if model.labeling_measurement is not None:
            for group in model.labeling_measurement.groups:
                spec = group.expression.textual or group.expression.mathml
                metabolite, positions = conv.parse_measurement_spec(
                    spec or group.id, self.data, self.atom_counts
                )
                metabolite = self.rename(metabolite)
                position_key = tuple(positions or ())
                for datum in data:
                    if not conv.datum_belongs(datum.id, group.id):
                        continue
                    point = (datum.time, datum.value, datum.stddev)
                    if datum.type is not None:
                        key = (metabolite, datum.pos, datum.type)
                        tables["nmr_measurement"][key].append(point)
                        continue
                    index = conv.isotopologue_index(datum)
                    key = (metabolite, position_key, index)
                    tables["mid_measurement"][key].append(point)
        if model.flux_measurement is not None:
            for flux_type, items in (
                ("net", model.flux_measurement.net_fluxes),
                ("xch", model.flux_measurement.xch_fluxes),
            ):
                for item in items:
                    expr = canonical_expression(
                        item.expression.textual or item.expression.mathml,
                        self.rename,
                    )
                    for datum in data:
                        if datum.id == item.id:
                            tables["flux_measurement"][
                                (expr, flux_type)
                            ].append((datum.time, datum.value, datum.stddev))
        if model.metabolitesize_measurement is not None:
            for item in model.metabolitesize_measurement.metabolite_sizes:
                expr = canonical_expression(
                    item.expression.textual or item.expression.mathml,
                    self.rename,
                )
                for datum in data:
                    if datum.id == item.id:
                        tables["pool_size_measurement"][(expr,)].append(
                            (datum.time, datum.value, datum.stddev)
                        )
        return tables


def _variant_weights(transition, rename) -> Dict[FrozenSet[Transition], float]:
    """Normalise an atom transition to mapping set -> summed weight.

    Returns:
        Each distinct variant mapping with its total weight.
    """
    keys = list(transition.maps)
    weights = transition.weights or {}
    if not all(k in weights for k in keys):
        weights = {k: 1.0 / len(keys) for k in keys}
    total = sum(weights[k] for k in keys) or 1.0
    result: Dict[FrozenSet[Transition], float] = defaultdict(float)
    for key, atom_map in transition.maps.items():
        mapping = frozenset(
            (
                rename(src.mol),
                src.instance,
                src.index,
                rename(dst.mol),
                dst.instance,
                dst.index,
                dst.element or src.element or "C",
            )
            for dst, src in atom_map.mapping.items()
        )
        result[mapping] += weights[key] / total
    return dict(result)


# -- canonical expressions -----------------------------------------------------


def canonical_expression(text: Optional[str], rename) -> str:
    """Write a linear expression in a canonical form.

    Args:
        text: Textual expression such as ``"R63-R63b"`` (MathML is
            compared with whitespace removed).
        rename: Identifier renaming function.

    Returns:
        The canonical string.
    """
    if not text:
        return ""
    if text.lstrip().startswith("<"):
        return "mathml:" + re.sub(r"\s+", "", text)
    parsed = _sympify(text, rename)
    if parsed is None:
        return re.sub(r"\s+", "", _rename_tokens(text, rename))
    return sp.sstr(sp.expand(parsed))


def canonical_constraint(text: str, is_mathml: bool, rename) -> str:
    """Write a constraint in a canonical form.

    ``a <= b`` and ``b >= a`` both become ``">=:b - a"``; an equality is
    written with the sign that sorts first, so ``x = y`` and ``y = x``
    agree.

    Args:
        text: The constraint expression.
        is_mathml: Whether ``text`` is MathML.
        rename: Identifier renaming function.

    Returns:
        The canonical string.
    """
    if is_mathml:
        return "mathml:" + re.sub(r"\s+", "", text)
    parts = conv.split_relation(text)
    if parts is None:
        return re.sub(r"\s+", "", _rename_tokens(text, rename))
    lhs, operator, rhs = parts
    left = _sympify(lhs, rename)
    right = _sympify(rhs, rename)
    if left is None or right is None:
        return re.sub(r"\s+", "", _rename_tokens(text, rename))
    if operator in ("<=", "<"):
        difference = sp.expand(right - left)
        operator = ">=" if operator == "<=" else ">"
    else:
        difference = sp.expand(left - right)
    written = sp.sstr(difference)
    if operator == "=":
        written = min(written, sp.sstr(sp.expand(-difference)))
    return f"{operator}:{written}"


def _sympify(text: str, rename):
    """Parse an expression with identifiers mapped to symbols.

    Returns:
        The SymPy expression, or ``None`` when it cannot be parsed.
    """
    symbols: Dict[str, sp.Symbol] = {}

    def placeholder(match: re.Match) -> str:
        name = f"_v{len(symbols)}"
        symbols[name] = sp.Symbol(rename(match.group(0)))
        return name

    try:
        return sp.sympify(_IDENT_RE.sub(placeholder, text), locals=symbols)
    except (sp.SympifyError, SyntaxError, TypeError, ValueError):
        return None


def _rename_tokens(text: str, rename) -> str:
    """Rename identifiers inside an expression that SymPy cannot parse.

    Returns:
        The expression with identifiers renamed.
    """
    return _IDENT_RE.sub(lambda m: rename(m.group(0)), text)


# -- comparison ----------------------------------------------------------------


class _Differ:
    """Accumulates differences between two views."""

    def __init__(self, *, rtol: float, atol: float, skipped: set) -> None:
        """Store tolerances and the kinds to skip."""
        self.rtol = rtol
        self.atol = atol
        self.skipped = skipped
        self.differences: List[Difference] = []

    def add(self, kind: str, key: str, message: str, a=None, b=None) -> None:
        """Record a difference unless its kind is skipped."""
        if kind not in self.skipped:
            self.differences.append(Difference(kind, key, message, a, b))

    def close(self, x: Optional[float], y: Optional[float]) -> bool:
        """Compare two optional numbers within tolerance.

        Returns:
            True when both are ``None`` or both are close.
        """
        if x is None or y is None:
            return x is None and y is None
        return math.isclose(x, y, rel_tol=self.rtol, abs_tol=self.atol)

    # -- network -----------------------------------------------------------

    def network(self, a: _View, b: _View) -> None:
        """Compare metabolites, reactions and atom transitions."""
        self._sets("metabolite", a.metabolites, b.metabolites)
        for met in sorted(a.metabolites & b.metabolites):
            if a.atoms[met] != b.atoms[met]:
                self.add(
                    "tracked_atoms",
                    met,
                    f"tracks {a.atoms[met]} atoms in a, {b.atoms[met]} in b",
                    a.atoms[met],
                    b.atoms[met],
                )
        self._sets("reaction", set(a.reactions), set(b.reactions))
        for rid in sorted(set(a.reactions) & set(b.reactions)):
            ra, rb = a.reactions[rid], b.reactions[rid]
            if ra.reversibility != rb.reversibility:
                self.add(
                    "reversibility",
                    rid,
                    f"reversible={ra.reversibility} in a, "
                    f"{rb.reversibility} in b",
                    ra.reversibility,
                    rb.reversibility,
                )
            sa, sb = a.stoichiometry(ra), b.stoichiometry(rb)
            if set(sa) != set(sb) or any(
                not self.close(sa[m], sb[m]) for m in sa
            ):
                self.add(
                    "stoichiometry",
                    rid,
                    f"{_format_stoich(sa)} in a, {_format_stoich(sb)} in b",
                    sa,
                    sb,
                )
            self._transitions(
                rid, a.transitions.get(rid), b.transitions.get(rid)
            )

    def _sets(self, kind: str, a: set, b: set) -> None:
        """Report members present on one side only."""
        for item in sorted(a - b):
            self.add(kind, item, "only in a")
        for item in sorted(b - a):
            self.add(kind, item, "only in b")

    def _transitions(
        self,
        rid: str,
        a: Optional[Dict[FrozenSet[Transition], float]],
        b: Optional[Dict[FrozenSet[Transition], float]],
    ) -> None:
        """Compare the variant mappings of one reaction."""
        if not a and not b:
            return
        if not a or not b:
            side = "b" if not a else "a"
            self.add("atom_transition", rid, f"atom transitions only in {side}")
            return
        only_a = [m for m in a if m not in b]
        only_b = [m for m in b if m not in a]
        if only_a or only_b:
            detail = _mapping_detail(only_a, only_b)
            self.add(
                "atom_transition",
                rid,
                f"{len(only_a)} variant mapping(s) only in a, "
                f"{len(only_b)} only in b{detail}",
                only_a,
                only_b,
            )
            return
        for mapping, weight in a.items():
            if not self.close(weight, b[mapping]):
                self.add(
                    "atom_transition",
                    rid,
                    f"variant weight {weight:g} in a, {b[mapping]:g} in b",
                    weight,
                    b[mapping],
                )
                return

    # -- experiments -------------------------------------------------------

    def experiments(self, a: _View, b: _View) -> None:
        """Pair experiments and compare tracers, constraints, measurements."""
        exps_a, exps_b = a.experiments(), b.experiments()
        pairs = [(n, exps_a[n], exps_b[n]) for n in exps_a if n in exps_b]
        rest_a = [n for n in exps_a if n not in exps_b]
        rest_b = [n for n in exps_b if n not in exps_a]
        if len(rest_a) == 1 and len(rest_b) == 1:
            pairs.append((rest_a[0], exps_a[rest_a[0]], exps_b[rest_b[0]]))
        else:
            for name in rest_a:
                self.add("experiment", name, "only in a")
            for name in rest_b:
                self.add("experiment", name, "only in b")
        if not exps_a and not exps_b:
            self._constraints(
                "network", a.constraints(None), b.constraints(None)
            )
        for name, exp_a, exp_b in pairs:
            self._tracers(name, a.tracers(exp_a), b.tracers(exp_b))
            self._constraints(name, a.constraints(exp_a), b.constraints(exp_b))
            tables_a, tables_b = a.measurements(exp_a), b.measurements(exp_b)
            for kind in tables_a:
                self._measurements(kind, name, tables_a[kind], tables_b[kind])

    def _tracers(self, exp: str, a: Dict, b: Dict) -> None:
        """Compare tracer compositions per input metabolite."""
        for met in sorted(set(a) | set(b)):
            key = f"{exp}/{met}"
            if met not in a or met not in b:
                side = "a" if met in a else "b"
                self.add("tracer", key, f"tracer input only in {side}")
                continue
            la, lb = a[met], b[met]
            same = len(la) == len(lb) and all(
                pa == pb and self.close(fa, fb) and self.close(ua, ub)
                for (pa, fa, ua), (pb, fb, ub) in zip(la, lb)
            )
            if not same:
                self.add(
                    "tracer",
                    key,
                    f"{_format_labels(la)} in a, {_format_labels(lb)} in b",
                    la,
                    lb,
                )

    def _constraints(
        self, exp: str, a: FrozenSet[str], b: FrozenSet[str]
    ) -> None:
        """Compare canonical constraint sets."""
        for item in sorted(a - b):
            self.add("constraint", f"{exp}/{item}", "only in a")
        for item in sorted(b - a):
            self.add("constraint", f"{exp}/{item}", "only in b")

    def _measurements(
        self,
        kind: str,
        exp: str,
        a: Dict[Tuple[Any, ...], List[Point]],
        b: Dict[Tuple[Any, ...], List[Point]],
    ) -> None:
        """Match measured values by key and time, within tolerance."""
        for key in sorted(set(a) | set(b), key=repr):
            label = f"{exp}/{_format_key(key)}"
            pending = sorted(b.get(key, []), key=_point_order)
            for point in sorted(a.get(key, []), key=_point_order):
                candidates = [
                    i
                    for i, other in enumerate(pending)
                    if self.close(point[0], other[0])
                ]
                if not candidates:
                    self.add(kind, label, f"{_format_point(point)} only in a")
                    continue
                exact = [
                    i
                    for i in candidates
                    if self.close(point[1], pending[i][1])
                    and self.close(point[2], pending[i][2])
                ]
                index = exact[0] if exact else candidates[0]
                other = pending.pop(index)
                if not exact:
                    self.add(
                        kind,
                        label,
                        f"{_format_point(point)} in a, "
                        f"{_format_point(other)} in b",
                        point,
                        other,
                    )
            for other in pending:
                self.add(kind, label, f"{_format_point(other)} only in b")


# -- formatting ----------------------------------------------------------------


def _point_order(point: Point) -> Tuple[float, float]:
    """Sort key for ``(time, value, sd)`` points.

    Returns:
        ``(time or -inf, value)``.
    """
    return (point[0] if point[0] is not None else -math.inf, point[1])


def _format_point(point: Point) -> str:
    """Describe one measured point.

    Returns:
        ``"value±sd"`` with ``" @t"`` when timed.
    """
    time, value, sd = point
    text = f"{value:g}±{sd:g}"
    return text if time is None else f"{text} @t={time:g}"


def _format_key(key: Tuple[Any, ...]) -> str:
    """Describe a measurement match key.

    Returns:
        A compact string.
    """
    parts = []
    for part in key:
        if isinstance(part, tuple):
            parts.append("[" + ",".join(str(p) for p in part) + "]")
        else:
            parts.append(str(part))
    return " ".join(parts)


def _format_stoich(stoich: Dict[str, float]) -> str:
    """Describe a net stoichiometry.

    Returns:
        ``"{A: -1, B: 1}"`` with sorted keys.
    """
    body = ", ".join(f"{m}: {c:g}" for m, c in sorted(stoich.items()))
    return "{" + body + "}"


def _format_labels(labels: List[Tuple[str, Any, Any]]) -> str:
    """Describe a tracer composition.

    Returns:
        ``"100000:0.75(p=1)"`` items joined by commas.
    """
    if not labels:
        return "no labels"
    return ", ".join(
        f"{pattern}:{fraction}(p={purity})"
        for pattern, fraction, purity in labels
    )


def _mapping_detail(
    only_a: List[FrozenSet[Transition]], only_b: List[FrozenSet[Transition]]
) -> str:
    """Name the atom transitions that differ between the first variants.

    Returns:
        ``"; e.g. A#1->B#2 in a vs A#1->B#3 in b"`` or an empty string.
    """
    if not only_a or not only_b:
        return ""
    first_a, first_b = only_a[0], only_b[0]
    diff_a = sorted(first_a - first_b)[:2]
    diff_b = sorted(first_b - first_a)[:2]

    def describe(items: List[Transition]) -> str:
        return ", ".join(f"{t[0]}#{t[2]}->{t[3]}#{t[5]}" for t in items)

    return f"; e.g. {describe(diff_a)} in a vs {describe(diff_b)} in b"
