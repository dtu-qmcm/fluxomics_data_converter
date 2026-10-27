"""Map each tool's identifiers onto the record's canonical ones.

The canonical id of a reaction or metabolite is the one in the record's
network. A tool input written by hand may call the same reaction ``v5``;
the record keeps that as a tool-namespaced alias (``"influx_si:v5"``) on
the canonical entity, once, so results from that tool can be lined up with
everyone else's (PATH_FORWARD §11).

- :func:`infer_aliases` matches a tool's model against the reference model
  (same id, then a normalised id for metabolites, then the same
  stoichiometry for reactions);
- :func:`add_aliases` stores matches in the record;
- :func:`alias_map` reads them back as ``{native id: canonical id}``, the
  ``id_map`` that :func:`~.equivalence.compare` takes.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

from fluxolink.datamodel import fluxolink as fl

from ..core.core import FluxomicsData


@dataclass
class AliasMatch:
    """Result of :func:`infer_aliases`.

    Attributes:
        metabolites: Tool metabolite id -> reference metabolite id, only
            where the two differ.
        reactions: Tool reaction id -> reference reaction id, only where the
            two differ.
        unmatched_metabolites: Tool metabolites with no unique match.
        unmatched_reactions: Tool reactions with no unique match.
    """

    metabolites: Dict[str, str] = field(default_factory=dict)
    reactions: Dict[str, str] = field(default_factory=dict)
    unmatched_metabolites: List[str] = field(default_factory=list)
    unmatched_reactions: List[str] = field(default_factory=list)

    @property
    def id_map(self) -> Dict[str, str]:
        """Return every rename, for :func:`~.equivalence.compare`."""
        return {**self.metabolites, **self.reactions}


def _normalised(identifier: str) -> str:
    return re.sub(r"[^0-9a-z]", "", identifier.lower())


def _unique_matches(
    keys: Dict[str, object], candidates: Dict[str, object]
) -> Dict[str, str]:
    """Pair ids whose keys match exactly one id on the other side.

    Returns:
        Tool id -> reference id.
    """
    by_key: Dict[object, List[str]] = defaultdict(list)
    for ref_id, key in candidates.items():
        by_key[key].append(ref_id)
    tool_count: Dict[object, int] = defaultdict(int)
    for key in keys.values():
        tool_count[key] += 1
    return {
        tool_id: by_key[key][0]
        for tool_id, key in keys.items()
        if len(by_key.get(key, [])) == 1 and tool_count[key] == 1
    }


def _signature(
    data: FluxomicsData, reaction_id: str, rename: Dict[str, str]
) -> FrozenSet[Tuple[str, float]]:
    reaction = data.model.reactions.get_by_id(reaction_id)
    coefficients: Dict[str, float] = defaultdict(float)
    if reaction.stoichiometry_dict:
        for met, coef in reaction.stoichiometry_dict.items():
            coefficients[rename.get(met, met)] += coef
    else:
        for met in reaction.reactants:
            coefficients[rename.get(met, met)] -= 1.0
        for met in reaction.products:
            coefficients[rename.get(met, met)] += 1.0
    return frozenset(
        (met, round(coef, 9)) for met, coef in coefficients.items() if coef
    )


def infer_aliases(reference: FluxomicsData, other: FluxomicsData) -> AliasMatch:
    """Match a tool's model onto the reference model.

    Metabolites match by id, then by id ignoring case and punctuation
    (``Gluc.ext`` and ``Gluc_ext``). Reactions match by id, then by the
    same stoichiometry under the metabolite matches, in either direction.
    Only unique matches are kept; the equivalence check reports the rest.

    Args:
        reference: The record's model.
        other: The model parsed from the tool input.

    Returns:
        The matches that rename something, plus what stayed unmatched.
    """
    match = AliasMatch()
    ref_mets = set(reference.model.metabolites.ids)
    met_map = {m: m for m in other.model.metabolites.ids if m in ref_mets}
    open_mets = [m for m in other.model.metabolites.ids if m not in met_map]
    taken = set(met_map.values())
    by_name = _unique_matches(
        {m: _normalised(m) for m in open_mets},
        {m: _normalised(m) for m in ref_mets - taken},
    )
    met_map.update(by_name)
    match.metabolites = by_name
    match.unmatched_metabolites = [m for m in open_mets if m not in by_name]

    ref_rxns = set(reference.model.reactions.ids)
    rxn_map = {r: r for r in other.model.reactions.ids if r in ref_rxns}
    open_rxns = [r for r in other.model.reactions.ids if r not in rxn_map]
    free_refs = ref_rxns - set(rxn_map.values())
    ref_keys = {r: _signature(reference, r, {}) for r in free_refs}
    tool_keys = {r: _signature(other, r, met_map) for r in open_rxns}
    forward = _unique_matches(tool_keys, ref_keys)
    flipped = {
        r: frozenset((m, -c) for m, c in key)
        for r, key in tool_keys.items()
        if r not in forward
    }
    backward = _unique_matches(
        flipped,
        {r: k for r, k in ref_keys.items() if r not in forward.values()},
    )
    match.reactions = {**forward, **backward}
    match.unmatched_reactions = [
        r for r in open_rxns if r not in match.reactions
    ]
    return match


def add_aliases(
    ds: fl.FluxoDataset,
    namespace: str,
    match: AliasMatch,
    network_id: Optional[str] = None,
) -> fl.FluxoDataset:
    """Store a tool's renames on the record's canonical entities.

    Args:
        ds: The record.
        namespace: Tool namespace of the aliases (``"influx_si"``).
        match: Result of :func:`infer_aliases`.
        network_id: Network to annotate when the record holds several.

    Returns:
        The record with the aliases added (a copy).
    """
    if not (match.metabolites or match.reactions):
        return ds
    ds = ds.model_copy(deep=True)
    networks = [n for n in ds.networks or [] if network_id in (None, n.id)]
    for network in networks[:1]:
        for entities, renames in (
            (network.metabolites or [], match.metabolites),
            (network.reactions or [], match.reactions),
        ):
            by_canonical: Dict[str, List[str]] = defaultdict(list)
            for native, canonical in renames.items():
                by_canonical[canonical].append(f"{namespace}:{native}")
            for entity in entities:
                new = by_canonical.get(entity.id)
                if new:
                    existing = list(entity.aliases or [])
                    entity.aliases = existing + [
                        a for a in new if a not in existing
                    ]
    return ds


def alias_map(
    ds: fl.FluxoDataset | dict, namespace: Optional[str] = None
) -> Dict[str, str]:
    """Read the record's aliases as ``{native id: canonical id}``.

    Args:
        ds: The record, as a ``FluxoDataset`` or as plain data.
        namespace: Only aliases of this tool; ``None`` takes all.

    Returns:
        Native id to canonical id, for reactions and metabolites.
    """
    doc = (
        ds.model_dump(mode="json", exclude_none=True)
        if isinstance(ds, fl.FluxoDataset)
        else ds
    )
    mapping: Dict[str, str] = {}
    for network in doc.get("networks") or []:
        for kind in ("metabolites", "reactions"):
            for entity in network.get(kind) or []:
                for alias in entity.get("aliases") or []:
                    tool, _, native = alias.partition(":")
                    if namespace in (None, tool):
                        mapping[native] = entity["id"]
    return mapping
