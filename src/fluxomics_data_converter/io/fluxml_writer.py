"""FluxML XML writer for writing FluxML files.

FluxML is an XML-based format used by 13CFlux/13CFlux2 for 13C metabolic
flux analysis.
"""

import warnings
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from itertools import combinations
from xml.dom import minidom
from typing import (
    TYPE_CHECKING,
    Dict,
    FrozenSet,
    List,
    NamedTuple,
    Optional,
    Tuple,
)
from datetime import datetime
from pathlib import Path

from ..core.core import FluxomicsData, LabelingExperiments
from ..model.atom_mapping import AtomMap
from .fluxml_parser import variant_combinations

if TYPE_CHECKING:
    import numpy as np

# An atom map with metadata stripped: product atom -> source atom, each as
# (compound, instance, index).  Two maps are the same transition iff their
# keys are equal.
_MapKey = FrozenSet[Tuple[Tuple[str, int, int], Tuple[str, int, int]]]


class _ReactionLayout(NamedTuple):
    """How one model reaction is written as a FluxML ``<reaction>``.

    ``reducts`` and ``rproducts`` hold, per participant position, the cfg
    strings to write: none (no ``cfg``), one (a plain ``cfg`` attribute) or
    several (a ``<variant>`` list).  ``ids`` lists the reaction ids in the
    order 13CFLUX enumerates the variant cross product.
    """

    ids: List[str]
    reducts: List[List[str]]
    rproducts: List[List[str]]


def _instance_positions(compounds: List[str]) -> Dict[Tuple[str, int], int]:
    """Map each ``(compound, instance)`` to its 1-based position.

    Returns:
        A dict keyed like :class:`AtomAddress` ``(mol, instance)``, where
        the instance counts repeated occurrences of the same compound.
    """
    seen: Dict[str, int] = {}
    positions: Dict[Tuple[str, int], int] = {}
    for pos, cpd in enumerate(compounds, start=1):
        seen[cpd] = seen.get(cpd, 0) + 1
        positions[(cpd, seen[cpd])] = pos
    return positions


def _map_key(atom_map: AtomMap) -> _MapKey:
    """Strip element metadata from an atom map so maps can be compared.

    Returns:
        The map as a frozenset of ``(product atom, source atom)`` pairs.
    """
    return frozenset(
        ((p.mol, p.instance, p.index), (s.mol, s.instance, s.index))
        for p, s in atom_map.mapping.items()
    )


class FluxMLWriter:
    """Serialiser for FluxML XML files (13CFlux2 format).

    Converts a :class:`~fluxomics_data_converter.FluxomicsData` into a
    pretty-printed UTF-8 XML file.

    Variant reactions
    -----------------
    13CFLUX reads the ``<variant>`` lists of a reaction as a cross product
    over every reduct and rproduct, and requires the reaction's id list to
    be exactly as long as that product.  The writer therefore factorises a
    reaction's atom maps into per-participant variants: only a participant
    whose cfg really differs between maps gets a ``<variant>`` list (a
    symmetric product as an rproduct variant, a symmetric substrate as a
    reduct variant), every other participant gets a plain ``cfg``.  The id
    list is ordered the way 13CFLUX enumerates the combinations (first
    reduct fastest).  A reaction whose maps cannot be written as such a
    cross product raises ``ValueError``.

    Input pools and drains
    ----------------------
    - Every tracer becomes an ``<input>``.  A metabolite that no reaction
      can produce (not a product, and not a reactant of a reversible
      reaction) and that carries atoms is declared as an *unlabelled*
      ``<input>`` in each configuration that has no tracer for it.
    - 13CFLUX balances every pool, so a pool that is produced but never
      consumed gets an extra irreversible ``<pool>_out`` drain reaction
      (``add_drains=True``, the default).  Metabolites whose id ends in
      ``_ext`` are treated as external boundary pools: they get no drain
      and are left out of the mass balance used to fill in missing flux
      values.

    Known limitations
    -----------------
    - Named constraints (``"name: expression"``) lose their name prefix on
      write; only the expression is emitted.
    - Experiment-level ``<constraints>`` sections are not written.
    - ``<variant>`` elements carry no ``ratio`` attribute, so every variant
      combination gets the same weight; reactions with non-uniform atom-map
      weights raise ``ValueError``.
    - Tracer ``type`` is always written as ``"isotopomer"``; the actual
      ``Tracers.type`` field value is ignored.

    Example::

        writer = FluxMLWriter()
        writer.write(model, "output/ecoli.fml")
        # or use the convenience function:
        from fluxomics_data_converter.io import write_fluxml
        write_fluxml(model, "output/ecoli.fml")
    """

    NAMESPACE = "http://www.13cflux.net/fluxml"

    def __init__(self):
        """Initialise the writer with empty state; call write() to serialise."""
        self._model: Optional[FluxomicsData] = None
        self._ns = self.NAMESPACE
        self._metabolite_atom_counts: Dict[str, int] = {}
        self._layouts: Dict[str, _ReactionLayout] = {}
        self._sources: List[str] = []
        self._drains: List[str] = []
        self._input_pools: set = set()
        self._natural_abundance = False
        self._add_drains = True

    def _compute_metabolite_atom_counts(self) -> Dict[str, int]:
        """Compute atom counts for metabolites by scanning atom transitions.

        Returns:
            A dict mapping metabolite id to atom count.
        """
        atom_counts: Dict[str, int] = {}

        for (
            reaction_id,
            atom_mapping,
        ) in self._model.model.atom_mappings.items():
            if not atom_mapping.maps:
                continue

            # Use the first atom map to get atom counts
            atom_map = next(iter(atom_mapping.maps.values()))

            # Scan reactant atoms (from source addresses)
            for src_addr in atom_map.mapping.values():
                mol_id = src_addr.mol
                atom_idx = src_addr.index
                if mol_id not in atom_counts:
                    atom_counts[mol_id] = atom_idx
                else:
                    atom_counts[mol_id] = max(atom_counts[mol_id], atom_idx)

            # Scan product atoms (from destination addresses)
            for prod_addr in atom_map.mapping.keys():
                mol_id = prod_addr.mol
                atom_idx = prod_addr.index
                if mol_id not in atom_counts:
                    atom_counts[mol_id] = atom_idx
                else:
                    atom_counts[mol_id] = max(atom_counts[mol_id], atom_idx)

        return atom_counts

    # Natural abundance of 13C per carbon position, as used by
    # x3cflux internally.
    # Source: x3cflux/src/main/cpp/model/system/NaturalLabelingInitializer.cpp
    NATURAL_ABUNDANCE_CARBON: float = 0.01109

    def write(
        self,
        model: FluxomicsData,
        filepath: str,
        apply_na_correction: bool = False,
        natural_abundance: bool = False,
        add_drains: bool = True,
    ) -> None:
        """Write FluxomicsData to a FluxML file.

        Args:
            model: FluxomicsData to write
            filepath: Path to output file
            apply_na_correction: If True, apply the ¹³C natural abundance
                correction matrix M_C (p=0.01109/C) to all MS measurement
                data before writing.  Set this when the data is synthetic
                (e.g. from cmfa) and does NOT already contain natural
                abundance, so that x3cflux can recover unbiased flux
                estimates.  Leave False when the data comes from real MS
                experiments (natural abundance is already present in the
                raw signal).  Only meaningful together with
                ``natural_abundance=True``: without it 13CFLUX simulates
                ideal inputs and the corrected data no longer matches.

                Note: M_C is a binomial approximation.  The exact correction
                is network-dependent; expect a residual of ~10⁻³ MID units.
            natural_abundance: Whether 13CFLUX should give the unlabelled
                atoms of every input pool their natural ¹³C abundance.  In
                FluxML this is switched by the ``purity`` attribute of
                ``<label>``: with the attribute present (even
                ``purity="1.0"``), 13CFLUX puts ~1.1 % ¹³C on every
                unlabelled input carbon; without it, the input is the ideal
                isotopomer.  ``False`` (default) writes no ``purity``
                attribute, so a label purity below 1 cannot be expressed
                and is dropped with a warning.  ``True`` writes ``purity``
                on every label (the tracer's purity, or 1.0 when unknown).
            add_drains: Add an irreversible ``<pool>_out`` reaction for
                every pool that is produced but never consumed (except
                ``*_ext`` pools).  13CFLUX rejects a network with such a
                pool, so turn this off only if the model already drains
                its terminal pools.
        """
        self._model = model
        self._apply_na_correction = apply_na_correction
        self._natural_abundance = natural_abundance
        self._add_drains = add_drains
        filepath = Path(filepath)

        if apply_na_correction and not natural_abundance:
            warnings.warn(
                "apply_na_correction=True adds natural abundance to the "
                "measurements, but natural_abundance=False writes ideal "
                "inputs, so 13CFLUX will not simulate it. Pass "
                "natural_abundance=True as well.",
                stacklevel=2,
            )

        # Ensure output directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Build XML tree
        root = self._build_fluxml()

        # Write to file with pretty formatting
        xml_str = ET.tostring(root, encoding="unicode")
        # Pretty print using minidom
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ", encoding="UTF-8")

        # Remove extra blank lines that minidom adds
        lines = pretty_xml.decode("UTF-8").split("\n")
        cleaned_lines = [line for line in lines if line.strip()]
        # Re-add proper XML declaration
        if cleaned_lines and cleaned_lines[0].startswith("<?xml"):
            cleaned_lines[0] = (
                '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
            )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(cleaned_lines))

    def _timestamp(self) -> str:
        """Return the model's own date, or the current time if it has none.

        Using ``metadata.date`` keeps regenerated files reproducible.

        Returns:
            The timestamp formatted as ``YYYY-MM-DD HH:MM:SS``.
        """
        metadata = self._model.metadata
        date = metadata.date if metadata and metadata.date else datetime.now()
        return date.strftime("%Y-%m-%d %H:%M:%S")

    def _build_fluxml(self) -> ET.Element:
        """Build the root fluxml element.

        Returns:
            The root ``<fluxml>`` XML element.
        """
        # Register namespace - use empty prefix for default namespace
        ET.register_namespace("", self.NAMESPACE)

        # Create root element with namespace (don't use xmlns
        # attribute separately)
        root = ET.Element("fluxml")
        root.set("xmlns", self.NAMESPACE)

        # Add metadata section
        self._add_metadata(root)

        # Add reaction network
        self._add_reaction_network(root)

        # Add constraints
        self._add_constraints(root)

        # Add experiments/configurations
        self._add_experiments(root)

        return root

    def _add_metadata(self, root: ET.Element) -> None:
        """Add metadata element with metadata."""
        if not self._model.metadata:
            return

        metadata = ET.SubElement(root, "info")

        date_elem = ET.SubElement(metadata, "date")
        date_elem.text = self._timestamp()

        if self._model.metadata.comment:
            comment_elem = ET.SubElement(metadata, "comment")
            comment_elem.text = self._model.metadata.comment

        if self._model.metadata.modeler:
            modeler_elem = ET.SubElement(metadata, "modeler")
            modeler_elem.text = self._model.metadata.modeler

        if self._model.metadata.strain:
            strain_elem = ET.SubElement(metadata, "strain")
            strain_elem.text = self._model.metadata.strain

    def _atom_count(self, metabolite_id: str) -> int:
        """Return a metabolite's atom count, inferred from the maps if unset.

        Returns:
            The number of atoms, or 0 if unknown.
        """
        metabolite = next(
            (m for m in self._model.model.metabolites if m.id == metabolite_id),
            None,
        )
        if metabolite is not None and metabolite.atoms:
            return metabolite.atoms
        return self._metabolite_atom_counts.get(metabolite_id, 0)

    def _classify_pools(self) -> None:
        """Find the source pools, input pools and drained sink pools.

        Sets ``self._sources`` (pools no reaction can produce, which become
        unlabelled inputs unless a tracer covers them), ``self._input_pools``
        and ``self._drains`` (pools that get an ``_out`` reaction), and
        warns about pools 13CFLUX would reject.
        """
        reactions = list(self._model.model.reactions)
        consumed: set = set()
        produced: set = set()
        producible: set = set()  # produced in either direction
        for rxn in reactions:
            consumed.update(rxn.reactants)
            produced.update(rxn.products)
            producible.update(rxn.products)
            if rxn.reversibility:
                producible.update(rxn.reactants)
        tracer_pools = {
            tracer.metabolite
            for exp in self._model.experiments
            for tracer in exp.tracers
        }

        sources = sorted(consumed - producible)
        self._sources = [m for m in sources if self._atom_count(m) > 0]
        self._input_pools = tracer_pools | set(self._sources)

        untraced = sorted(
            m
            for m in self._sources
            if any(
                m not in {t.metabolite for t in exp.tracers}
                for exp in self._model.experiments
            )
        )
        if untraced:
            warnings.warn(
                f"Declared {len(untraced)} pool(s) that no reaction produces "
                f"as unlabelled inputs: {untraced}",
                stacklevel=4,
            )

        sinks = sorted(
            m
            for m in (produced - consumed - self._input_pools)
            if not m.endswith("_ext")
        )
        self._drains = sinks if self._add_drains else []
        if sinks and not self._add_drains:
            warnings.warn(
                f"{len(sinks)} pool(s) are produced but never consumed and "
                f"add_drains=False, so 13CFLUX will reject the network: "
                f"{sinks}",
                stacklevel=4,
            )

        only_reactant = sorted(consumed - produced - self._input_pools)
        if only_reactant:
            warnings.warn(
                f"{len(only_reactant)} pool(s) never appear as a reaction "
                f"product and are not inputs, which 13CFLUX rejects (a pool "
                f"filled only by the backward direction of reversible "
                f"reactions, or a source without atoms): {only_reactant}. "
                f"Write those reactions in the direction of their net flux, "
                f"or declare the pool as an input.",
                stacklevel=4,
            )

    def _add_reaction_network(self, root: ET.Element) -> None:
        """Add reactionnetwork element."""
        rn = ET.SubElement(root, "reactionnetwork")

        # Compute atom counts from atom transitions for metabolites
        # missing this info
        self._metabolite_atom_counts = self._compute_metabolite_atom_counts()
        self._classify_pools()

        # Add metabolite pools
        pools = ET.SubElement(rn, "metabolitepools")
        for metabolite in self._model.model.metabolites:
            pool = ET.SubElement(pools, "pool")
            pool.set("id", metabolite.id)

            # Use metabolite.atoms if available, otherwise infer from
            # atom transitions
            atom_count = metabolite.atoms
            if not atom_count:
                atom_count = self._metabolite_atom_counts.get(metabolite.id)
            if atom_count:
                pool.set("atoms", str(atom_count))

            # Add annotations
            if metabolite.annotations:
                for ann in metabolite.annotations:
                    ann_elem = ET.SubElement(pool, "annotation")
                    ann_elem.set("name", ann.name)
                    if ann.content:
                        ann_elem.text = ann.content

        # Add reactions
        self._layouts = {}
        for reaction in self._model.model.reactions:
            layout = self._layout_reaction(reaction)
            self._layouts[reaction.id] = layout
            self._add_reaction(rn, reaction, layout)

        # Add drain reactions for sink metabolites (produced but
        # never consumed).  13CFlux2 / x3cflux requires every output
        # metabolite to have an explicit sink reaction so the network
        # can be validated as structurally consistent.
        for metab_id in self._drains:
            drain = ET.SubElement(rn, "reaction")
            drain.set("id", f"{metab_id}_out")
            drain.set("bidirectional", "false")
            reduct = ET.SubElement(drain, "reduct")
            reduct.set("id", metab_id)
            atom_count = self._atom_count(metab_id)
            if atom_count:
                cfg = " ".join(f"C#{i}@1" for i in range(1, atom_count + 1))
                reduct.set("cfg", cfg)

    def _add_reaction(
        self, rn: ET.Element, reaction, layout: _ReactionLayout
    ) -> None:
        """Add a single reaction element."""
        rxn = ET.SubElement(rn, "reaction")
        rxn.set("id", " ".join(layout.ids))

        # Reversibility (FluxML default is bidirectional=true)
        if not reaction.reversibility:
            rxn.set("bidirectional", "false")

        # Add annotations
        if hasattr(reaction, "annotations") and reaction.annotations:
            for ann in reaction.annotations:
                ann_elem = ET.SubElement(rxn, "annotation")
                ann_elem.set("name", ann.name)
                if ann.content:
                    ann_elem.text = ann.content

        for tag, compounds, cfg_lists in (
            ("reduct", reaction.reactants, layout.reducts),
            ("rproduct", reaction.products, layout.rproducts),
        ):
            for compound_id, cfgs in zip(compounds, cfg_lists):
                elem = ET.SubElement(rxn, tag)
                elem.set("id", compound_id)
                if len(cfgs) == 1:
                    elem.set("cfg", cfgs[0])
                elif len(cfgs) > 1:
                    for cfg in cfgs:
                        ET.SubElement(elem, "variant").set("cfg", cfg)

    def _layout_reaction(self, reaction) -> _ReactionLayout:
        """Work out a reaction's ids and per-participant cfg strings.

        The atom maps of a reaction are factorised into per-participant
        variants relative to the first map: a map that differs from it only
        in one product's atoms gives that product a variant, one that
        differs only in which atoms of one reactant are used gives that
        reactant a (relabelling) variant.  The smallest set of such
        participants whose cross product reproduces the maps - with their
        multiplicities if possible, else their weights - is written.

        Cfg tokens are ``E#k@r``: atom ``k`` of the reactant at position
        ``r``.  13CFLUX only matches tokens between the reduct and the
        rproduct side, so a reduct variant is a permutation of its tokens.

        Returns:
            The reaction's :class:`_ReactionLayout`.

        Raises:
            ValueError: If the reaction's atom maps are not a cross product
                of per-participant variants, which FluxML cannot express.
        """
        reducts: List[List[str]] = [[] for _ in reaction.reactants]
        rproducts: List[List[str]] = [[] for _ in reaction.products]
        mapping = self._model.model.atom_mappings.get(reaction.id)
        if not mapping or not mapping.maps:
            self._fill_efflux_cfgs(reaction, reducts, rproducts)
            return _ReactionLayout([reaction.id], reducts, rproducts)

        r_pos = _instance_positions(reaction.reactants)
        p_pos = _instance_positions(reaction.products)
        r_at = {pos: key for key, pos in r_pos.items()}
        p_at = {pos: key for key, pos in p_pos.items()}
        maps = list(mapping.maps.items())

        # Element of every reactant atom used by any map
        elements: Dict[int, Dict[int, str]] = defaultdict(dict)
        for _, atom_map in maps:
            for src in atom_map.mapping.values():
                pos = r_pos.get((src.mol, src.instance))
                if pos is not None:
                    elements[pos].setdefault(src.index, src.element or "C")

        def symbol(pos: int, index: int) -> str:
            return f"{elements[pos].get(index, 'C')}#{index}@{pos}"

        def product_cfg(atom_map: AtomMap, slot: int) -> str:
            tokens = []
            for prod, src in atom_map.mapping.items():
                if p_pos.get((prod.mol, prod.instance)) != slot:
                    continue
                pos = r_pos.get((src.mol, src.instance))
                if pos is not None:
                    tokens.append((prod.index, symbol(pos, src.index)))
            return " ".join(token for _, token in sorted(tokens))

        base_id, base = maps[0]
        base_red = {
            pos: " ".join(symbol(pos, i) for i in range(1, max(atoms) + 1))
            for pos, atoms in elements.items()
        }
        base_prod = {
            slot: cfg for slot in p_at if (cfg := product_cfg(base, slot))
        }
        product_atoms = {
            slot: sorted(
                p.index
                for p in base.mapping
                if p_pos.get((p.mol, p.instance)) == slot
            )
            for slot in base_prod
        }

        def resolve(
            red: Dict[int, str], prod: Dict[int, str]
        ) -> Optional[_MapKey]:
            """Read an atom map back from cfgs, by matching tokens.

            Returns:
                The map, or ``None`` if the cfgs do not determine one.
            """
            where = {}
            for pos, cfg in red.items():
                mol, inst = r_at[pos]
                for k, token in enumerate(cfg.split(), start=1):
                    where[token] = (mol, inst, k)
            pairs = []
            for slot, cfg in prod.items():
                mol, inst = p_at[slot]
                tokens = cfg.split()
                if len(tokens) != len(product_atoms[slot]):
                    return None
                for index, token in zip(product_atoms[slot], tokens):
                    if token not in where:
                        return None
                    pairs.append(((mol, inst, index), where[token]))
            return frozenset(pairs)

        def layout(red: Dict[int, List[str]], prod: Dict[int, List[str]]):
            return (
                [red.get(pos, []) for pos in range(1, len(reducts) + 1)],
                [prod.get(slot, []) for slot in range(1, len(rproducts) + 1)],
            )

        keys = {map_id: _map_key(atom_map) for map_id, atom_map in maps}
        if (
            len(set(keys.values())) == 1
            or resolve(base_red, base_prod) != keys[base_id]
        ):
            # One distinct transition (or one these cfgs cannot express):
            # write the first map as plain cfgs under the reaction id.
            if len(set(keys.values())) > 1:
                raise ValueError(
                    f"Reaction {reaction.id!r}: its atom maps cannot be "
                    f"written as FluxML cfg strings."
                )
            red, prod = layout(
                {pos: [cfg] for pos, cfg in base_red.items()},
                {slot: [cfg] for slot, cfg in base_prod.items()},
            )
            return _ReactionLayout([reaction.id], red, prod)

        # Candidate variants per participant: (0, slot) for a product,
        # (1, position) for a reactant; the base cfg comes first.
        candidates: Dict[Tuple[int, int], List[str]] = {}

        def add(part: Tuple[int, int], cfg: str, base_cfg: str) -> None:
            known = candidates.setdefault(part, [base_cfg])
            if cfg not in known:
                known.append(cfg)

        b = base.mapping
        for map_id, atom_map in maps[1:]:
            m = atom_map.mapping
            if m.keys() != b.keys():
                continue
            diff = [x for x in b if m[x] != b[x]]
            if not diff:
                continue

            slots = {p_pos.get((x.mol, x.instance)) for x in diff}
            if len(slots) == 1 and None not in slots:
                slot = slots.pop()
                if slot in base_prod:
                    cfg = product_cfg(atom_map, slot)
                    if (
                        resolve(base_red, {**base_prod, slot: cfg})
                        == keys[map_id]
                    ):
                        add((0, slot), cfg, base_prod[slot])

            sources = {r_pos.get((b[x].mol, b[x].instance)) for x in diff}
            sources |= {r_pos.get((m[x].mol, m[x].instance)) for x in diff}
            if len(sources) == 1 and None not in sources:
                pos = sources.pop()
                cfg = self._relabelled_reduct(
                    [(b[x].index, m[x].index) for x in diff],
                    len(base_red[pos].split()),
                    lambda i, pos=pos: symbol(pos, i),
                )
                if (
                    cfg
                    and resolve({**base_red, pos: cfg}, base_prod)
                    == keys[map_id]
                ):
                    add((1, pos), cfg, base_red[pos])

        # Target: the maps' transitions with their multiplicities/weights
        target = Counter(keys.values())
        weights = mapping.weights or {}
        target_weight: Dict[_MapKey, float] = defaultdict(float)
        for map_id, _ in maps:
            target_weight[keys[map_id]] += weights.get(map_id, 1.0)
        total = sum(target_weight.values())
        uniform = len({round(weights.get(i, 1.0), 12) for i, _ in maps}) == 1

        parts = sorted(candidates)  # products first, then reactants
        exact = by_weight = None
        for size in range(len(parts) + 1):
            for subset in sorted(
                combinations(parts, size), key=lambda s: sum(p[0] for p in s)
            ):
                red = {
                    pos: candidates[(1, pos)] if (1, pos) in subset else [cfg]
                    for pos, cfg in base_red.items()
                }
                prod = {
                    slot: (
                        candidates[(0, slot)] if (0, slot) in subset else [cfg]
                    )
                    for slot, cfg in base_prod.items()
                }
                order = [(1, pos) for pos in sorted(red)] + [
                    (0, slot) for slot in sorted(prod)
                ]
                lists = [red[i] if side else prod[i] for side, i in order]
                combos = variant_combinations(lists)
                if len(combos) > 4096:
                    continue
                combo_keys = []
                for combo in combos:
                    chosen = dict(zip(order, combo))
                    combo_keys.append(
                        resolve(
                            {p: chosen[(1, p)] for p in red},
                            {s: chosen[(0, s)] for s in prod},
                        )
                    )
                if exact is None and uniform and Counter(combo_keys) == target:
                    exact = (red, prod, combo_keys)
                if by_weight is None:
                    counts = Counter(combo_keys)
                    if counts.keys() == target_weight.keys() and all(
                        abs(counts[k] / len(combo_keys) - w / total) < 1e-9
                        for k, w in target_weight.items()
                    ):
                        by_weight = (red, prod, combo_keys)
            if exact:
                break

        if exact:
            red, prod, combo_keys = exact
            unused: Dict[_MapKey, List[str]] = defaultdict(list)
            for map_id, _ in maps:
                unused[keys[map_id]].append(map_id)
            ids = [unused[k].pop(0) for k in combo_keys]
        elif by_weight:
            red, prod, combo_keys = by_weight
            ids = [
                f"{reaction.id}___{i}" for i in range(1, len(combo_keys) + 1)
            ]
        else:
            raise ValueError(
                f"Reaction {reaction.id!r}: its {len(maps)} atom maps are not "
                f"a cross product of per-participant variants with equal "
                f"weights, which is the only kind of alternative mapping "
                f"FluxML can express."
            )
        if len(ids) == 1:
            ids = [reaction.id]
        red_lists, prod_lists = layout(red, prod)
        return _ReactionLayout(ids, red_lists, prod_lists)

    @staticmethod
    def _relabelled_reduct(
        moves: List[Tuple[int, int]], n_atoms: int, symbol
    ) -> Optional[str]:
        """Build the cfg of a reduct whose atoms are relabelled.

        Args:
            moves: ``(atom index in the base map, atom index in the
                variant)`` for every product atom whose source moved.
            n_atoms: Number of atoms of the reactant.
            symbol: Function giving the cfg token of an atom index.

        Returns:
            The reduct cfg string, or None if *moves* is not part of a
            permutation of the reactant's atoms.
        """
        perm: Dict[int, int] = {}
        for source, target in moves:
            if perm.setdefault(source, target) != target:
                return None
        if len(set(perm.values())) != len(perm):
            return None
        if any(not 1 <= i <= n_atoms for i in list(perm) + list(perm.values())):
            return None
        rest_dom = [i for i in range(1, n_atoms + 1) if i not in perm]
        rest_rng = set(range(1, n_atoms + 1)) - set(perm.values())
        for i in [i for i in rest_dom if i in rest_rng]:
            perm[i] = i
        rest_dom = [i for i in rest_dom if i not in perm]
        rest_rng = sorted(rest_rng - set(perm.values()))
        perm.update(zip(rest_dom, rest_rng))
        inverse = {target: source for source, target in perm.items()}
        return " ".join(symbol(inverse[k]) for k in range(1, n_atoms + 1))

    def _fill_efflux_cfgs(
        self,
        reaction,
        reducts: List[List[str]],
        rproducts: List[List[str]],
    ) -> None:
        """Give the reducts of an efflux reaction a cfg.

        A reaction with no product atoms (an efflux in FluxML) has no atom
        map, but 13CFLUX still requires a ``cfg`` on every reduct that has
        atoms.  Its atoms go nowhere, so any unique labelling will do.
        """
        if any(self._atom_count(p) for p in reaction.products):
            return
        for pos, reactant_id in enumerate(reaction.reactants, start=1):
            n_atoms = self._atom_count(reactant_id)
            if n_atoms:
                reducts[pos - 1] = [
                    " ".join(f"C#{i}@{pos}" for i in range(1, n_atoms + 1))
                ]

    def _add_constraints(self, root: ET.Element) -> None:
        """Add constraints element."""
        constraints = self._model.constraints
        if not constraints:
            return

        # Check if there are any constraints to write
        has_net = constraints.net and constraints.net.formulas
        has_xch = constraints.xch and constraints.xch.formulas
        has_psize = (
            constraints.metabolitesize and constraints.metabolitesize.formulas
        )

        if not (has_net or has_xch or has_psize):
            return

        constr_elem = ET.SubElement(root, "constraints")

        # Add net constraints
        if has_net:
            net_elem = ET.SubElement(constr_elem, "net")
            textual = ET.SubElement(net_elem, "textual")
            formulas = []
            for formula in constraints.net.formulas:
                formulas.append(formula.expression)
            textual.text = ";\n".join(formulas)

        # Add xch constraints
        if has_xch:
            xch_elem = ET.SubElement(constr_elem, "xch")
            textual = ET.SubElement(xch_elem, "textual")
            formulas = []
            for formula in constraints.xch.formulas:
                formulas.append(formula.expression)
            textual.text = ";\n".join(formulas)

        # Add psize constraints
        if has_psize:
            psize_elem = ET.SubElement(constr_elem, "psize")
            textual = ET.SubElement(psize_elem, "textual")
            formulas = []
            for formula in constraints.metabolitesize.formulas:
                formulas.append(formula.expression)
            textual.text = ";\n".join(formulas)

    def _add_experiments(self, root: ET.Element) -> None:
        """Add configuration elements for experiments."""
        dropped: List[str] = []
        for experiment in self._model.experiments:
            dropped += self._add_experiment(root, experiment)
        if dropped:
            warnings.warn(
                f"Label purity below 1 dropped for {dropped}: FluxML can "
                f"only express purity together with natural abundance. "
                f"Pass natural_abundance=True to keep it.",
                stacklevel=4,
            )

    def _add_experiment(
        self, root: ET.Element, experiment: LabelingExperiments
    ) -> List[str]:
        """Add a single configuration/experiment element.

        Returns:
            Descriptions of label purities that could not be written.
        """
        config = ET.SubElement(root, "configuration")
        config.set("name", experiment.name)
        config.set("stationary", "true" if experiment.stationary else "false")

        if experiment.time is not None:
            config.set("time", str(experiment.time))

        # Add tracer inputs, then an unlabelled input for every other source
        dropped: List[str] = []
        for tracer in experiment.tracers:
            dropped += self._add_tracer_input(config, tracer)
        traced = {tracer.metabolite for tracer in experiment.tracers}
        for metabolite_id in self._sources:
            if metabolite_id not in traced:
                self._add_unlabelled_input(config, metabolite_id)

        # Add measurement section
        if experiment.measurement:
            self._add_measurement(config, experiment)

        # Add simulation section
        if experiment.simulation:
            self._add_simulation(config, experiment)
        return dropped

    def _add_simulation(
        self, config: ET.Element, experiment: LabelingExperiments
    ) -> None:
        """Add simulation element with flux and pool-size initial values.

        Writes the stored simulation values, then derives net values for
        reactions that have none - D-type reactions from MTF parsing and
        the auto-generated ``_out`` drains - from the steady-state mass
        balance S·v = 0.  Only values that the stored ones determine
        uniquely are written; see :meth:`_derive_net_fluxes`.

        A flux stored under the id of a variant reaction is split evenly
        over its variant ids (``SCS`` -> ``SCS___1``, ``SCS___2``), which is
        what 13CFLUX's own constraints between variants require.
        """
        sim = ET.SubElement(config, "simulation")
        sim.set("method", "auto")
        sim.set("type", "full")

        variables = ET.SubElement(sim, "variables")
        stored = experiment.simulation.variables
        flux_values = stored.flux_values if stored else []

        reactions = list(self._model.model.reactions)
        ids_of = {r.id: self._layouts[r.id].ids for r in reactions}
        variant_of = {
            variant_id: rxn_id
            for rxn_id, ids in ids_of.items()
            if len(ids) > 1
            for variant_id in ids
        }
        drain_ids = [f"{m}_out" for m in self._drains]
        known = set(ids_of) | set(variant_of) | set(drain_ids)

        def emit(flux_id: str, flux_type: str, value: float) -> None:
            fv_elem = ET.SubElement(variables, "fluxvalue")
            fv_elem.set("flux", flux_id)
            fv_elem.set("type", flux_type)
            fv_elem.text = str(value)

        net_known: Dict[str, float] = {}
        unknown: List[str] = []
        for fv in flux_values:
            # Skip D-type fluxes with no stored value (influx_si MTF parsing
            # produces these; they must be derived from mass balance instead).
            if fv.value is None:
                continue
            # MTF free/constrained/dependent (F/C/D) values are all net
            flux_type = fv.type if fv.type in ("net", "xch") else "net"
            ids = ids_of.get(fv.flux, [fv.flux])
            for flux_id in ids:
                emit(flux_id, flux_type, fv.value / len(ids))
            if fv.flux not in known:
                unknown.append(fv.flux)
            if flux_type != "net":
                continue
            if fv.flux in variant_of:
                # One variant of a reaction: 13CFLUX ties the others to it
                rxn_id = variant_of[fv.flux]
                net_known.setdefault(rxn_id, fv.value * len(ids_of[rxn_id]))
            else:
                net_known[fv.flux] = fv.value
        if unknown:
            warnings.warn(
                f"<fluxvalue> names that are not reaction ids in the written "
                f"network: {sorted(set(unknown))}",
                stacklevel=5,
            )

        columns = [r.id for r in reactions] + drain_ids
        missing = [c for c in columns if c not in net_known]
        if missing:
            derived = self._derive_net_fluxes(columns, net_known, missing)
            for rxn_id in missing:
                if rxn_id not in derived:
                    continue
                ids = ids_of.get(rxn_id, [rxn_id])
                for flux_id in ids:
                    emit(flux_id, "net", derived[rxn_id] / len(ids))

        for pv in stored.metabolitesize_values if stored else []:
            pv_elem = ET.SubElement(variables, "poolvalue")
            pv_elem.set("pool", pv.metabolite)
            pv_elem.text = str(pv.value)

    def _derive_net_fluxes(
        self,
        columns: List[str],
        net_known: Dict[str, float],
        missing: List[str],
    ) -> Dict[str, float]:
        """Solve the steady-state mass balance for the missing net fluxes.

        Input pools and ``*_ext`` pools are not balanced.  A missing flux is
        derived only if the known fluxes determine it uniquely (it has no
        component in the null space of the balance restricted to the
        missing fluxes); the others are left out of the file, with a
        warning, rather than given an arbitrary least-squares value.

        Args:
            columns: Model reaction ids followed by drain reaction ids.
            net_known: Net flux of every reaction that has a stored value.
            missing: Columns without a stored value.

        Returns:
            Derived net flux per determined column.

        Raises:
            ValueError: If a derived flux is negative on an irreversible
                reaction (the stored flux map contradicts the network).
        """
        import numpy as np

        reactions = list(self._model.model.reactions)
        metabolites = list(self._model.model.metabolites)
        met_idx = {m.id: i for i, m in enumerate(metabolites)}
        col_idx = {c: j for j, c in enumerate(columns)}

        S = np.zeros((len(metabolites), len(columns)))
        for j, rxn in enumerate(reactions):
            for mid in rxn.reactants:
                if mid in met_idx:
                    S[met_idx[mid], j] -= 1.0
            for mid in rxn.products:
                if mid in met_idx:
                    S[met_idx[mid], j] += 1.0
        for metab_id in self._drains:
            if metab_id in met_idx:
                S[met_idx[metab_id], col_idx[f"{metab_id}_out"]] -= 1.0

        rows = [
            i
            for i, m in enumerate(metabolites)
            if m.id not in self._input_pools and not m.id.endswith("_ext")
        ]
        known_cols = [c for c in columns if c in net_known]
        A = S[np.ix_(rows, [col_idx[c] for c in missing])]
        rhs = -S[np.ix_(rows, [col_idx[c] for c in known_cols])] @ np.array(
            [net_known[c] for c in known_cols]
        )

        scale = max([1.0] + [abs(v) for v in net_known.values()])
        if A.shape[0] == 0:
            undetermined = list(missing)
            solution = np.zeros(len(missing))
        else:
            solution, *_ = np.linalg.lstsq(A, rhs, rcond=None)
            _, s, vt = np.linalg.svd(A)
            tol = max(A.shape) * np.finfo(float).eps * (s[0] if s.size else 0)
            null_space = vt[int((s > tol).sum()) :]
            undetermined = [
                c
                for i, c in enumerate(missing)
                if null_space.size and np.abs(null_space[:, i]).max() > 1e-8
            ]
            residual = np.abs(A @ solution - rhs).max() if rhs.size else 0.0
            if residual > 1e-6 * scale:
                warnings.warn(
                    f"The stored net fluxes do not balance (largest residual "
                    f"{residual:.3g}); derived fluxes are least-squares "
                    f"estimates.",
                    stacklevel=6,
                )

        if undetermined:
            warnings.warn(
                f"{len(undetermined)} net flux(es) have no stored value and "
                f"are not determined by the stored ones, so no value is "
                f"written for them: {undetermined}",
                stacklevel=6,
            )
        derived = {
            c: float(v)
            for c, v in zip(missing, solution)
            if c not in undetermined
        }

        irreversible = {r.id for r in reactions if not r.reversibility}
        irreversible.update(f"{m}_out" for m in self._drains)
        negative = {
            c: round(v, 6)
            for c, v in derived.items()
            if c in irreversible and v < -1e-9 * scale
        }
        if negative:
            raise ValueError(
                f"Derived net flux is negative on irreversible reaction(s) "
                f"{negative}: the stored flux values contradict the "
                f"reaction directions. Provide a complete, consistent flux "
                f"map."
            )
        return derived

    def _add_tracer_input(self, config: ET.Element, tracer) -> List[str]:
        """Add input element for tracer specification.

        Returns:
            Descriptions of label purities that could not be written.
        """
        input_elem = ET.SubElement(config, "input")
        input_elem.set("pool", tracer.metabolite)
        input_elem.set("type", "isotopomer")

        dropped = []
        for label in tracer.labels:
            label_elem = ET.SubElement(input_elem, "label")
            label_elem.set("cfg", label.labeled_pattern)
            purity = getattr(label, "purity", None)
            if self._natural_abundance:
                label_elem.set("purity", str(1.0 if purity is None else purity))
            elif purity is not None and purity < 1.0:
                dropped.append(
                    f"{tracer.metabolite}[{label.labeled_pattern}]={purity}"
                )
            label_elem.text = str(label.fraction)
        return dropped

    def _add_unlabelled_input(
        self, config: ET.Element, metabolite_id: str
    ) -> None:
        """Add an unlabelled isotopomer input for a source pool."""
        input_elem = ET.SubElement(config, "input")
        input_elem.set("pool", metabolite_id)
        input_elem.set("type", "isotopomer")
        label_elem = ET.SubElement(input_elem, "label")
        label_elem.set("cfg", "0" * self._atom_count(metabolite_id))
        if self._natural_abundance:
            label_elem.set("purity", "1.0")
        label_elem.text = "1.0"

    def _add_measurement(
        self, config: ET.Element, experiment: LabelingExperiments
    ) -> None:
        """Add measurement section to configuration."""
        meas_elem = ET.SubElement(config, "measurement")

        # Add mlabel with metadata
        mlabel = ET.SubElement(meas_elem, "mlabel")
        date_elem = ET.SubElement(mlabel, "date")
        date_elem.text = self._timestamp()

        # Check if this is time-course data
        if experiment.measurement.data and experiment.measurement.data.data:
            times = set()
            for datum in experiment.measurement.data.data:
                if datum.time is not None:
                    times.add(datum.time)
            if times:
                timeunit = ET.SubElement(mlabel, "timeunit")
                timeunit.text = "s"  # Default to seconds

        # Add model section
        model_elem = ET.SubElement(meas_elem, "model")

        # Add labeling measurements
        if experiment.measurement.model.labeling_measurement:
            labeling = ET.SubElement(model_elem, "labelingmeasurement")
            self._add_labeling_groups(
                labeling,
                experiment.measurement.model.labeling_measurement,
                experiment.measurement.data,
            )

        # Add flux measurements
        if experiment.measurement.model.flux_measurement:
            flux_meas = ET.SubElement(model_elem, "fluxmeasurement")
            self._add_flux_measurements(
                flux_meas, experiment.measurement.model.flux_measurement
            )

        # Add data section
        if experiment.measurement.data and experiment.measurement.data.data:
            data_elem = ET.SubElement(meas_elem, "data")

            # Add dlabel
            dlabel = ET.SubElement(data_elem, "dlabel")
            date_elem = ET.SubElement(dlabel, "date")
            date_elem.text = self._timestamp()

            # Add data points
            for datum in experiment.measurement.data.data:
                datum_elem = ET.SubElement(data_elem, "datum")
                datum_elem.set("id", datum.id.split(":")[0])  # Group ID
                datum_elem.set("stddev", str(datum.stddev))
                if datum.time is not None:
                    datum_elem.set("time", str(datum.time))
                # Weight is typically the isotopomer index
                if datum.weight is not None:
                    datum_elem.set("weight", str(datum.weight))
                elif "M" in datum.id:
                    weight = datum.id.split("M")[-1]
                    try:
                        datum_elem.set("weight", weight)
                    except (ValueError, TypeError):
                        pass
                datum_elem.text = str(datum.value)

            # Apply ¹³C natural abundance correction if requested
            if getattr(self, "_apply_na_correction", False):
                self._apply_na_to_data(data_elem)

    def _na_correction_matrix(self, n_C: int) -> "np.ndarray":
        """Binomial ¹³C natural-abundance correction matrix M_C.

        The matrix has size ``(n_C+1) × (n_C+1)``.
        ``M_C[i,j] = Binom(n_C-j, i-j, p)`` for ``i>=j``, else 0.
        Transforms a pure-tracer (no NA) MID vector into the expected
        measured (with NA) MID: ``mid_measured = M_C @ mid_tracer``.
        ``p = NATURAL_ABUNDANCE_CARBON = 0.01109`` (x3cflux constant).

        Returns:
            The natural-abundance correction matrix.
        """
        import numpy as np
        from scipy.stats import binom as sp_binom

        p = self.NATURAL_ABUNDANCE_CARBON
        n = n_C
        M = np.zeros((n + 1, n + 1))
        for j in range(n + 1):
            for extra in range(n - j + 1):
                i = j + extra
                M[i, j] = sp_binom.pmf(extra, n - j, p)
        return M

    def _apply_na_to_data(self, data_elem: ET.Element) -> None:
        """Re-write <datum> children of data_elem with ¹³C NA correction.

        Groups datums by group id, applies M_C to each group's
        (value, stddev) vectors, and rewrites the text / stddev
        attributes in place.  Only groups with integer weights (MS
        isotopologue data) are corrected.
        """
        import numpy as np

        datums = list(data_elem)
        # Group datums: {group_id: [(weight_int, elem), ...]}
        from collections import defaultdict

        groups: dict = defaultdict(list)
        non_weight: list = []
        for elem in datums:
            if elem.tag == "datum":
                try:
                    w = int(elem.get("weight", ""))
                    groups[elem.get("id", "")].append((w, elem))
                except (ValueError, TypeError):
                    non_weight.append(elem)

        for gid, pairs in groups.items():
            pairs_sorted = sorted(pairs, key=lambda x: x[0])
            weights = [w for w, _ in pairs_sorted]
            if weights != list(range(len(weights))):
                continue  # Non-sequential weights — skip (NMR, etc.)
            n_C = len(weights) - 1
            M = self._na_correction_matrix(n_C)
            vals = np.array([float(e.text or "0") for _, e in pairs_sorted])
            sds = np.array(
                [float(e.get("stddev", "0")) for _, e in pairs_sorted]
            )
            new_vals = M @ vals
            new_sds = np.sqrt((M**2) @ (sds**2))
            for i, (_, elem) in enumerate(pairs_sorted):
                elem.text = f"{new_vals[i]:.8e}"
                elem.set("stddev", f"{new_sds[i]:.8e}")

    def _add_labeling_groups(
        self, labeling: ET.Element, labeling_meas, data
    ) -> None:
        """Add labeling measurement groups."""
        # Collect time points from data
        times = set()
        if data and data.data:
            for datum in data.data:
                if datum.time is not None:
                    times.add(datum.time)

        times_str = ",".join(str(t) for t in sorted(times)) if times else "0.0"

        for group in labeling_meas.groups:
            group_elem = ET.SubElement(labeling, "group")
            group_elem.set("id", group.id)
            group_elem.set("scale", group.scale or "auto")
            if times:
                group_elem.set("times", times_str)

            # Add textual expression
            textual = ET.SubElement(group_elem, "textual")
            if group.expression:
                textual.text = group.expression.textual
            else:
                textual.text = group.id

    def _add_flux_measurements(
        self, flux_meas: ET.Element, flux_measurement
    ) -> None:
        """Add flux measurement elements."""
        for net_flux in flux_measurement.net_fluxes:
            netflux_elem = ET.SubElement(flux_meas, "netflux")
            netflux_elem.set("id", net_flux.id)

            textual = ET.SubElement(netflux_elem, "textual")
            if net_flux.expression:
                textual.text = net_flux.expression.textual
            else:
                textual.text = net_flux.id


def write_fluxml(
    model: FluxomicsData,
    filepath: str,
    apply_na_correction: bool = False,
    natural_abundance: bool = False,
    add_drains: bool = True,
) -> None:
    """Convenience function to write FluxomicsData to a FluxML file.

    Args:
        model: FluxomicsData to write
        filepath: Path to output file
        apply_na_correction: Apply ¹³C natural abundance
            correction (p=0.01109/C) to all MS measurement data.
            Use when data is synthetic (no NA present) and the
            target tool is x3cflux, together with
            ``natural_abundance=True``.
        natural_abundance: Write the ``purity`` attribute on every input
            label, which makes 13CFLUX give unlabelled input atoms their
            natural ¹³C abundance.  Default False: ideal inputs, no
            ``purity`` attribute.  See :meth:`FluxMLWriter.write`.
        add_drains: Add a ``<pool>_out`` reaction for every pool that is
            produced but never consumed (default True).
    """
    writer = FluxMLWriter()
    writer.write(
        model,
        filepath,
        apply_na_correction=apply_na_correction,
        natural_abundance=natural_abundance,
        add_drains=add_drains,
    )
