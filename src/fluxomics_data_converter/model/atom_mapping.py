"""atom transition functionality for fluxomics reactions.

This module provides classes and utilities for handling atom transitions in
metabolic reactions, including support for:
- Multiple mapping formats (letter notation, FluxML cfg strings, RDKit SMILES)
- Symmetric reactions with multiple equivalent mappings
- Isotopomer transformations
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field, ConfigDict
import re
import numpy as np


@dataclass(frozen=True)
class AtomAddress:
    """Address for a specific atom in a reaction.

    Attributes:
        mol: Molecule/compound ID
        index: Atom index within the molecule (1-based)
        element: Element type (e.g., 'C', 'N', 'O')
        instance: Instance number when multiple same compounds exist (1-based)
    """

    mol: str
    index: int
    element: Optional[str] = None
    instance: int = 1


class AtomMap(BaseModel):
    """A single atom transition for a reaction.

    Maps product atoms to their source reactant atoms.
    Keys and values are AtomAddress objects with 1-based indices.
    """

    # We need to use tuple representation for Pydantic compatibility
    mapping: Dict[AtomAddress, AtomAddress] = Field(
        default_factory=dict,
        description="Mapping from product atoms to source reactant atoms",
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    def add(
        self,
        product_cpd: str,
        product_atom: int,
        src_cpd: str,
        src_atom: int,
        atom_type: Optional[str] = "C",
        product_instance: int = 1,
        src_instance: int = 1,
    ) -> "AtomMap":
        """Add a mapping from product atom to source atom.

        Returns:
            A new ``AtomMap`` with the added mapping (immutable update).
        """
        # Create new dict to maintain immutability
        new_mapping = dict(self.mapping)
        new_mapping[
            AtomAddress(product_cpd, product_atom, atom_type, product_instance)
        ] = AtomAddress(src_cpd, src_atom, atom_type, src_instance)
        return AtomMap(mapping=new_mapping)

    def product_atoms(self) -> Dict[Tuple[str, int], int]:
        """Return the maximum atom index per product compound instance.

        Iterates over the mapping *keys* (which are product-side
        ``AtomAddress`` objects) and, for each ``(compound_id, instance)``
        pair, records the highest atom index seen.  The result therefore
        gives the number of labelling positions for every product compound
        instance that appears in this map.

        Returns:
            Dict mapping ``(compound_id, instance)`` to the highest
            (1-based) atom index found for that compound instance on the
            product side.

        Example::

            # AtomMap with two product atoms for compound "C"
            result = atom_map.product_atoms()
            # result == {("C", 1): 2}  # C has atoms at indices 1 and 2
        """
        result: Dict[Tuple[str, int], int] = {}
        for addr in self.mapping.keys():
            key = (addr.mol, addr.instance)
            result[key] = max(result.get(key, 0), addr.index)
        return result

    def reactant_atoms(self) -> Dict[Tuple[str, int], int]:
        """Return the maximum atom index per reactant compound instance.

        Iterates over the mapping *values* (which are reactant-side
        ``AtomAddress`` objects) and, for each ``(compound_id, instance)``
        pair, records the highest atom index seen.  The result therefore
        gives the number of labelling positions for every reactant compound
        instance that participates in this map.

        Returns:
            Dict mapping ``(compound_id, instance)`` to the highest
            (1-based) atom index found for that compound instance on the
            reactant side.

        Example::

            # AtomMap where reactant "A" contributes atoms 1-3
            result = atom_map.reactant_atoms()
            # result == {("A", 1): 3}
        """
        result: Dict[Tuple[str, int], int] = {}
        for addr in self.mapping.values():
            key = (addr.mol, addr.instance)
            result[key] = max(result.get(key, 0), addr.index)
        return result


class AtomTransition(BaseModel):
    """Complete atom transition information for a reaction.

    Supports multiple alternative mappings (variants/symmetries).
    Separated from Reaction class for cleaner design.
    """

    reaction_id: str = Field(
        description="Reaction identifier this mapping belongs to"
    )
    reactants: List[str] = Field(
        description="Ordered list of reactant IDs (needed for "
        "atom-level operations)"
    )
    products: List[str] = Field(
        description="Ordered list of product IDs (needed for "
        "atom-level operations)"
    )
    maps: Dict[str, AtomMap] = Field(
        default_factory=dict,
        description="Atom map variants keyed by atom_map_id "
        "(e.g., 'SCS___1' -> map)",
    )
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Weights for each mapping variant (defaults to uniform)",
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    def __str__(self) -> str:
        """String representation with summary metadatarmation.

        Returns:
            A summary string describing the atom transition.
        """
        return self.summary()

    def summary(self) -> str:
        """Return a summary of the atom transition.

        Returns:
            String with metadatarmation about the number of variants and
            their weights
        """
        num_maps = len(self.maps)

        if num_maps == 0:
            return f"AtomTransition(reaction_id='{self.reaction_id}', no maps)"

        if num_maps == 1:
            map_id = next(iter(self.maps.keys()))
            return (
                f"AtomTransition(reaction_id='{self.reaction_id}', "
                f"single map: {map_id})"
            )

        # Multiple variants
        lines = [
            f"AtomTransition(reaction_id='{self.reaction_id}', "
            f"{num_maps} variants):",
        ]

        for map_id in sorted(self.maps.keys()):
            weight_str = ""
            if self.weights and map_id in self.weights:
                weight_str = f" [weight: {self.weights[map_id]:.3f}]"
            lines.append(f"  - {map_id}{weight_str}")

        return "\n".join(lines)

    @staticmethod
    def parse_letter_notation(
        reactant_items: List[Tuple[str, str]],
        product_items: List[Tuple[str, str]],
    ) -> AtomMap:
        """Parse atom transitions from letter-notation format.

        In letter notation each compound is paired with a string of lowercase
        (or uppercase) letters.  Each unique letter is an atom label; the
        same letter appearing on a reactant and on a product means that atom
        is transferred directly between those positions.

        The bijection rule requires that every letter that appears in a
        product string must appear the same number of times across all
        reactant strings.

        Example::

            # Reaction: A(abc) + B(de) -> C(abcde)
            reactant_items = [("A", "abc"), ("B", "de")]
            product_items  = [("C", "abcde")]
            atom_map = AtomTransition.parse_letter_notation(
                reactant_items, product_items
            )
            # atom_map.mapping maps each product AtomAddress to its
            # reactant source:
            #   C@1 <- A@1 (letter 'a')
            #   C@2 <- A@2 (letter 'b')
            #   C@3 <- A@3 (letter 'c')
            #   C@4 <- B@1 (letter 'd')
            #   C@5 <- B@2 (letter 'e')

        Args:
            reactant_items: List of ``(compound_id, atom_string)`` pairs for
                each reactant that carries labelling.  Cofactors without
                labelling are omitted.  Repeated compound IDs are allowed and
                will be treated as separate instances (instance counter
                increments per compound).
            product_items: List of ``(compound_id, atom_string)`` pairs for
                each product.  Every letter in a product string must appear
                in the concatenated reactant atom strings (bijection).

        Returns:
            An :class:`AtomMap` where each product :class:`AtomAddress`
            maps to its source reactant :class:`AtomAddress`.  All atom
            indices are 1-based.

        Raises:
            ValueError: If a product atom label appears more times in the
                products than it does in the reactants, violating the
                bijection constraint.
        """
        # Build a global list of all atoms from all reactants
        # (cpd, atom_idx, instance)
        all_reactant_atoms: List[Tuple[str, int, int]] = []
        # Sequence of all letters from reactants
        letter_sequence: List[str] = []

        # Count instances for each compound on reactant side
        reactant_instance_counts: Dict[str, int] = {}
        for cpd, letters in reactant_items:
            reactant_instance_counts[cpd] = (
                reactant_instance_counts.get(cpd, 0) + 1
            )
            instance = reactant_instance_counts[cpd]
            for i, ch in enumerate(letters, start=1):
                all_reactant_atoms.append((cpd, i, instance))
                letter_sequence.append(ch)

        # Build mapping for products
        atom_map = AtomMap()
        letter_usage_count: Dict[str, int] = {}

        # Count instances for each compound on product side
        product_instance_counts: Dict[str, int] = {}
        for cpd, letters in product_items:
            product_instance_counts[cpd] = (
                product_instance_counts.get(cpd, 0) + 1
            )
            instance = product_instance_counts[cpd]
            for i, ch in enumerate(letters, start=1):
                # Count how many times we've used this letter
                usage_count = letter_usage_count.get(ch, 0)

                # Find the nth occurrence of this letter in the sequence
                occurrences = [
                    idx
                    for idx, letter in enumerate(letter_sequence)
                    if letter == ch
                ]

                if usage_count >= len(occurrences):
                    raise ValueError(
                        f"Letter '{ch}' used more times in products "
                        f"than available in reactants. Available: "
                        f"{len(occurrences)}, Used: {usage_count + 1}"
                    )

                atom_idx = occurrences[usage_count]
                src_cpd, src_atom, src_inst = all_reactant_atoms[atom_idx]

                atom_map = atom_map.add(
                    cpd, i, src_cpd, src_atom, None, instance, src_inst
                )

                letter_usage_count[ch] = usage_count + 1

        return atom_map

    def to_letter_notation(self, atom_map_id: Optional[str] = None) -> str:
        r"""Convert atom transition to letter notation string.

        Args:
            atom_map_id: Optional atom map ID to use. If None, shows
                all variants or uses the single map if only one exists.

        Returns a string representation of the atom transition in letter
        notation format. If multiple elements are present, they are shown
        on separate lines. If multiple mapping variants exist and no atom_map_id
        is specified, shows all variants.

        Example: ``"A(abc) + B(de) -> C(abcde)"``

        With multiple elements: ``"C: A(ab) -> B(ab)\nN: A(c) -> B(c)"``

        Returns:
            Letter notation string representation of the mapping(s).

        Raises:
            KeyError: If *atom_map_id* is not found in ``self.maps``.
        """
        from collections import defaultdict

        if not self.maps:
            return ""

        # If specific atom_map_id is requested
        if atom_map_id is not None:
            if atom_map_id not in self.maps:
                raise KeyError(f"Atom map ID '{atom_map_id}' not found")
            atom_map = self.maps[atom_map_id]
        # Handle multiple mapping variants
        elif len(self.maps) > 1:
            # For multiple variants, show each on a separate line
            results = []
            for map_id, atom_map in self.maps.items():
                weight_str = (
                    f" [{self.weights[map_id]:.3f}]"
                    if self.weights and map_id in self.weights
                    else ""
                )
                # Create temporary AtomTransition for single map recursion
                temp_mapping = AtomTransition(
                    reaction_id=self.reaction_id,
                    reactants=self.reactants,
                    products=self.products,
                    maps={map_id: atom_map},
                )
                result = temp_mapping.to_letter_notation(atom_map_id=map_id)
                if result:
                    results.append(f"{map_id}{weight_str}: {result}")
            return "\n".join(results)
        else:
            # Single mapping
            atom_map = next(iter(self.maps.values()))
        if not atom_map.mapping:
            return ""

        # Group mappings by element type
        element_groups = defaultdict(list)
        for prod_addr, react_addr in atom_map.mapping.items():
            element = prod_addr.element or "C"
            element_groups[element].append((prod_addr, react_addr))

        # Process each element separately
        element_results = []

        for element in sorted(element_groups.keys()):
            mappings = element_groups[element]

            # Collect compounds and their atoms for this element
            reactant_atoms = defaultdict(lambda: defaultdict(list))
            product_atoms = defaultdict(lambda: defaultdict(list))

            # Assign letters sequentially based on reactant order
            letters = (
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )
            letter_idx = 0
            atom_to_letter = {}

            # Sort mappings by reactant position for consistent
            # letter assignment
            sorted_mappings = sorted(
                mappings, key=lambda x: (x[1].mol, x[1].instance, x[1].index)
            )

            for prod_addr, react_addr in sorted_mappings:
                key = (react_addr.mol, react_addr.instance, react_addr.index)
                if key not in atom_to_letter:
                    if letter_idx >= len(letters):
                        # Generate extended letters like aa, ab, ac...
                        div, mod = divmod(letter_idx - len(letters), 26)
                        atom_to_letter[key] = letters[div % 26] + letters[mod]
                    else:
                        atom_to_letter[key] = letters[letter_idx]
                    letter_idx += 1

                letter = atom_to_letter[key]
                reactant_atoms[react_addr.mol][react_addr.instance].append(
                    (react_addr.index, letter)
                )
                product_atoms[prod_addr.mol][prod_addr.instance].append(
                    (prod_addr.index, letter)
                )

            # Build reaction string for this element
            def build_side(atoms_dict):
                """Build one side of the reaction string.

                Returns:
                    A list of ``compound(letters)`` string parts.
                """
                parts = []
                for cpd in sorted(atoms_dict.keys()):
                    instances = atoms_dict[cpd]
                    for instance in sorted(instances.keys()):
                        atoms = sorted(instances[instance])
                        letter_str = "".join(ltr for _, ltr in atoms)
                        parts.append(f"{cpd}({letter_str})")
                return parts

            reactant_parts = build_side(reactant_atoms)
            product_parts = build_side(product_atoms)

            if reactant_parts and product_parts:
                reaction = (
                    f"{' + '.join(reactant_parts)} -> "
                    f"{' + '.join(product_parts)}"
                )
                # Only add element prefix if there are multiple elements
                if len(element_groups) > 1:
                    reaction = f"{element}: {reaction}"
                element_results.append(reaction)

        # Join results
        if len(element_results) == 0:
            return ""
        elif len(element_results) == 1:
            return element_results[0]
        else:
            return "\n".join(element_results)

    @staticmethod
    def parse_fluxml_cfg(
        reactant_cfgs: Union[
            Dict[str, str], Sequence[Tuple[str, Optional[str]]]
        ],
        product_cfgs: List[Tuple[str, str]],
        reactant_order: List[str],
    ) -> AtomMap:
        """Parse atom transitions from the FluxML ``C#k@r`` cfg format.

        FluxML encodes atom transitions in product cfg strings using tokens of
        the form ``Element#atom_index@reactant_position``.  For example, the
        token ``C#3@2`` means "this product atom comes from carbon atom 3 of
        the 2nd reactant compound listed in *reactant_order*".

        Tokens are symbols, as 13CFLUX reads them: a product token is
        resolved to wherever the same token appears in the reactant cfgs.
        With the usual identity reactant cfgs (``C#1@1 C#2@1 ...``) that is
        the literal reading above; a reactant ``<variant>`` that permutes
        its tokens (``C#4@1 C#3@1 C#2@1 C#1@1``) moves the source atoms
        accordingly.  A token that appears in no reactant cfg falls back to
        the literal reading.

        Example (aldolase FBP → GAP + DHAP from the FluxML spec)::

            reactant_order = ["FBP"]
            reactant_cfgs  = {"FBP": "C#1@1 C#2@1 C#3@1 C#4@1 C#5@1 C#6@1"}
            product_cfgs   = [
                ("GAP",  "C#4@1 C#1@1 C#3@1"),
                ("DHAP", "C#5@1 C#2@1 C#6@1"),
            ]
            atom_map = AtomTransition.parse_fluxml_cfg(
                reactant_cfgs, product_cfgs, reactant_order
            )
            # GAP atom 1 ← FBP atom 4
            # GAP atom 2 ← FBP atom 1
            # GAP atom 3 ← FBP atom 3
            # DHAP atom 1 ← FBP atom 5
            # ...

        Token grammar::

            token ::= element "#" source_atom_index "@" reactant_position
            element             ::= [A-Z]           (e.g. "C", "N")
            source_atom_index   ::= integer >= 1    (1-based atom number)
            reactant_position   ::= integer >= 1    (1-based index into
                                                     reactant_order)

        Args:
            reactant_cfgs: The reactant cfg strings, either as a list of
                ``(reactant_id, cfg_or_None)`` pairs aligned with
                *reactant_order* (needed when a compound appears twice), or
                as a mapping ``reactant_id -> cfg_string``.
            product_cfgs: Ordered list of ``(product_id, cfg_string)`` pairs.
                Products are processed in order; repeated compound IDs create
                separate instances.
            reactant_order: Ordered list of reactant IDs corresponding to the
                ``@r`` positions in the product tokens.  Repeated compound IDs
                are allowed and increment the instance counter for that
                compound.

        Returns:
            An :class:`AtomMap` where each product :class:`AtomAddress`
            (1-based) maps to its source reactant :class:`AtomAddress`.

        Raises:
            ValueError: If a product cfg token does not match the
                ``Element#k@r`` pattern.
            IndexError: If the ``@r`` reactant position is out of range for
                *reactant_order*.
        """
        atom_map = AtomMap()

        # Build instance mapping: position -> (compound, instance)
        # This handles repeated compounds in reactant_order
        position_to_instance = {}
        compound_counts = {}
        for idx, cpd in enumerate(reactant_order):
            compound_counts[cpd] = compound_counts.get(cpd, 0) + 1
            position_to_instance[idx] = (cpd, compound_counts[cpd])

        # Where each reactant token sits: token -> (position, atom index)
        token_source: Dict[str, Tuple[int, int]] = {}
        for idx, cpd in enumerate(reactant_order):
            if isinstance(reactant_cfgs, dict):
                reactant_cfg = reactant_cfgs.get(cpd)
            elif idx < len(reactant_cfgs) and reactant_cfgs[idx][0] == cpd:
                reactant_cfg = reactant_cfgs[idx][1]
            else:
                reactant_cfg = None
            for k, tok in enumerate((reactant_cfg or "").split(), start=1):
                token_source[tok] = (idx, k)

        # Track product instances
        product_instances = {}
        product_atom_indices = {}

        # Process product cfgs
        for pcid, cfg in product_cfgs:
            # Track instance number for this product
            product_instances[pcid] = product_instances.get(pcid, 0) + 1
            prod_instance = product_instances[pcid]

            # Track atom index for this product instance
            key = (pcid, prod_instance)
            if key not in product_atom_indices:
                product_atom_indices[key] = 0

            for tok in cfg.split():
                m = re.fullmatch(r"([A-Z])#(\d+)@(\d+)", tok)
                if not m:
                    raise ValueError(f"Bad product cfg token: {tok}")
                atom_type = m.group(1)
                if tok in token_source:
                    src_reactant_idx, src_atom = token_source[tok]
                else:
                    src_atom = int(m.group(2))
                    src_reactant_idx = int(m.group(3)) - 1  # to 0-based

                if src_reactant_idx not in position_to_instance:
                    raise IndexError(
                        f"Reactant index @{src_reactant_idx + 1} out of range"
                    )

                src_cpd, src_instance = position_to_instance[src_reactant_idx]
                product_atom_indices[key] += 1
                prod_atom_idx = product_atom_indices[key]

                atom_map = atom_map.add(
                    pcid,
                    prod_atom_idx,
                    src_cpd,
                    src_atom,
                    atom_type,
                    prod_instance,
                    src_instance,
                )

        return atom_map

    def to_fluxml_string(
        self,
        atom_map_id: Optional[str] = None,
    ) -> Optional[str]:
        """Convert to FluxML-style atom transition string.

        Args:
            atom_map_id: Optional atom map ID to use. If None and only
                one map exists, uses that map. If multiple maps exist,
                must specify atom_map_id.

        Returns:
            FluxML-style string representation of the mapping

        Raises:
            KeyError: If *atom_map_id* is not found in ``self.maps``.
            ValueError: If multiple maps exist and no *atom_map_id* is
                specified, or if a source compound is not found in the
                reactant list.
        """
        if not self.maps:
            return None

        # Select which atom map to use
        if atom_map_id is not None:
            if atom_map_id not in self.maps:
                raise KeyError(f"Atom map ID '{atom_map_id}' not found")
            atom_map = self.maps[atom_map_id]
        elif len(self.maps) == 1:
            atom_map = next(iter(self.maps.values()))
        else:
            raise ValueError(
                "Multiple atom maps exist. Must specify atom_map_id."
            )

        # Use stored reactants and products for order
        reactants = self.reactants
        products = self.products

        # Build mapping from (compound, instance) to reactant position (1-based)
        reactant_position = {}
        instance_counts: Dict[str, int] = {}
        for pos, cpd in enumerate(reactants, start=1):
            instance_counts[cpd] = instance_counts.get(cpd, 0) + 1
            instance = instance_counts[cpd]
            reactant_position[(cpd, instance)] = pos

        # Build reactant and product strings
        reactant_parts = []
        product_parts = []

        # Group mappings by compound and instance
        reactant_instances: Dict[str, set] = {}
        for addr in atom_map.mapping.values():
            if addr.mol not in reactant_instances:
                reactant_instances[addr.mol] = set()
            reactant_instances[addr.mol].add(addr.instance)

        for cpd in reactants:
            if cpd in reactant_instances:
                for instance in sorted(reactant_instances[cpd]):
                    atoms = []
                    for addr in atom_map.mapping.values():
                        if addr.mol == cpd and addr.instance == instance:
                            atoms.append((addr.index, addr.element or "C"))
                    if atoms:
                        atom_str = " ".join(
                            f"{at}#{idx}" for idx, at in sorted(atoms)
                        )
                        num_inst = len(reactant_instances[cpd])
                        prefix = f"{num_inst}" if num_inst > 1 else ""
                        reactant_parts.append(f"{prefix}{cpd}({atom_str})")

        product_instances: Dict[str, set] = {}
        for addr in atom_map.mapping.keys():
            if addr.mol not in product_instances:
                product_instances[addr.mol] = set()
            product_instances[addr.mol].add(addr.instance)

        for cpd in products:
            if cpd in product_instances:
                for instance in sorted(product_instances[cpd]):
                    atoms = []
                    for prod_addr in atom_map.mapping.keys():
                        if (
                            prod_addr.mol == cpd
                            and prod_addr.instance == instance
                        ):
                            src_addr = atom_map.mapping[prod_addr]
                            # Get the reactant position for this source atom
                            src_pos = reactant_position.get(
                                (src_addr.mol, src_addr.instance)
                            )
                            if src_pos is None:
                                raise ValueError(
                                    f"Source compound {src_addr.mol} "
                                    f"instance {src_addr.instance} "
                                    f"not found in reactant list"
                                )
                            atoms.append(
                                (
                                    prod_addr.index,
                                    prod_addr.element or "C",
                                    src_addr.index,
                                    src_pos,
                                )
                            )
                    if atoms:
                        # Format as "element#source_atom@reactant_position"
                        atom_str = " ".join(
                            f"{at}#{src_idx}@{src_pos}"
                            for pa, at, src_idx, src_pos in sorted(atoms)
                        )
                        num_inst = len(product_instances[cpd])
                        prefix = f"{num_inst}" if num_inst > 1 else ""
                        product_parts.append(f"{prefix}{cpd}({atom_str})")

        if not reactant_parts or not product_parts:
            return None

        reactant_side = " + ".join(reactant_parts)
        product_side = " + ".join(product_parts)
        return f'"{reactant_side} => {product_side}"'

    def merge_symmetric_mappings(
        self, other: "AtomTransition"
    ) -> "AtomTransition":
        """Merge with another mapping to handle symmetric reactions.

        Args:
            other: Another AtomTransition to merge with (must have same
                reaction_id)

        Returns:
            New AtomTransition with combined maps

        Raises:
            ValueError: If the two transitions belong to different reactions.
        """
        if self.reaction_id != other.reaction_id:
            raise ValueError("Can only merge mappings for the same reaction")

        # Combine maps with unique IDs
        new_maps = dict(self.maps)
        # Add suffix to other map IDs if there are conflicts
        for map_id, atom_map in other.maps.items():
            if map_id in new_maps:
                # Add suffix to avoid conflict
                counter = 1
                new_id = f"{map_id}_alt{counter}"
                while new_id in new_maps:
                    counter += 1
                    new_id = f"{map_id}_alt{counter}"
                new_maps[new_id] = atom_map
            else:
                new_maps[map_id] = atom_map

        # If weights exist, combine them proportionally
        new_weights = None
        if self.weights and other.weights:
            total = len(self.maps) + len(other.maps)
            self_weight = len(self.maps) / total
            other_weight = len(other.maps) / total
            new_weights = {}
            for map_id, w in self.weights.items():
                new_weights[map_id] = w * self_weight
            for map_id, w in other.weights.items():
                # Use the potentially renamed key
                actual_key = (
                    map_id if map_id not in self.maps else f"{map_id}_alt1"
                )
                new_weights[actual_key] = w * other_weight

        return AtomTransition(
            reaction_id=self.reaction_id,
            reactants=self.reactants,
            products=self.products,
            maps=new_maps,
            weights=new_weights,
        )

    def transform_isotopomers(
        self,
        reactant_distributions: Dict[Tuple[str, int], np.ndarray],
        reactant_order: Optional[List[str]] = None,
        product_order: Optional[List[str]] = None,
        atom_map_id: Optional[str] = None,
    ) -> Dict[Tuple[str, int], np.ndarray]:
        """Propagate isotopomer distributions through a single reaction mapping.

        Each isotopomer distribution is a dense 1-D numpy array of length
        ``2**n_atoms``.  Index *i* encodes a labelling pattern as a binary
        integer: bit *k* equals 1 if atom position *k+1* carries a heavy
        isotope label (e.g. ¹³C), and 0 otherwise.  Values must sum to 1.

        Algorithm
        ---------
        1. **Joint distribution** — the individual reactant vectors are
           combined into a single joint distribution by taking successive
           Kronecker products.  This assumes statistical independence between
           reactant metabolites (standard assumption in ¹³C MFA).
        2. **Bit permutation** — for each index in the joint distribution,
           the bits corresponding to atoms that map to a given product are
           extracted and re-packed according to the atom map, accumulating
           probability into the product distribution via ``numpy.add.at``.
        3. **Splitting** — the full concatenated product distribution is
           sliced into per-product-compound arrays.

        Example::

            # Simple 1-carbon transfer: A(a) -> B(a)
            reactant_dist = {("A", 1): np.array([0.5, 0.5])}  # 50% labeled
            result = mapping.transform_isotopomers(reactant_dist)
            # result == {("B", 1): array([0.5, 0.5])}

        Args:
            reactant_distributions: Mapping from ``(compound_id, instance)``
                to a 1-D numpy array of isotopomer fractions of length
                ``2**n_atoms``.  Every reactant compound that appears in
                *reactant_order* must have an entry here.
            reactant_order: Ordered list of reactant IDs to process (with
                repeated entries for multiple instances of the same compound).
                Defaults to ``self.reactants``.
            product_order: Ordered list of product IDs.
                Defaults to ``self.products``.
            atom_map_id: Key into ``self.maps`` selecting a specific variant.
                Must be provided when ``self.maps`` contains more than one
                entry; if there is exactly one map and this is ``None``, that
                map is used automatically.

        Returns:
            Mapping from ``(compound_id, instance)`` to a 1-D numpy array
            of product isotopomer fractions.

        Raises:
            KeyError: If *atom_map_id* is not found in ``self.maps``.
            ValueError: If *atom_map_id* is ``None`` but multiple maps exist,
                or if a required reactant distribution is missing, or if the
                distribution length is not a power of two.
        """
        # Use stored order if not provided
        if reactant_order is None:
            reactant_order = self.reactants
        if product_order is None:
            product_order = self.products

        # Select which atom map to use
        if atom_map_id is not None:
            if atom_map_id not in self.maps:
                raise KeyError(f"Atom map ID '{atom_map_id}' not found")
            atom_map = self.maps[atom_map_id]
        elif len(self.maps) == 1:
            atom_map = next(iter(self.maps.values()))
        else:
            raise ValueError(
                "Multiple atom maps exist. Must specify atom_map_id."
            )

        # Build concatenated reactant atom order with instances
        src_atoms: List[Tuple[str, int, str, int]] = []
        # Count instances per compound
        reactant_instances: Dict[str, int] = {}
        for cpd in reactant_order:
            reactant_instances[cpd] = reactant_instances.get(cpd, 0) + 1
            instance = reactant_instances[cpd]
            key = (cpd, instance)
            if key not in reactant_distributions:
                raise ValueError(f"Missing distribution for {key}")
            n_atoms = len(reactant_distributions[key]).bit_length() - 1
            for i in range(1, n_atoms + 1):
                src_atoms.append((cpd, i, "C", instance))

        # Build product atom order with instances
        prod_atoms: List[Tuple[str, int, str, int]] = []
        product_instances: Dict[str, int] = {}
        for cpd in product_order:
            product_instances[cpd] = product_instances.get(cpd, 0) + 1
            instance = product_instances[cpd]
            # Find max atom index for this product instance
            max_idx = 0
            for prod_addr in atom_map.mapping.keys():
                if prod_addr.mol == cpd and prod_addr.instance == instance:
                    max_idx = max(max_idx, prod_addr.index)
            for i in range(1, max_idx + 1):
                prod_atoms.append((cpd, i, "C", instance))

        # Create mapping from product atom positions to source positions
        src_pos_lookup = {atom: i for i, atom in enumerate(src_atoms)}
        src_positions = []
        for prod_atom_tuple in prod_atoms:
            # Convert tuple to AtomAddress
            prod_atom = AtomAddress(*prod_atom_tuple)
            if prod_atom not in atom_map.mapping:
                raise ValueError(f"Product atom {prod_atom} not in mapping")
            src_atom = atom_map.mapping[prod_atom]
            # Convert src_atom back to tuple for lookup
            src_atom_tuple = (
                src_atom.mol,
                src_atom.index,
                src_atom.element,
                src_atom.instance,
            )
            src_positions.append(src_pos_lookup[src_atom_tuple])

        # Build joint reactant distribution
        joint_dist = np.array([1.0])
        instance_counts: Dict[str, int] = {}
        for cpd in reactant_order:
            instance_counts[cpd] = instance_counts.get(cpd, 0) + 1
            instance = instance_counts[cpd]
            key = (cpd, instance)
            joint_dist = np.kron(joint_dist, reactant_distributions[key])

        # Transform to product distribution
        n_src = len(src_atoms)
        n_prod = len(prod_atoms)

        if joint_dist.size != (1 << n_src):
            raise ValueError("Reactant distribution size mismatch")

        # Compute product distribution
        prod_dist = np.zeros(1 << n_prod)
        src_indices = np.arange(1 << n_src, dtype=np.uint32)

        # Map bits from source to product positions
        prod_indices = np.zeros_like(src_indices)
        for k, src_pos in enumerate(src_positions):
            bit = (src_indices >> src_pos) & 1
            prod_indices |= bit << k

        np.add.at(prod_dist, prod_indices, joint_dist)

        # Split into per-product distributions
        result = {}
        offset = 0
        product_instance_counts: Dict[str, int] = {}
        for cpd in product_order:
            product_instance_counts[cpd] = (
                product_instance_counts.get(cpd, 0) + 1
            )
            instance = product_instance_counts[cpd]
            n_atoms = sum(
                1
                for (c, _, _, inst) in prod_atoms
                if c == cpd and inst == instance
            )
            if n_atoms > 0:
                size = 1 << n_atoms
                result[(cpd, instance)] = prod_dist[offset : offset + size]
                offset += size

        return result

    def transform_with_symmetry(
        self,
        reactant_distributions: Dict[Tuple[str, int], np.ndarray],
        reactant_order: Optional[List[str]] = None,
        product_order: Optional[List[str]] = None,
    ) -> Dict[Tuple[str, int], np.ndarray]:
        """Propagate isotopomer distributions by weighted average over variants.

        For reactions with symmetric substrates (e.g. succinate, which is
        geometrically symmetric and can bind in two orientations),
        :class:`AtomTransition` stores multiple atom maps in ``self.maps``.
        This method calls :meth:`transform_isotopomers` once per variant and
        returns the weighted sum.

        If ``self.weights`` is ``None``, all variants receive equal weight
        ``1 / n_variants``.

        Example::

            # Symmetric reaction with two equal variants
            result = mapping.transform_with_symmetry(reactant_dists)
            # Equivalent to:
            #   0.5 * transform_isotopomers(..., atom_map_id="id___1")
            # + 0.5 * transform_isotopomers(..., atom_map_id="id___2")

        Args:
            reactant_distributions: Mapping from ``(compound_id, instance)``
                to a 1-D numpy array of isotopomer fractions.  See
                :meth:`transform_isotopomers` for the encoding convention.
            reactant_order: Ordered reactant IDs.  Defaults to
                ``self.reactants``.
            product_order: Ordered product IDs.  Defaults to
                ``self.products``.

        Returns:
            Mapping from ``(compound_id, instance)`` to a 1-D numpy array
            of product isotopomer fractions (weighted sum over all variants).

        Raises:
            ValueError: If ``self.maps`` is empty.
        """
        if not self.maps:
            raise ValueError("No atom maps available")

        # Use stored order if not provided
        if reactant_order is None:
            reactant_order = self.reactants
        if product_order is None:
            product_order = self.products

        # Prepare weights
        if self.weights:
            weights = self.weights
        else:
            uniform_weight = 1.0 / len(self.maps)
            weights = {map_id: uniform_weight for map_id in self.maps.keys()}

        result: Dict[Tuple[str, int], np.ndarray] = {}
        for map_id, weight in weights.items():
            prod_dists = self.transform_isotopomers(
                reactant_distributions,
                reactant_order,
                product_order,
                atom_map_id=map_id,
            )
            for key, dist in prod_dists.items():
                if key not in result:
                    result[key] = weight * dist
                else:
                    result[key] += weight * dist

        return result
