"""Freeflux tabular format parser for tsv/csv/xlsx files.

Freeflux is a Python package for 13C metabolic flux analysis that uses
tabular input files. This parser supports the following file types:
    - reactions: Network definition with atom transitions
    - fluxes: Flux values (simulation/reference)
    - concentrations: Metabolite pool sizes
    - label_input: Tracer labeling strategy (substrate + pattern + fraction
      + purity)
    - flux_bounds / constraints: Per-reaction or global flux bounds [lo, hi]
    - measured_MDVs: Steady-state mass distribution vector measurements
    - measured_fluxes: Measured flux values with uncertainties
    - measured_inst_MDVs: Time-course MDV measurements

Supports .tsv, .csv, and .xlsx file formats.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import pandas as pd

from ..core.core import (
    FluxomicsData,
    Metadata,
    MetabolicNetworkModel,
    LabelingExperiments,
)
from ..core.common import DictList, TextualOrMath, ErrorModel
from ..model.metabolite import Metabolite
from ..model.reaction import Reaction
from ..model.atom_mapping import AtomTransition
from ..experiment.measurement import (
    Measurement,
    MeasurementModel,
    MeasurementData,
    LabelingMeasurement,
    FluxMeasurement,
    Group,
    NetFlux,
    Datum,
)
from ..output.simulation import (
    Simulation,
    Variables,
    FluxValue,
    MetaboliteSizeValue,
)
from ..experiment.tracer import Tracers, LabelComposition
from ..model.constraint import Constraints, NetConstraints, ConstraintFormula


class FreefluxParser:
    """Parser for Freeflux tabular format files (tsv/csv/xlsx).

    Freeflux uses a set of tabular files stored in a directory.  All files
    in the set are optional except ``reactions.*``::

        reactions.*          — network definition with atom transitions
                               (required)
        fluxes.*             — flux values (simulation / reference)
        concentrations.*     — metabolite pool sizes
        measured_MDVs.*      — steady-state mass distribution vectors
        measured_fluxes.*    — measured flux values with uncertainties
        measured_inst_MDVs.* — time-course MDV measurements

    Supported file extensions: ``.tsv``, ``.csv``, ``.xlsx``.

    Example::

        parser = FreefluxParser()
        model = parser.parse("data/ecoli/")         # directory
        model = parser.parse("data/ecoli/reactions.tsv")
        # or the reactions file

    atom transition notation
    ---------------------
    Compounds are written as ``Name(atoms)`` where *atoms* is a string of
    lowercase letters.  Symmetric compounds (where two or more mappings are
    equally valid) are expressed as comma-separated alternatives, e.g.
    ``SUCC(abcd,dcba)``.  The parser generates the Cartesian product of all
    variant combinations and creates a multi-map
    :class:`~fluxomics_data_converter.AtomTransition`.
    """

    # File patterns for each data type
    FILE_PATTERNS = {
        "reactions": ["reactions", "reaction"],
        "fluxes": ["fluxes", "flux"],
        "concentrations": ["concentrations", "concentration"],
        "label_input": ["label_input", "label_inputs"],
        "flux_bounds": [
            "flux_bounds",
            "flux_bound",
            "constraints",
            "constraint",
        ],
        "measured_MDVs": [
            "measured_MDVs",
            "measured_MDV",
            "measured_MID",
            "measured_MIDs",
        ],
        "measured_fluxes": ["measured_fluxes", "measured_flux"],
        "measured_inst_MDVs": [
            "measured_inst_MDVs",
            "measured_inst_MDV",
            "measured_inst_MID",
            "measured_inst_MIDs",
        ],
    }

    # Supported file extensions
    EXTENSIONS = [".tsv", ".csv", ".xlsx"]

    def __init__(self):
        """Initialise parser state; call parse() to read files."""
        self._metabolites: Dict[str, Metabolite] = {}
        self._reactions: Dict[str, Reaction] = {}
        self._atom_mappings: Dict[str, AtomTransition] = {}

    def parse(self, base_path: str) -> FluxomicsData:
        """Parse Freeflux files from a directory.

        Args:
            base_path: Path to directory containing Freeflux files.

        Returns:
            FluxomicsData containing the parsed data

        Raises:
            FileNotFoundError: If the reactions file is not found in the
                given directory.
        """
        base_path = Path(base_path)

        # If given a file, use its parent directory
        if base_path.is_file():
            base_path = base_path.parent

        # Find and parse reactions file (required)
        reactions_path = self._find_file(base_path, "reactions")
        if not reactions_path:
            raise FileNotFoundError(f"Reactions file not found in: {base_path}")

        self._parse_reactions(reactions_path)

        # Build Model from parsed network
        model = MetabolicNetworkModel(
            metabolites=DictList(list(self._metabolites.values())),
            reactions=DictList(list(self._reactions.values())),
            atom_mappings=self._atom_mappings,
        )

        # Parse optional files
        fluxes = self._parse_fluxes(base_path)
        concentrations = self._parse_concentrations(base_path)
        tracers = self._parse_label_input(base_path)
        constraints = self._parse_flux_bounds(base_path)
        measurement = self._parse_measurements(base_path)

        # Create experiment if we have any experimental data
        experiments = []
        simulation = None

        # Build simulation from fluxes and concentrations
        if fluxes or concentrations:
            flux_values = fluxes or []
            metabolitesize_values = concentrations or []
            simulation = Simulation(
                variables=Variables(
                    flux_values=flux_values,
                    metabolitesize_values=metabolitesize_values,
                ),
            )

        if measurement or simulation or tracers:
            # Determine if we have inst_MDVs (non-stationary)
            has_inst_mdvs = (
                self._find_file(base_path, "measured_inst_MDVs") is not None
            )

            experiment = LabelingExperiments(
                name=base_path.name,
                stationary=not has_inst_mdvs,
                tracers=tracers,
                measurement=measurement,
                simulation=simulation,
            )
            experiments.append(experiment)

        # Create metadata
        metadata = Metadata(
            name=base_path.name,
            comment=f"Imported from Freeflux format: {base_path.name}",
        )

        return FluxomicsData(
            model=model,
            metadata=metadata,
            experiments=experiments,
            constraints=constraints,
        )

    def _find_file(self, base_path: Path, file_type: str) -> Optional[Path]:
        """Find a file matching the pattern with any supported extension.

        Returns:
            Path to the first matching file, or None if not found.
        """
        patterns = self.FILE_PATTERNS[file_type]
        if isinstance(patterns, str):
            patterns = [patterns]
        for pattern in patterns:
            for ext in self.EXTENSIONS:
                file_path = base_path / f"{pattern}{ext}"
                if file_path.exists():
                    return file_path
        return None

    def _read_tabular(self, file_path: Path) -> pd.DataFrame:
        """Read tabular file (tsv/csv/xlsx) into DataFrame.

        Returns:
            DataFrame with cleaned column names (leading # removed).
        """
        suffix = file_path.suffix.lower()

        if suffix == ".xlsx":
            df = pd.read_excel(file_path)
        elif suffix == ".csv":
            df = pd.read_csv(file_path)
        else:  # .tsv
            df = pd.read_csv(file_path, sep="\t")

        # Clean column names - remove leading # and strip whitespace
        df.columns = [
            col.lstrip("#").strip() if isinstance(col, str) else col
            for col in df.columns
        ]

        return df

    def _parse_reactions(self, filepath: Path) -> None:
        """Parse reactions file with reaction network and atom transitions.

        Format:
        - #reaction_ID: Reaction identifier (section headers start with #)
        - substrate_IDs(atom) or reactant_IDs(atom): Substrates with
          atom transition
        - product_IDs(atom): Products with atom transition
        - reversibility: 0 (irreversible) or 1 (reversible)
        """
        self._metabolites = {}
        self._reactions = {}
        self._atom_mappings = {}

        df = self._read_tabular(filepath)

        # Standardize column names
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if "reaction" in col_lower and "id" in col_lower:
                col_map["reaction_id"] = col
            elif "substrate" in col_lower or "reactant" in col_lower:
                col_map["substrates"] = col
            elif "product" in col_lower:
                col_map["products"] = col
            elif "reversib" in col_lower:
                col_map["reversibility"] = col

        for _, row in df.iterrows():
            rxn_id = str(row.get(col_map.get("reaction_id", ""), "")).strip()

            # Skip empty rows or section headers (start with #)
            if (
                not rxn_id
                or rxn_id.startswith("#")
                or pd.isna(row.get(col_map.get("reaction_id", "")))
            ):
                continue

            substrates_str = str(row.get(col_map.get("substrates", ""), ""))
            products_str = str(row.get(col_map.get("products", ""), ""))
            rev_value = row.get(col_map.get("reversibility", ""), 0)

            # Skip if no substrates
            if (
                pd.isna(substrates_str)
                or substrates_str == "nan"
                or not substrates_str.strip()
            ):
                continue

            # Handle empty products (sink reactions)
            if (
                pd.isna(products_str)
                or products_str == "nan"
                or not products_str.strip()
            ):
                products_str = ""

            # Parse reversibility
            reversible = False
            if not pd.isna(rev_value):
                if isinstance(rev_value, (int, float)):
                    reversible = int(rev_value) == 1
                elif isinstance(rev_value, str):
                    reversible = rev_value.strip().lower() in (
                        "1",
                        "true",
                        "yes",
                    )

            # Parse substrates and products with atom transitions
            reactants, reactant_atoms = self._parse_compounds(substrates_str)
            products, product_atoms = self._parse_compounds(products_str)

            # Create metabolites if not already defined
            for cpd_id in reactants + products:
                if cpd_id not in self._metabolites:
                    self._metabolites[cpd_id] = Metabolite(id=cpd_id)

            # Create reaction
            reaction = Reaction(
                id=rxn_id,
                reactants=reactants,
                products=products,
                reversibility=reversible,
            )
            self._reactions[rxn_id] = reaction

            # Create atom transition if atoms are specified (skip if
            # there's an error)
            if reactant_atoms and product_atoms:
                try:
                    atom_mapping = self._create_atom_mapping(
                        rxn_id,
                        reactants,
                        products,
                        reactant_atoms,
                        product_atoms,
                    )
                    if atom_mapping:
                        self._atom_mappings[rxn_id] = atom_mapping
                except (ValueError, KeyError):
                    # Skip reactions with invalid atom transitions
                    pass

    def _parse_compounds(
        self, compounds_str: str
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Parse compounds string with optional atom transitions.

        Args:
            compounds_str: String like "G6P(abcdef)+AcCoA(gh)" or
                "G6P(abcdef) + AcCoA(gh)"

        Returns:
            Tuple of (compound_ids, [(compound_id, atoms), ...])
        """
        compound_ids = []
        compound_atoms = []

        # Split by + and parse each compound
        parts = compounds_str.split("+")

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Match compound with optional atom transition:
            # "Compound(atoms)", "Compound (atoms)", or "123Compound(atoms)"
            # \s* allows optional space before parentheses (Freeflux format)
            match = re.match(r"^([\d.]*)?(\S+?)\s*(?:\(([^)]+)\))?$", part)
            if match:
                match.group(1) or ""
                cpd_id = match.group(2)
                atoms = match.group(3) or ""

                # Handle symmetric compounds with comma-separated
                # atoms like "abcd,dcba"
                # Just use the first variant for the compound id
                atoms_clean = atoms.split(",")[0] if atoms else ""

                compound_ids.append(cpd_id)
                if atoms_clean:
                    compound_atoms.append((cpd_id, atoms))

        return compound_ids, compound_atoms

    def _create_atom_mapping(
        self,
        rxn_id: str,
        reactants: List[str],
        products: List[str],
        reactant_atoms: List[Tuple[str, str]],
        product_atoms: List[Tuple[str, str]],
    ) -> Optional[AtomTransition]:
        """Create AtomTransition from letter notation.

        Handles symmetric compounds with comma-separated atom variants.

        Returns:
            AtomTransition object, or None if no valid mappings could be
            created.
        """
        # Check for symmetric compounds (comma-separated atom strings)
        has_variants = any(
            "," in atoms for _, atoms in reactant_atoms + product_atoms
        )

        if has_variants:
            # Generate all variant combinations
            return self._create_variant_atom_mapping(
                rxn_id, reactants, products, reactant_atoms, product_atoms
            )
        else:
            # Simple case: single atom transition
            atom_map = AtomTransition.parse_letter_notation(
                reactant_atoms, product_atoms
            )
            return AtomTransition(
                reaction_id=rxn_id,
                reactants=reactants,
                products=products,
                maps={rxn_id: atom_map},
                weights={rxn_id: 1.0},
            )

    def _create_variant_atom_mapping(
        self,
        rxn_id: str,
        reactants: List[str],
        products: List[str],
        reactant_atoms: List[Tuple[str, str]],
        product_atoms: List[Tuple[str, str]],
    ) -> AtomTransition:
        """Create atom transition with variants for symmetric compounds.

        Returns:
            AtomTransition with all variant combinations and uniform
            weights, or None if no valid mappings could be created.
        """
        from itertools import product as cartesian_product

        # Extract variants for each compound
        reactant_variants = []
        for cpd, atoms in reactant_atoms:
            variants = atoms.split(",")
            reactant_variants.append([(cpd, v.strip()) for v in variants])

        product_variants = []
        for cpd, atoms in product_atoms:
            variants = atoms.split(",")
            product_variants.append([(cpd, v.strip()) for v in variants])

        # Generate all combinations
        all_reactant_combos = (
            list(cartesian_product(*reactant_variants))
            if reactant_variants
            else [()]
        )
        all_product_combos = (
            list(cartesian_product(*product_variants))
            if product_variants
            else [()]
        )

        maps_dict = {}
        variant_idx = 1

        for r_combo in all_reactant_combos:
            for p_combo in all_product_combos:
                r_items = list(r_combo) if r_combo else reactant_atoms
                p_items = list(p_combo) if p_combo else product_atoms

                try:
                    atom_map = AtomTransition.parse_letter_notation(
                        r_items, p_items
                    )
                    map_id = (
                        f"{rxn_id}___{variant_idx}"
                        if len(all_reactant_combos) * len(all_product_combos)
                        > 1
                        else rxn_id
                    )
                    maps_dict[map_id] = atom_map
                    variant_idx += 1
                except Exception:
                    # Skip invalid combinations
                    continue

        if not maps_dict:
            return None

        # Calculate uniform weights
        uniform_weight = 1.0 / len(maps_dict)
        weights_dict = {k: uniform_weight for k in maps_dict.keys()}

        # Store variant IDs in reaction if multiple variants
        if len(maps_dict) > 1:
            if rxn_id in self._reactions:
                # Update reaction with variant IDs
                old_rxn = self._reactions[rxn_id]
                self._reactions[rxn_id] = Reaction(
                    id=rxn_id,
                    reactants=old_rxn.reactants,
                    products=old_rxn.products,
                    reversibility=old_rxn.reversibility,
                    atom_transition_ids=list(maps_dict.keys()),
                )

        return AtomTransition(
            reaction_id=rxn_id,
            reactants=reactants,
            products=products,
            maps=maps_dict,
            weights=weights_dict,
        )

    def _resolve_flux_id(self, flux_id: str) -> str:
        """Map a flux ID to the base reaction ID.

        For variant reactions, the base ID (e.g., 'R24') is kept as-is
        since the data model stores flux values against base reaction IDs.
        Strips any computational suffixes (e.g., '___1') if present.

        Returns:
            The base reaction ID with any computational suffix removed.
        """
        if "___" in flux_id:
            return flux_id.split("___")[0]
        return flux_id

    def _parse_flux_bounds(self, base_path: Path) -> Optional[Constraints]:
        """Parse flux_bounds / constraints file.

        Mirrors the Freeflux ``set_flux_bounds(fluxid, bounds=[lo, hi])`` API.

        Format:
        - #reaction_id: reaction ID, or the special value ``all`` (applies to
          every reaction defined in the reactions file)
        - lo: lower bound (leave blank for no lower bound)
        - hi: upper bound (leave blank for no upper bound)

        Each row produces up to two ``ConstraintFormula`` entries in
        ``NetConstraints``:  ``reaction_id >= lo`` and ``reaction_id <= hi``.

        Returns:
            Constraints object with flux bound formulas, or None if no
            flux_bounds file found or no formulas parsed.
        """
        filepath = self._find_file(base_path, "flux_bounds")
        if not filepath:
            return None

        df = self._read_tabular(filepath)

        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if "reaction" in col_lower and "id" in col_lower:
                col_map["reaction_id"] = col
            elif col_lower in ("lo", "lower", "min", "lb"):
                col_map["lo"] = col
            elif col_lower in ("hi", "upper", "max", "ub"):
                col_map["hi"] = col

        formulas: List[ConstraintFormula] = []

        for _, row in df.iterrows():
            rxn_id = str(row.get(col_map.get("reaction_id", ""), "")).strip()
            if not rxn_id or rxn_id == "nan":
                continue

            lo_raw = row.get(col_map.get("lo", ""), None)
            hi_raw = row.get(col_map.get("hi", ""), None)

            lo = None if pd.isna(lo_raw) else float(lo_raw)
            hi = None if pd.isna(hi_raw) else float(hi_raw)

            # Expand 'all' to every reaction defined in the network. Only the
            # reactions the user actually declared receive these bounds;
            # auto-generated drain reactions for sink metabolites are synthetic
            # and are left unbounded.
            if rxn_id.lower() == "all":
                targets = list(self._reactions.keys())
            else:
                targets = [rxn_id]

            for rid in targets:
                if lo is not None:
                    formulas.append(
                        ConstraintFormula(expression=f"{rid} >= {lo}")
                    )
                if hi is not None:
                    formulas.append(
                        ConstraintFormula(expression=f"{rid} <= {hi}")
                    )

        if not formulas:
            return None

        return Constraints(net=NetConstraints(formulas=formulas))

    def _parse_label_input(self, base_path: Path) -> List[Tracers]:
        """Parse label_input file containing tracer labeling strategies.

        Format (Freeflux set_labeling_strategy convention):
        - #metabolite_ID: Substrate metabolite being labeled
        - labeling_pattern: '0'=unlabeled, '1'=labeled (e.g. '010' for
          2nd-carbon label). Comma-separated list for mixtures.
        - percentage: Molar fraction [0, 1]. Comma-separated list.
        - purity (optional): Labeled atom purity [0, 1]. Comma-separated
          list. Left unset (None) when the column or cell is empty; the
          FluxML writer treats purity and natural abundance together (see
          :meth:`FluxMLWriter.write`).
        - label_atom (optional): Element type, default 'C'.

        Multiple rows for the same metabolite are merged into one Tracers
        entry (each row becomes one LabelComposition).

        Returns:
            List of Tracers objects, one per labeled metabolite. Empty
            list if no label_input file found.
        """
        filepath = self._find_file(base_path, "label_input")
        if not filepath:
            return []

        df = self._read_tabular(filepath)

        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if "metabolite" in col_lower and "id" in col_lower:
                col_map["metabolite_id"] = col
            elif "pattern" in col_lower:
                col_map["labeling_pattern"] = col
            elif "percent" in col_lower:
                col_map["percentage"] = col
            elif "purity" in col_lower:
                col_map["purity"] = col
            elif "atom" in col_lower:
                col_map["label_atom"] = col

        labels_by_metabolite: Dict[str, List[LabelComposition]] = defaultdict(
            list
        )

        for _, row in df.iterrows():
            metab_id = str(
                row.get(col_map.get("metabolite_id", ""), "")
            ).strip()
            if not metab_id or pd.isna(metab_id) or metab_id == "nan":
                continue

            pattern_str = (
                str(row.get(col_map.get("labeling_pattern", ""), ""))
                .strip()
                .strip("'\"")
            )
            pct_str = str(row.get(col_map.get("percentage", ""), "1.0")).strip()
            purity_str = str(row.get(col_map.get("purity", ""), "")).strip()

            if not pattern_str or pattern_str == "nan":
                continue

            # Support comma-separated lists of patterns in a single cell
            patterns = [p.strip().strip("'\"") for p in pattern_str.split(",")]
            percentages = self._parse_comma_values(pct_str) or [1.0] * len(
                patterns
            )
            purities: List[Optional[float]] = list(
                self._parse_comma_values(purity_str)
            )

            # Pad if lengths don't match
            while len(percentages) < len(patterns):
                percentages.append(1.0)
            while len(purities) < len(patterns):
                purities.append(None)

            for pattern, pct, purity in zip(patterns, percentages, purities):
                labels_by_metabolite[metab_id].append(
                    LabelComposition(
                        labeled_pattern=pattern,
                        fraction=float(pct),
                        purity=purity,
                    )
                )

        return [
            Tracers(metabolite=metab_id, type="isotopomer", labels=labels)
            for metab_id, labels in labels_by_metabolite.items()
        ]

    def _parse_fluxes(self, base_path: Path) -> Optional[List[FluxValue]]:
        """Parse fluxes file containing flux values.

        Format:
        - #flux_ID: Flux identifier (e.g., "v1" or "v1_f", "v1_b" for
          reversible)
        - value: Flux value

        Converts FreeFlux forward/backward (_f/_b) convention to
        13CFlux2 net/xch convention:
            net = forward - backward
            xch = min(forward, backward)

        Returns:
            List of FluxValue objects, or None if no fluxes file found
            or no flux values parsed.
        """
        filepath = self._find_file(base_path, "fluxes")
        if not filepath:
            return None

        df = self._read_tabular(filepath)

        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if "flux" in col_lower and "id" in col_lower:
                col_map["flux_id"] = col
            elif "value" in col_lower:
                col_map["value"] = col

        raw_pairs: Dict[str, Dict[str, float]] = defaultdict(dict)
        irreversible: Dict[str, float] = {}

        for _, row in df.iterrows():
            flux_id = str(row.get(col_map.get("flux_id", ""), "")).strip()
            value = row.get(col_map.get("value", ""), None)

            if not flux_id or pd.isna(value):
                continue

            if flux_id.endswith("_f"):
                base_id = flux_id[:-2]
                resolved = self._resolve_flux_id(base_id)
                raw_pairs[resolved]["forward"] = float(value)
            elif flux_id.endswith("_b"):
                base_id = flux_id[:-2]
                resolved = self._resolve_flux_id(base_id)
                raw_pairs[resolved]["backward"] = float(value)
            else:
                resolved = self._resolve_flux_id(flux_id)
                irreversible[resolved] = float(value)

        flux_values = []

        for rxn_id, vals in raw_pairs.items():
            fwd = vals.get("forward", 0.0)
            bwd = vals.get("backward", 0.0)
            net = fwd - bwd
            xch = min(fwd, bwd)

            flux_values.append(
                FluxValue(flux=rxn_id, type="net", value=net, lo=net, hi=net)
            )
            flux_values.append(
                FluxValue(flux=rxn_id, type="xch", value=xch, lo=xch, hi=xch)
            )

        for rxn_id, value in irreversible.items():
            flux_values.append(
                FluxValue(
                    flux=rxn_id, type="net", value=value, lo=value, hi=value
                )
            )

        return flux_values if flux_values else None

    def _parse_concentrations(
        self, base_path: Path
    ) -> Optional[List[MetaboliteSizeValue]]:
        """Parse concentrations file containing metabolite pool sizes.

        Format:
        - #metabolite_ID or #metab_ID: Metabolite identifier
        - value or value (umol/gCDW): Concentration value

        Returns:
            List of MetaboliteSizeValue objects, or None if no
            concentrations file found or no values parsed.
        """
        filepath = self._find_file(base_path, "concentrations")
        if not filepath:
            return None

        df = self._read_tabular(filepath)

        # Standardize column names
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if "metabolite" in col_lower or "metab" in col_lower:
                col_map["metab_id"] = col
            elif "value" in col_lower:
                col_map["value"] = col

        metabolitesize_values = []
        for _, row in df.iterrows():
            metab_id = str(row.get(col_map.get("metab_id", ""), "")).strip()
            value = row.get(col_map.get("value", ""), None)

            if not metab_id or pd.isna(value):
                continue

            metabolitesize_value = MetaboliteSizeValue(
                metabolite=metab_id,
                lo=float(value),
                hi=float(value),
                value=float(value),
            )
            metabolitesize_values.append(metabolitesize_value)

        return metabolitesize_values if metabolitesize_values else None

    def _parse_measurements(self, base_path: Path) -> Optional[Measurement]:
        """Parse measurement files.

        Parses measured_MDVs, measured_fluxes, and measured_inst_MDVs.

        Returns:
            Measurement object containing labeling and flux measurements,
            or None if no measurement files found.
        """
        # Parse steady-state MDV measurements
        labeling_measurement, labeling_data = self._parse_measured_mdvs(
            base_path
        )

        # Parse time-course MDV measurements
        inst_labeling, inst_data = self._parse_measured_inst_mdvs(base_path)

        # Parse flux measurements
        flux_measurement, flux_data = self._parse_measured_fluxes(base_path)

        # Merge labeling measurements
        if inst_labeling and labeling_measurement:
            existing_ids = {g.id for g in inst_labeling.groups}
            unique_static = [
                g
                for g in labeling_measurement.groups
                if g.id not in existing_ids
            ]
            all_groups = list(inst_labeling.groups) + unique_static
            labeling_measurement = LabelingMeasurement(groups=all_groups)
        elif inst_labeling:
            labeling_measurement = inst_labeling

        # Combine data
        all_data = []
        if labeling_data:
            all_data.extend(labeling_data)
        if inst_data:
            all_data.extend(inst_data)
        if flux_data:
            all_data.extend(flux_data)

        if not labeling_measurement and not flux_measurement:
            return None

        return Measurement(
            model=MeasurementModel(
                labeling_measurement=labeling_measurement,
                flux_measurement=flux_measurement,
            ),
            data=MeasurementData(data=all_data),
        )

    def _parse_measured_mdvs(
        self, base_path: Path
    ) -> Tuple[Optional[LabelingMeasurement], Optional[List[Datum]]]:
        """Parse measured_MDVs file containing steady-state MDV measurements.

        Format:
        - #fragment_ID: Fragment identifier like "Glu_12345"
          (metabolite_positions)
        - mean: Comma-separated MDV values
        - sd: Comma-separated standard deviations

        Returns:
            Tuple of (LabelingMeasurement, list of Datum) or (None, None)
            if file not found or no groups parsed.
        """
        filepath = self._find_file(base_path, "measured_MDVs")
        if not filepath:
            return None, None

        df = self._read_tabular(filepath)

        # Standardize column names
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if "fragment" in col_lower and "id" in col_lower:
                col_map["fragment_id"] = col
            elif col_lower == "mean":
                col_map["mean"] = col
            elif col_lower == "sd":
                col_map["sd"] = col

        groups: Dict[str, Group] = {}
        data: List[Datum] = []

        for _, row in df.iterrows():
            fragment_id = str(
                row.get(col_map.get("fragment_id", ""), "")
            ).strip()
            mean_str = str(row.get(col_map.get("mean", ""), ""))
            sd_str = str(row.get(col_map.get("sd", ""), ""))

            if not fragment_id or pd.isna(mean_str) or mean_str == "nan":
                continue

            # Parse fragment_id to extract metabolite and positions
            # Format: "Metabolite_positions" e.g., "Glu_12345"
            parts = fragment_id.rsplit("_", 1)
            if len(parts) == 2:
                metabolite = parts[0]
                positions = parts[1]
            else:
                metabolite = fragment_id
                positions = ""

            # Parse mean and sd values first so we know n_mdv for the expression
            mean_values = self._parse_comma_values(mean_str)
            sd_values = self._parse_comma_values(sd_str)

            # Create group if not exists.
            # Expression format: Metabolite[p1,p2,p3]#M0,1,...,N
            # This is the x3cflux / 13CFlux2 convention for MS fragments.
            if fragment_id not in groups:
                n_mdv = len(mean_values)
                if positions:
                    pos_list = ",".join(positions)
                    mdv_weights = ",".join(str(i) for i in range(n_mdv))
                    expression = f"{metabolite}[{pos_list}]#M{mdv_weights}"
                else:
                    expression = metabolite

                groups[fragment_id] = Group(
                    id=fragment_id,
                    scale="auto",
                    expression=TextualOrMath(textual=expression),
                )

            # Create datum for each MDV component (M0, M1, M2, ...)
            for i, (mean, sd) in enumerate(zip(mean_values, sd_values)):
                datum_id = f"{fragment_id}:M{i}"
                datum = Datum(
                    id=datum_id,
                    value=mean,
                    stddev=sd if sd else 0.01,
                    pos=i,
                )
                data.append(datum)

        if not groups:
            return None, None

        return (LabelingMeasurement(groups=list(groups.values())), data)

    def _parse_measured_inst_mdvs(
        self, base_path: Path
    ) -> Tuple[Optional[LabelingMeasurement], Optional[List[Datum]]]:
        """Parse measured_inst_MDVs file with time-course MDV measurements.

        Format:
        - #fragment_ID: Fragment identifier
        - time or time (s): Time point
        - mean: Comma-separated MDV values
        - sd: Comma-separated standard deviations

        Returns:
            Tuple of (LabelingMeasurement, list of Datum) or (None, None)
            if file not found or no groups parsed.
        """
        filepath = self._find_file(base_path, "measured_inst_MDVs")
        if not filepath:
            return None, None

        df = self._read_tabular(filepath)

        # Standardize column names
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if "fragment" in col_lower and "id" in col_lower:
                col_map["fragment_id"] = col
            elif "time" in col_lower:
                col_map["time"] = col
            elif col_lower == "mean":
                col_map["mean"] = col
            elif col_lower == "sd":
                col_map["sd"] = col

        groups: Dict[str, Group] = {}
        data: List[Datum] = []
        time_points: Dict[str, List[float]] = defaultdict(list)

        for _, row in df.iterrows():
            fragment_id = str(
                row.get(col_map.get("fragment_id", ""), "")
            ).strip()
            time_val = row.get(col_map.get("time", ""), None)
            mean_str = str(row.get(col_map.get("mean", ""), ""))
            sd_str = str(row.get(col_map.get("sd", ""), ""))

            if not fragment_id or pd.isna(mean_str) or mean_str == "nan":
                continue

            # Parse time
            time_float = float(time_val) if not pd.isna(time_val) else None

            # Track time points for this fragment
            if time_float is not None:
                time_points[fragment_id].append(time_float)

            # Parse fragment_id
            parts = fragment_id.rsplit("_", 1)
            if len(parts) == 2:
                metabolite = parts[0]
                positions = parts[1]
            else:
                metabolite = fragment_id
                positions = ""

            # Create group if not exists (will update times later)
            if fragment_id not in groups:
                if positions:
                    pos_list = "+".join(positions)
                    expression = f"{metabolite}[{pos_list}]"
                else:
                    expression = metabolite

                groups[fragment_id] = Group(
                    id=fragment_id,
                    scale="auto",
                    expression=TextualOrMath(textual=expression),
                )

            # Parse mean and sd values
            mean_values = self._parse_comma_values(mean_str)
            sd_values = self._parse_comma_values(sd_str)

            # Create datum for each MDV component
            for i, (mean, sd) in enumerate(zip(mean_values, sd_values)):
                datum_id = f"{fragment_id}:M{i}"
                datum = Datum(
                    id=datum_id,
                    value=mean,
                    stddev=sd if sd else 0.01,
                    time=time_float,
                    pos=i,
                )
                data.append(datum)

        if not groups:
            return None, None

        # Update groups with time information
        updated_groups = []
        for group_id, group in groups.items():
            times = sorted(set(time_points.get(group_id, [])))
            times_str = ",".join(str(t) for t in times) if times else None
            updated_groups.append(
                Group(
                    id=group.id,
                    times=times_str,
                    scale=group.scale,
                    expression=group.expression,
                )
            )

        return (LabelingMeasurement(groups=updated_groups), data)

    def _parse_measured_fluxes(
        self, base_path: Path
    ) -> Tuple[Optional[FluxMeasurement], Optional[List[Datum]]]:
        """Parse measured_fluxes file containing flux measurements.

        Format:
        - #reaction_ID: Reaction identifier
        - mean: Mean flux value
        - sd: Standard deviation

        Returns:
            Tuple of (FluxMeasurement, list of Datum) or (None, None) if
            file not found or no flux measurements parsed.
        """
        filepath = self._find_file(base_path, "measured_fluxes")
        if not filepath:
            return None, None

        df = self._read_tabular(filepath)

        # Standardize column names
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if "reaction" in col_lower and "id" in col_lower:
                col_map["reaction_id"] = col
            elif col_lower == "mean":
                col_map["mean"] = col
            elif col_lower == "sd":
                col_map["sd"] = col

        net_fluxes: List[NetFlux] = []
        data: List[Datum] = []

        for _, row in df.iterrows():
            rxn_id = str(row.get(col_map.get("reaction_id", ""), "")).strip()
            mean = row.get(col_map.get("mean", ""), None)
            sd = row.get(col_map.get("sd", ""), 0.01)

            if not rxn_id or pd.isna(mean):
                continue

            # Create net flux measurement
            sd_value = float(sd) if not pd.isna(sd) else 0.01
            net_flux = NetFlux(
                id=rxn_id,
                expression=TextualOrMath(textual=rxn_id),
                errormodel=ErrorModel(
                    expression=TextualOrMath(textual=str(sd_value))
                ),
            )
            net_fluxes.append(net_flux)

            # Create datum
            datum = Datum(
                id=rxn_id,
                value=float(mean),
                stddev=sd_value,
            )
            data.append(datum)

        if not net_fluxes:
            return None, None

        return (FluxMeasurement(net_fluxes=net_fluxes), data)

    def _parse_comma_values(self, value_str: str) -> List[float]:
        """Parse comma-separated values into list of floats.

        Returns:
            List of float values parsed from the comma-separated string.
            Invalid values are replaced with 0.0.
        """
        if not value_str or value_str == "nan":
            return []

        # Remove quotes if present
        value_str = value_str.strip("\"'")

        values = []
        for v in value_str.split(","):
            v = v.strip()
            try:
                values.append(float(v))
            except ValueError:
                values.append(0.0)
        return values


def parse_freeflux(base_path: str) -> FluxomicsData:
    """Convenience function to parse Freeflux files.

    Args:
        base_path: Path to directory containing Freeflux files.

    Returns:
        FluxomicsData containing the parsed data
    """
    parser = FreefluxParser()
    return parser.parse(base_path)
