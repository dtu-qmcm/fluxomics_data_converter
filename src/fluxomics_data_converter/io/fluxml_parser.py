"""Parser for FluxML XML files (13CFlux2 format).

FluxML is a hierarchical XML format for ¹³C metabolic flux analysis.
This module converts a ``.fml`` file into a
:class:`~fluxomics_data_converter.FluxomicsData`.

Entry point::

    from fluxomics_data_converter.io import parse_fluxml_file, parse
    model = parse_fluxml_file("ecoli.fml")
    model = parse("ecoli.fml")              # auto-detect via unified API

Encoding handling
-----------------
The parser attempts UTF-8, latin-1, iso-8859-1, and cp1252 in order before
falling back to binary mode, because many real-world FluxML files are not
strict UTF-8.

Limitations
-----------
- MathML constraints are parsed structurally but raise
  ``NotImplementedError`` when evaluated.
- NMR (1H-NMR, 13C-NMR), MIMS, flux-ratio, and poolsize-ratio measurement
  types are not parsed.
- FluxML Level 3 multi-tracer labelling profiles are partially supported
  (constant profiles only).
"""

import logging
import xml.etree.ElementTree as ET
from typing import Optional, List, Tuple
from datetime import datetime
import re
from itertools import product

from ..model import (
    FluxomicsData,
    Metadata,
    MetabolicNetworkModel,
    LabelingExperiments,
    Metabolite,
    Reaction,
    AtomTransition,
    AtomMap,
    Variables,
    FluxValue,
    MetaboliteSizeValue,
    Tracers,
    LabelComposition,
    Measurement,
    MeasurementData,
    MeasurementModel,
    Datum,
    LabelingMeasurement,
    Group,
    FluxMeasurement,
    NetFlux,
    ExchangeFlux,
    MetaboliteSizeMeasurement,
    MetaboliteSize,
    Simulation,
    Constraints,
    NetConstraints,
    ExchangeConstraints,
    MetaboliteSizeConstraints,
    Annotation,
    TextualOrMath,
    ErrorModel,
    DictList,
)

logger = logging.getLogger(__name__)


def variant_combinations(lists: List[list]) -> List[tuple]:
    """Cross product of variant lists in the order 13CFLUX enumerates it.

    13CFLUX counts a reaction's variant combinations like an odometer whose
    *first* participant (reducts in document order, then rproducts) turns
    fastest; the i-th id of the reaction's id list names the i-th
    combination.  ``itertools.product`` turns the last one fastest.

    Returns:
        One tuple per combination, with one element per list.
    """
    return [tuple(reversed(c)) for c in product(*reversed(lists))]


class FluxMLParser:
    """Parser for FluxML XML files (13CFlux2 format).

    Reads a ``.fml`` file and converts it into a
    :class:`~fluxomics_data_converter.FluxomicsData`.

    Encoding handling
    -----------------
    Many real-world FluxML files are not strict UTF-8.  The parser tries
    UTF-8, latin-1, iso-8859-1, and cp1252 in order before falling back to
    binary mode, which lets the standard library infer the encoding.

    Limitations
    -----------
    - MathML constraints are parsed structurally but raise
      ``NotImplementedError`` when evaluated.
    - NMR, MIMS, flux-ratio, and poolsize-ratio measurement types are not
      parsed.
    - FluxML Level 3 multi-tracer labelling profiles are partially
      supported (constant profiles only).

    Example::

        parser = FluxMLParser()
        model = parser.parse("models/ecoli.fml")
        # or use the convenience function:
        from fluxomics_data_converter.io import parse_fluxml_file
        model = parse_fluxml_file("models/ecoli.fml")
    """

    def __init__(self):
        """Initialise the parser with FluxML and MathML namespace mappings."""
        self.namespaces = {
            "fluxml": "http://www.13cflux.net/fluxml",
            "mml": "http://www.w3.org/1998/Math/MathML",
        }

    def parse(self, file_path) -> FluxomicsData:
        """Parse a FluxML file and return a :class:`FluxomicsData`.

        This is the canonical method name required by the
        :class:`~fluxomics_data_converter.io.base.FluxomicsParser` protocol.
        ``parse_file`` is kept as a backward-compatible alias.

        Args:
            file_path: Path to the ``.fml`` (or ``.xml``) FluxML file.

        Returns:
            A fully validated :class:`~fluxomics_data_converter.FluxomicsData`.

        Raises:
            ValueError: If the file cannot be parsed (XML error, encoding
                failure, or internal validation failure).
        """
        try:
            # Try to parse with different encodings
            encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]

            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read()

                    # Parse the XML content
                    root = ET.fromstring(content)
                    return self._parse_fluxml(root)

                except UnicodeDecodeError:
                    continue
                except ET.ParseError:
                    # If XML parsing fails, try next encoding
                    continue

            # If all encodings failed, try binary mode with ET.parse
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                return self._parse_fluxml(root)
            except ET.ParseError as e:
                raise ValueError(f"XML parsing error in {file_path}: {e}")

        except Exception as e:
            raise ValueError(f"Error parsing {file_path}: {e}")

    def parse_file(self, file_path) -> FluxomicsData:
        """Backward-compatible alias for :meth:`parse`.

        .. deprecated::
            Use :meth:`parse` instead, which satisfies the
            :class:`~fluxomics_data_converter.io.base.FluxomicsParser` protocol.

        Returns:
            The parsed :class:`~fluxomics_data_converter.FluxomicsData`.
        """
        return self.parse(file_path)

    def _parse_fluxml(self, root: ET.Element) -> FluxomicsData:
        """Parse the root fluxml element.

        Returns:
            The parsed :class:`FluxomicsData`.
        """
        # Handle namespace prefix
        if root.tag.startswith("{"):
            # Namespaced element
            ns_prefix = root.tag.split("}")[0] + "}"
        else:
            # No namespace
            ns_prefix = ""

        # Parse components
        metadata = self._parse_metadata(root.find(f"{ns_prefix}info"))
        model = self._parse_model(root.find(f"{ns_prefix}reactionnetwork"))
        constraints = self._parse_constraints(
            root.find(f"{ns_prefix}constraints")
        )

        # Parse experiments
        experiments = []
        for exp_elem in root.findall(f"{ns_prefix}configuration"):
            experiments.append(self._parse_experiments(exp_elem))

        return FluxomicsData(
            metadata=metadata,
            model=model,
            constraints=constraints,
            experiments=experiments,
        )

    def _parse_metadata(
        self, metadata_elem: Optional[ET.Element]
    ) -> Optional[Metadata]:
        """Parse metadata element.

        Returns:
            The parsed :class:`Metadata`, or None if *metadata_elem*
            is None.
        """
        if metadata_elem is None:
            return None

        # Get namespace prefix
        ns_prefix = self._get_namespace_prefix(metadata_elem)

        name = self._get_text(metadata_elem.find(f"{ns_prefix}name"))
        version = self._get_text(metadata_elem.find(f"{ns_prefix}version"))
        date_str = self._get_text(metadata_elem.find(f"{ns_prefix}date"))
        comment = self._get_text(metadata_elem.find(f"{ns_prefix}comment"))
        modeler = self._get_text(metadata_elem.find(f"{ns_prefix}modeler"))
        strain = self._get_text(metadata_elem.find(f"{ns_prefix}strain"))

        # Parse date
        date = None
        if date_str:
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass  # Invalid date format

        return Metadata(
            name=name,
            version=version,
            date=date,
            comment=comment,
            modeler=modeler,
            strain=strain,
        )

    def _parse_model(self, rn_elem: ET.Element) -> MetabolicNetworkModel:
        """Parse reactionnetwork element into Model object.

        Returns:
            The parsed :class:`MetabolicNetworkModel`.

        Raises:
            ValueError: If *rn_elem* is None or missing required elements.
        """
        if rn_elem is None:
            raise ValueError("reactionnetwork element is required")

        ns_prefix = self._get_namespace_prefix(rn_elem)

        # Parse metabolites
        pools_elem = rn_elem.find(f"{ns_prefix}metabolitepools")
        if pools_elem is None:
            raise ValueError("metabolitepools element is required")

        metabolites = DictList[Metabolite]()
        compartments = set()
        for pool_elem in pools_elem.findall(f"{ns_prefix}pool"):
            metabolite = self._parse_metabolite(pool_elem)
            metabolites.append(metabolite)
            if metabolite.compartment:
                compartments.add(metabolite.compartment)

        # Parse reactions and collect atom transitions
        reactions = DictList[Reaction]()
        atom_mappings = {}
        for reaction_elem in rn_elem.findall(f"{ns_prefix}reaction"):
            reaction, atom_mapping = self._parse_reaction(reaction_elem)
            reactions.append(reaction)
            if atom_mapping:
                atom_mappings[reaction.id] = atom_mapping

        return MetabolicNetworkModel(
            metabolites=metabolites,
            reactions=reactions,
            atom_mappings=atom_mappings,
            compartments=list(compartments),
        )

    def _parse_metabolite(self, pool_elem: ET.Element) -> Metabolite:
        """Parse metabolite (pool) element.

        Returns:
            The parsed :class:`Metabolite`.

        Raises:
            ValueError: If the pool element has no ``id`` attribute.
        """
        metabolite_id = pool_elem.get("id")
        if not metabolite_id:
            raise ValueError("Metabolite (pool) must have an id attribute")

        # Parse optional attributes - only set if present in XML
        atoms_str = pool_elem.get("atoms")
        atoms = int(atoms_str) if atoms_str is not None else None

        weight_str = pool_elem.get("size")
        weight = float(weight_str) if weight_str is not None else None

        formula = pool_elem.get("cfg")
        compartment = pool_elem.get("compartment")

        # Parse annotations
        annotations = []
        ns_prefix = self._get_namespace_prefix(pool_elem)
        for ann_elem in pool_elem.findall(f"{ns_prefix}annotation"):
            annotations.append(self._parse_annotation(ann_elem))

        return Metabolite(
            id=metabolite_id,
            name=None,  # FluxML doesn't typically have separate name field
            atoms=atoms,
            weight=weight,
            formula=formula,
            compartment=compartment,
            annotations=annotations,
        )

    def _parse_reaction(self, reaction_elem: ET.Element) -> Reaction:
        """Parse reaction element.

        Returns:
            A tuple of (Reaction, atom_mapping or None).

        Raises:
            ValueError: If the reaction, reactant, or product lacks an id.
        """
        reaction_id = reaction_elem.get("id")
        if not reaction_id:
            raise ValueError("Reaction must have an id attribute")

        # Detect variant reactions: space-separated IDs like "SCS___1 SCS___2"
        atom_transition_ids = None
        reaction_id = reaction_id
        if " " in reaction_id:
            # This is a variant reaction
            atom_transition_ids = reaction_id.split()
            # Extract base name from first variant (e.g., "SCS___1" -> "SCS")
            reaction_id = atom_transition_ids[0].split("___")[0]

        reversibility = (
            reaction_elem.get("bidirectional", "true").lower() == "true"
        )

        ns_prefix = self._get_namespace_prefix(reaction_elem)

        # Parse annotations
        annotations = []
        for ann_elem in reaction_elem.findall(f"{ns_prefix}annotation"):
            annotations.append(self._parse_annotation(ann_elem))

        # Parse reactants and products, keeping each participant's cfg
        # variants by position: [(cfg, ratio), ...] - one entry for a plain
        # cfg attribute, none for a participant without atoms.
        def participant_variants(
            elem: ET.Element,
        ) -> List[Tuple[str, Optional[float]]]:
            variant_elems = elem.findall(f"{ns_prefix}variant")
            if not variant_elems:
                cfg = elem.get("cfg")
                return [(cfg, None)] if cfg else []
            variants = []
            for variant_elem in variant_elems:
                cfg = variant_elem.get("cfg")
                ratio_str = variant_elem.get("ratio")
                if cfg:
                    variants.append(
                        (cfg, float(ratio_str) if ratio_str else None)
                    )
            return variants

        reactants = []
        reactant_variants = []
        for reduct_elem in reaction_elem.findall(f"{ns_prefix}reduct"):
            reduct_id = reduct_elem.get("id")
            if not reduct_id:
                raise ValueError("Reactant must have an id attribute")
            reactants.append(reduct_id)
            reactant_variants.append(participant_variants(reduct_elem))

        products = []
        product_variants = []
        for rproduct_elem in reaction_elem.findall(f"{ns_prefix}rproduct"):
            rproduct_id = rproduct_elem.get("id")
            if not rproduct_id:
                raise ValueError("Product must have an id attribute")
            products.append(rproduct_id)
            product_variants.append(participant_variants(rproduct_elem))

        # Build atom transition if configurations exist
        atom_mapping = None
        if any(product_variants):
            options = [
                variants or [(None, None)]
                for variants in reactant_variants + product_variants
            ]
            has_variants = any(len(o) > 1 for o in options)
            if has_variants and atom_transition_ids:
                # One combination per reaction id, in 13CFLUX's order
                combinations = variant_combinations(options)
                if len(atom_transition_ids) != len(combinations):
                    base_name = atom_transition_ids[0].split("___")[0]
                    atom_transition_ids = [
                        f"{base_name}___{i + 1}"
                        for i in range(len(combinations))
                    ]
            else:
                combinations = [tuple(o[0] for o in options)]

            maps_dict = {}
            weights_dict = {}
            n_reactants = len(reactants)
            for variant_idx, combination in enumerate(combinations):
                reactant_items = [
                    (cpd, cfg)
                    for cpd, (cfg, _) in zip(
                        reactants, combination[:n_reactants]
                    )
                ]
                product_items = [
                    (cpd, cfg)
                    for cpd, (cfg, _) in zip(
                        products, combination[n_reactants:]
                    )
                    if cfg
                ]
                atom_map = self._parse_atom_map(
                    reactant_items, product_items, reactants
                )
                if not atom_map:
                    continue
                if not (has_variants and atom_transition_ids):
                    # Single mapping, keyed by the reaction id
                    maps_dict[reaction_id] = atom_map
                    continue
                map_id = atom_transition_ids[variant_idx]
                maps_dict[map_id] = atom_map
                ratios = [r for _, r in combination if r is not None]
                if ratios:
                    weight = 1.0
                    for ratio in ratios:
                        weight *= ratio
                    weights_dict[map_id] = weight

            if maps_dict and has_variants and atom_transition_ids:
                if weights_dict:
                    # Normalize weights
                    total_weight = sum(weights_dict.values())
                    if total_weight > 0:
                        weights_dict = {
                            k: v / total_weight for k, v in weights_dict.items()
                        }
                else:
                    # Use uniform weights
                    uniform_weight = 1.0 / len(maps_dict)
                    weights_dict = {k: uniform_weight for k in maps_dict.keys()}
                atom_mapping = AtomTransition(
                    reaction_id=reaction_id,
                    reactants=reactants,
                    products=products,
                    maps=maps_dict,
                    weights=weights_dict,
                )
            elif maps_dict:
                atom_mapping = AtomTransition(
                    reaction_id=reaction_id,
                    reactants=reactants,
                    products=products,
                    maps=maps_dict,
                )

        return Reaction(
            id=reaction_id,
            name=None,
            reversibility=reversibility,
            annotations=annotations,
            reactants=reactants,
            products=products,
            atom_transition_ids=atom_transition_ids,
        ), atom_mapping

    @staticmethod
    def _parse_atom_map(
        reactant_items: List[Tuple[str, Optional[str]]],
        product_items: List[Tuple[str, str]],
        reactants: List[str],
    ) -> Optional[AtomMap]:
        """Parse one atom map from reactant and product cfg strings.

        Args:
            reactant_items: ``(reactant_id, cfg or None)`` per reactant, in
                reaction order.
            product_items: ``(product_id, cfg)`` for products with a cfg.
            reactants: Reactant ids in reaction order.

        Returns:
            The atom map, or None if the cfgs are in neither the ``C#k@r``
            nor the letter notation.
        """
        sample_cfg = next(
            (cfg for _, cfg in reactant_items + product_items if cfg), ""
        )
        if re.search(r"[A-Z]#\d+@\d+", sample_cfg):
            # C#1@2 format
            return AtomTransition.parse_fluxml_cfg(
                reactant_cfgs=reactant_items,
                product_cfgs=product_items,
                reactant_order=reactants,
            )
        if re.search(r"[a-zA-Z]", sample_cfg):
            # Letter notation (abc format)
            return AtomTransition.parse_letter_notation(
                reactant_items=[(c, cfg) for c, cfg in reactant_items if cfg],
                product_items=product_items,
            )
        return None

    def _parse_annotation(self, ann_elem: ET.Element) -> Annotation:
        """Parse annotation element.

        Returns:
            The parsed :class:`Annotation`.

        Raises:
            ValueError: If the annotation has no ``name`` attribute.
        """
        name = ann_elem.get("name")
        if not name:
            raise ValueError("Annotation must have a name attribute")

        content = ann_elem.text
        return Annotation(name=name, content=content)

    def _parse_constraints(
        self, constraints_elem: Optional[ET.Element]
    ) -> Optional[Constraints]:
        """Parse constraints element.

        Returns:
            The parsed :class:`Constraints`, or None if *constraints_elem*
            is None.
        """
        if constraints_elem is None:
            return None

        ns_prefix = self._get_namespace_prefix(constraints_elem)

        # Parse net constraints
        net_elem = constraints_elem.find(f"{ns_prefix}net")
        net = (
            self._parse_net_constraints(net_elem)
            if net_elem is not None
            else None
        )

        # Parse xch constraints (also check for 'exchange' element)
        xch_elem = constraints_elem.find(f"{ns_prefix}xch")
        if xch_elem is None:
            xch_elem = constraints_elem.find(f"{ns_prefix}exchange")
        xch = (
            self._parse_xch_constraints(xch_elem)
            if xch_elem is not None
            else None
        )

        # Parse psize constraints
        psize_elem = constraints_elem.find(f"{ns_prefix}psize")
        psize = (
            self._parse_metabolitesize_constraints(psize_elem)
            if psize_elem is not None
            else None
        )

        return Constraints(net=net, xch=xch, metabolitesize=psize)

    def _parse_net_constraints(self, net_elem: ET.Element) -> NetConstraints:
        """Parse net constraints element.

        Returns:
            The parsed :class:`NetConstraints`.
        """
        formulas = self._parse_constraint_formulas(net_elem)
        return NetConstraints(formulas=formulas)

    def _parse_xch_constraints(
        self, xch_elem: ET.Element
    ) -> ExchangeConstraints:
        """Parse xch constraints element.

        Returns:
            The parsed :class:`ExchangeConstraints`.
        """
        formulas = self._parse_constraint_formulas(xch_elem)
        return ExchangeConstraints(formulas=formulas)

    def _parse_metabolitesize_constraints(
        self, psize_elem: ET.Element
    ) -> MetaboliteSizeConstraints:
        """Parse psize constraints element.

        Returns:
            The parsed :class:`MetaboliteSizeConstraints`.
        """
        formulas = self._parse_constraint_formulas(psize_elem)
        return MetaboliteSizeConstraints(formulas=formulas)

    def _parse_constraint_formulas(self, elem: ET.Element) -> List:
        """Parse constraint formulas from textual or MathML content.

        Textual constraints are semicolon-separated formulas, one per line.
        Format: [name:] expression

        Examples:
            - bmALA>=0.75*0.22601*mu;
            - uptGLYC=.5154448999999999;
            - ratio: uptUGlyc=0.12*uptGLYC;

        Returns:
            A list of parsed constraint formulas.
        """
        from ..model.constraint import ConstraintFormula

        ns_prefix = self._get_namespace_prefix(elem)
        formulas = []

        # Check for textual content
        textual_elem = elem.find(f"{ns_prefix}textual")
        if textual_elem is not None and textual_elem.text:
            textual = textual_elem.text
            # Split by semicolons and process each formula
            lines = textual.split(";")
            for line in lines:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("<!--") or line.startswith("//"):
                    continue

                # Check for named constraint (format: "name: expression")
                name = None
                if ":" in line:
                    parts = line.split(":", 1)
                    # Check if this is actually a constraint name
                    # (not part of formula)
                    # Named constraints have format "name: formula"
                    # where name doesn't contain operators
                    potential_name = parts[0].strip()
                    if not any(
                        op in potential_name
                        for op in ["<", ">", "=", "+", "-", "*", "/", "(", ")"]
                    ):
                        name = potential_name
                        line = parts[1].strip()

                if line:  # Only add non-empty expressions
                    formulas.append(
                        ConstraintFormula(
                            name=name, expression=line, is_mathml=False
                        )
                    )

        # Check for MathML content
        math_elems = elem.findall(f"{ns_prefix}math")
        for math_elem in math_elems:
            mathml = ET.tostring(math_elem, encoding="unicode")
            # Try to extract constraint name from attributes if present
            name = math_elem.get("name") or math_elem.get("id")
            formulas.append(
                ConstraintFormula(name=name, expression=mathml, is_mathml=True)
            )

        return formulas

    def _parse_textual_or_math(self, elem: ET.Element) -> TextualOrMath:
        """Parse textual or MathML content.

        Returns:
            The parsed :class:`TextualOrMath`.
        """
        ns_prefix = self._get_namespace_prefix(elem)

        textual_elem = elem.find(f"{ns_prefix}textual")
        textual = textual_elem.text if textual_elem is not None else None

        # For MathML, we'd need to serialize the subtree
        math_elem = elem.find(f"{ns_prefix}math")
        mathml = None
        if math_elem is not None:
            mathml = ET.tostring(math_elem, encoding="unicode")

        # If neither textual nor mathml, use element text
        if textual is None and mathml is None:
            textual = elem.text

        return TextualOrMath(textual=textual, mathml=mathml)

    def _parse_experiments(
        self, config_elem: ET.Element
    ) -> LabelingExperiments:
        """Parse configuration element into a LabelingExperiments object.

        Returns:
            The parsed :class:`LabelingExperiments`.

        Raises:
            ValueError: If the configuration has no ``name`` attribute.
        """
        name = config_elem.get("name")
        if not name:
            raise ValueError(
                "Experiment (configuration) must have a name attribute"
            )

        stationary = config_elem.get("stationary", "true").lower() == "true"
        time_str = config_elem.get("time")
        time = float(time_str) if time_str else None

        ns_prefix = self._get_namespace_prefix(config_elem)

        # Parse comment
        comment_elem = config_elem.find(f"{ns_prefix}comment")
        comment = comment_elem.text if comment_elem is not None else None

        # Parse tracers (inputs)
        tracers = []
        for input_elem in config_elem.findall(f"{ns_prefix}input"):
            tracers.append(self._parse_tracer(input_elem))

        # Parse constraints (optional)
        constraints_elem = config_elem.find(f"{ns_prefix}constraints")
        constraints = (
            self._parse_constraints(constraints_elem)
            if constraints_elem is not None
            else None
        )

        # Parse measurement (optional)
        measurement_elem = config_elem.find(f"{ns_prefix}measurement")
        measurement = (
            self._parse_measurement(measurement_elem)
            if measurement_elem is not None
            else None
        )

        # Parse simulation (optional)
        simulation_elem = config_elem.find(f"{ns_prefix}simulation")
        simulation = (
            self._parse_simulation(simulation_elem)
            if simulation_elem is not None
            else None
        )

        return LabelingExperiments(
            name=name,
            stationary=stationary,
            time=time,
            comment=comment,
            tracers=tracers,
            constraints=constraints,
            measurement=measurement,
            simulation=simulation,
        )

    def _parse_tracer(self, input_elem: ET.Element) -> Tracers:
        """Parse input element into a Tracers object.

        Returns:
            The parsed :class:`Tracers`.

        Raises:
            ValueError: If the input has no ``pool`` attribute.
        """
        metabolite = input_elem.get("pool")
        if not metabolite:
            raise ValueError("Tracer (input) must have a pool attribute")

        input_id = input_elem.get("id")
        input_type = input_elem.get("type", "isotopomer")
        profile = input_elem.get("profile")

        # Parse labels
        labels = []
        ns_prefix = self._get_namespace_prefix(input_elem)
        for label_elem in input_elem.findall(f"{ns_prefix}label"):
            labels.append(self._parse_label(label_elem))

        return Tracers(
            id=input_id,
            metabolite=metabolite,
            type=input_type,
            profile=profile,
            labels=labels,
        )

    def _parse_label(self, label_elem: ET.Element) -> LabelComposition:
        """Parse label element.

        Returns:
            The parsed :class:`LabelComposition`.

        Raises:
            ValueError: If the label has no ``cfg`` attribute.
        """
        cfg = label_elem.get("cfg")
        if not cfg:
            raise ValueError("Label must have a cfg attribute")

        # Parse purity as float
        purity_str = label_elem.get("purity")
        purity = float(purity_str) if purity_str else None

        # Parse cost as float
        cost_str = label_elem.get("cost")
        cost = float(cost_str) if cost_str else None

        # Parse fraction from content
        fraction_str = label_elem.text
        fraction = (
            float(fraction_str.strip())
            if fraction_str and fraction_str.strip()
            else None
        )

        return LabelComposition(
            labeled_pattern=cfg, purity=purity, cost=cost, fraction=fraction
        )

    def _parse_measurement(self, measurement_elem: ET.Element) -> Measurement:
        """Parse measurement element according to FluxML schema.

        Returns:
            The parsed :class:`Measurement`.

        Raises:
            ValueError: If the measurement lacks a model or data element.
        """
        ns_prefix = self._get_namespace_prefix(measurement_elem)

        # Parse model section (required)
        model_elem = measurement_elem.find(f"{ns_prefix}model")
        if model_elem is None:
            raise ValueError("measurement must have a model element")
        model = self._parse_measurement_model(model_elem)

        # Parse data section (required)
        data_elem = measurement_elem.find(f"{ns_prefix}data")
        if data_elem is None:
            raise ValueError("measurement must have a data element")
        data = self._parse_measurement_data(data_elem)

        return Measurement(model=model, data=data)

    def _parse_measurement_model(
        self, model_elem: ET.Element
    ) -> MeasurementModel:
        """Parse measurement model element.

        Returns:
            The parsed :class:`MeasurementModel`.
        """
        ns_prefix = self._get_namespace_prefix(model_elem)

        # Parse labelingmeasurement (optional)
        labeling_elem = model_elem.find(f"{ns_prefix}labelingmeasurement")
        labeling_measurement = None
        if labeling_elem is not None:
            labeling_measurement = self._parse_labeling_measurement(
                labeling_elem
            )

        # Parse fluxmeasurement (optional)
        flux_elem = model_elem.find(f"{ns_prefix}fluxmeasurement")
        flux_measurement = None
        if flux_elem is not None:
            flux_measurement = self._parse_flux_measurement(flux_elem)

        # Parse poolsizemeasurement (optional)
        poolsize_elem = model_elem.find(f"{ns_prefix}poolsizemeasurement")
        metabolitesize_measurement = None
        if poolsize_elem is not None:
            metabolitesize_measurement = self._parse_metabolitesize_measurement(
                poolsize_elem
            )

        return MeasurementModel(
            labeling_measurement=labeling_measurement,
            flux_measurement=flux_measurement,
            metabolitesize_measurement=metabolitesize_measurement,
        )

    def _parse_labeling_measurement(
        self, labeling_elem: ET.Element
    ) -> LabelingMeasurement:
        """Parse labelingmeasurement element.

        Returns:
            The parsed :class:`LabelingMeasurement`.
        """
        ns_prefix = self._get_namespace_prefix(labeling_elem)

        groups = []
        # Check for both 'group' and 'MSgroup' elements
        for group_elem in labeling_elem.findall(f"{ns_prefix}group"):
            groups.append(self._parse_group(group_elem))
        for group_elem in labeling_elem.findall(f"{ns_prefix}MSgroup"):
            groups.append(self._parse_group(group_elem))

        return LabelingMeasurement(groups=groups)

    def _parse_group(self, group_elem: ET.Element) -> Group:
        """Parse group element.

        Returns:
            The parsed :class:`Group`.

        Raises:
            ValueError: If the group has no ``id`` attribute.
        """
        group_id = group_elem.get("id")
        if not group_id:
            raise ValueError("Group must have an id attribute")

        times = group_elem.get("times")
        scale = group_elem.get("scale", "auto")
        spec = group_elem.get("spec")

        ns_prefix = self._get_namespace_prefix(group_elem)

        # Parse errormodel (optional)
        errormodel_elem = group_elem.find(f"{ns_prefix}errormodel")
        errormodel = None
        if errormodel_elem is not None:
            errormodel = ErrorModel(
                expression=self._parse_textual_or_math(errormodel_elem)
            )

        # Parse expression (textual or math)
        # If spec attribute exists, use it as textual expression
        if spec:
            expression = TextualOrMath(textual=spec)
        else:
            expression = self._parse_textual_or_math(group_elem)

        return Group(
            id=group_id,
            times=times,
            scale=scale,
            errormodel=errormodel,
            expression=expression,
        )

    def _parse_flux_measurement(self, flux_elem: ET.Element) -> FluxMeasurement:
        """Parse fluxmeasurement element.

        Returns:
            The parsed :class:`FluxMeasurement`.
        """
        ns_prefix = self._get_namespace_prefix(flux_elem)

        net_fluxes = []
        for netflux_elem in flux_elem.findall(f"{ns_prefix}netflux"):
            net_fluxes.append(self._parse_netflux(netflux_elem))

        xch_fluxes = []
        for xchflux_elem in flux_elem.findall(f"{ns_prefix}xchflux"):
            xch_fluxes.append(self._parse_xchflux(xchflux_elem))

        return FluxMeasurement(net_fluxes=net_fluxes, xch_fluxes=xch_fluxes)

    def _parse_netflux(self, netflux_elem: ET.Element) -> NetFlux:
        """Parse netflux element.

        Returns:
            The parsed :class:`NetFlux`.

        Raises:
            ValueError: If the netflux has no ``id`` attribute.
        """
        netflux_id = netflux_elem.get("id")
        if not netflux_id:
            raise ValueError("NetFlux must have an id attribute")

        ns_prefix = self._get_namespace_prefix(netflux_elem)

        # Parse errormodel (optional)
        errormodel_elem = netflux_elem.find(f"{ns_prefix}errormodel")
        errormodel = None
        if errormodel_elem is not None:
            errormodel = ErrorModel(
                expression=self._parse_textual_or_math(errormodel_elem)
            )

        # Parse expression
        expression = self._parse_textual_or_math(netflux_elem)

        return NetFlux(
            id=netflux_id, errormodel=errormodel, expression=expression
        )

    def _parse_xchflux(self, xchflux_elem: ET.Element) -> ExchangeFlux:
        """Parse xchflux element.

        Returns:
            The parsed :class:`ExchangeFlux`.

        Raises:
            ValueError: If the xchflux has no ``id`` attribute.
        """
        xchflux_id = xchflux_elem.get("id")
        if not xchflux_id:
            raise ValueError("ExchangeFlux must have an id attribute")

        ns_prefix = self._get_namespace_prefix(xchflux_elem)

        # Parse errormodel (optional)
        errormodel_elem = xchflux_elem.find(f"{ns_prefix}errormodel")
        errormodel = None
        if errormodel_elem is not None:
            errormodel = ErrorModel(
                expression=self._parse_textual_or_math(errormodel_elem)
            )

        # Parse expression
        expression = self._parse_textual_or_math(xchflux_elem)

        return ExchangeFlux(
            id=xchflux_id, errormodel=errormodel, expression=expression
        )

    def _parse_metabolitesize_measurement(
        self, poolsize_elem: ET.Element
    ) -> MetaboliteSizeMeasurement:
        """Parse poolsizemeasurement element.

        Returns:
            The parsed :class:`MetaboliteSizeMeasurement`.
        """
        ns_prefix = self._get_namespace_prefix(poolsize_elem)

        metabolite_sizes = []
        for poolsize_item_elem in poolsize_elem.findall(f"{ns_prefix}poolsize"):
            metabolite_sizes.append(
                self._parse_metabolitesize(poolsize_item_elem)
            )

        return MetaboliteSizeMeasurement(metabolite_sizes=metabolite_sizes)

    def _parse_metabolitesize(
        self, poolsize_elem: ET.Element
    ) -> MetaboliteSize:
        """Parse poolsize element.

        Returns:
            The parsed :class:`MetaboliteSize`.

        Raises:
            ValueError: If the poolsize has no ``id`` attribute.
        """
        metabolitesize_id = poolsize_elem.get("id")
        if not metabolitesize_id:
            raise ValueError(
                "MetaboliteSize (poolsize) must have an id attribute"
            )

        ns_prefix = self._get_namespace_prefix(poolsize_elem)

        # Parse errormodel (optional)
        errormodel_elem = poolsize_elem.find(f"{ns_prefix}errormodel")
        errormodel = None
        if errormodel_elem is not None:
            errormodel = ErrorModel(
                expression=self._parse_textual_or_math(errormodel_elem)
            )

        # Parse expression
        expression = self._parse_textual_or_math(poolsize_elem)

        return MetaboliteSize(
            id=metabolitesize_id, errormodel=errormodel, expression=expression
        )

    def _parse_measurement_data(self, data_elem: ET.Element) -> MeasurementData:
        """Parse measurement data element.

        Returns:
            The parsed :class:`MeasurementData`.
        """
        ns_prefix = self._get_namespace_prefix(data_elem)

        data = []
        for datum_elem in data_elem.findall(f"{ns_prefix}datum"):
            data.append(self._parse_datum(datum_elem))

        return MeasurementData(data=data)

    def _parse_datum_value(self, value_text: str, datum_id: str) -> float:
        """Parse a datum value string, handling malformed scientific notation.

        Some FluxML files have truncated exponents like 'e-0' where the final
        digit was lost (e.g., '8.769e-07' became '8.769e-0'). Since the lost
        digit cannot be recovered, the value is set to 0.0 with a warning.

        Returns:
            The parsed float value, or 0.0 if the notation is malformed.
        """
        stripped = value_text.strip()
        if re.search(r"e[+-]0$", stripped):
            logger.warning(
                "Datum '%s': malformed scientific notation '%s' "
                "(truncated exponent). Setting value to 0.0.",
                datum_id,
                stripped,
            )
            return 0.0
        return float(stripped)

    def _parse_datum(self, datum_elem: ET.Element) -> Datum:
        """Parse datum element.

        Returns:
            The parsed :class:`Datum`.

        Raises:
            ValueError: If the datum lacks an id, stddev, or value.
        """
        datum_id = datum_elem.get("id")
        if not datum_id:
            raise ValueError("Datum must have an id attribute")

        stddev_str = datum_elem.get("stddev")
        if not stddev_str:
            raise ValueError("Datum must have a stddev attribute")
        stddev = float(stddev_str.strip())

        # Parse optional attributes
        row_str = datum_elem.get("row")
        row = int(row_str) if row_str else None

        time_str = datum_elem.get("time")
        time = float(time_str) if time_str else None

        weight = datum_elem.get("weight")

        pos_str = datum_elem.get("pos")
        pos = int(pos_str) if pos_str else None

        # FluxML uses "weight" to denote the mass isotopomer position
        # (M0, M1, ...)
        # Map it to pos when pos is not explicitly set
        if pos is None and weight is not None:
            try:
                pos = int(weight)
            except ValueError:
                pass

        datum_type = datum_elem.get("type")

        # Parse value from element text
        value_text = datum_elem.text
        if not value_text:
            raise ValueError("Datum must have a value in its text content")
        value = self._parse_datum_value(value_text, datum_id)

        return Datum(
            id=datum_id,
            value=value,
            stddev=stddev,
            row=row,
            time=time,
            weight=weight,
            pos=pos,
            type=datum_type,
        )

    def _parse_simulation(self, simulation_elem: ET.Element) -> Simulation:
        """Parse simulation element.

        Returns:
            The parsed :class:`Simulation`.
        """
        sim_type = simulation_elem.get("type", "auto")
        method = simulation_elem.get("method", "auto")

        ns_prefix = self._get_namespace_prefix(simulation_elem)

        # Parse variables
        variables_elem = simulation_elem.find(f"{ns_prefix}variables")
        variables = None
        if variables_elem is not None:
            flux_values = []
            for flux_elem in variables_elem.findall(f"{ns_prefix}fluxvalue"):
                flux_values.append(self._parse_flux_value(flux_elem))

            metabolite_values = []
            for tag in ("poolsizevalue", "metabolitesizevalue"):
                for met_elem in variables_elem.findall(f"{ns_prefix}{tag}"):
                    metabolite_values.append(
                        self._parse_metabolite_size_value(met_elem)
                    )

            variables = Variables(
                flux_values=flux_values, metabolitesize_values=metabolite_values
            )

        return Simulation(type=sim_type, method=method, variables=variables)

    def _parse_flux_value(self, flux_elem: ET.Element) -> FluxValue:
        """Parse a ``<fluxvalue>`` element.

        The text content is the *initial value* of the free flux parameter.
        The optional ``lo``, ``hi``, and ``inc`` attributes carry the lower
        bound, upper bound, and increment respectively. The ``ed-weight``
        attribute (hyphenated in XML) maps to the ``edweight`` field.

        Per the FluxML spec::

            <fluxvalue flux="Glc_upt" type="net"
                ed-weight="0.8">2.234</fluxvalue>

        Returns:
            The parsed :class:`FluxValue`.

        Raises:
            ValueError: If the element has no ``flux`` attribute.
        """
        flux = flux_elem.get("flux")
        if not flux:
            raise ValueError("FluxValue must have a flux attribute")

        flux_type = flux_elem.get("type", "net")

        def _opt_float(attr: str) -> Optional[float]:
            """Return float attribute value, or None if absent/unparseable."""
            raw = flux_elem.get(attr)
            if raw is None:
                return None
            try:
                return float(raw.strip())
            except ValueError:
                return None

        # Text content → initial value (not a bound).
        value: Optional[float] = None
        value_str = flux_elem.text
        if value_str and value_str.strip():
            try:
                value = float(value_str.strip())
            except ValueError:
                value = None

        lo = _opt_float("lo")
        hi = _opt_float("hi")
        inc = _opt_float("inc")
        # FluxML uses "ed-weight" (hyphenated); also accept "edweight".
        edweight_str = flux_elem.get("ed-weight") or flux_elem.get("edweight")
        edweight = 1.0
        if edweight_str:
            try:
                edweight = float(edweight_str.strip())
            except ValueError:
                pass

        return FluxValue(
            flux=flux,
            type=flux_type,
            value=value,
            lo=lo,
            hi=hi,
            inc=inc,
            edweight=edweight,
        )

    def _parse_metabolite_size_value(
        self, met_elem: ET.Element
    ) -> MetaboliteSizeValue:
        """Parse a ``<poolsizevalue>`` element.

        The text content is the *initial value* of the free pool-size
        parameter.  The optional ``lo``, ``hi``, and ``inc`` attributes carry
        the lower bound, upper bound, and increment respectively.

        Per the FluxML spec::

            <poolsizevalue pool="Ala" edweight="0.1">0.4654</poolsizevalue>

        Returns:
            The parsed :class:`MetaboliteSizeValue`.

        Raises:
            ValueError: If the element has no ``pool`` attribute.
        """
        pool = met_elem.get("pool")
        if not pool:
            raise ValueError("MetaboliteSizeValue must have a pool attribute")

        def _opt_float(attr: str) -> Optional[float]:
            raw = met_elem.get(attr)
            if raw is None:
                return None
            try:
                return float(raw.strip())
            except ValueError:
                return None

        # Text content → initial value (not a bound).
        value: Optional[float] = None
        value_str = met_elem.text
        if value_str and value_str.strip():
            try:
                value = float(value_str.strip())
            except ValueError:
                value = None

        lo = _opt_float("lo")
        hi = _opt_float("hi")
        inc = _opt_float("inc")
        edweight_str = met_elem.get("edweight") or met_elem.get("ed-weight")
        edweight = 1.0
        if edweight_str:
            try:
                edweight = float(edweight_str.strip())
            except ValueError:
                pass

        return MetaboliteSizeValue(
            metabolite=pool,
            value=value,
            lo=lo,
            hi=hi,
            inc=inc,
            edweight=edweight,
        )

    def _get_namespace_prefix(self, elem: ET.Element) -> str:
        """Get namespace prefix for element.

        Returns:
            The namespace prefix string, or empty string if none.
        """
        if elem.tag.startswith("{"):
            return elem.tag.split("}")[0] + "}"
        return ""

    def _get_text(self, elem: Optional[ET.Element]) -> Optional[str]:
        """Get text content from element.

        Returns:
            The stripped text content, or None if *elem* is None or empty.
        """
        if elem is None:
            return None
        return elem.text.strip() if elem.text else None


def parse_fluxml_file(file_path: str) -> FluxomicsData:
    """Convenience wrapper: parse a FluxML file and return a FluxomicsData.

    Args:
        file_path: Path to the ``.fml`` (or ``.xml``) FluxML file.

    Returns:
        A fully validated :class:`~fluxomics_data_converter.FluxomicsData`.
    """
    return FluxMLParser().parse(file_path)
