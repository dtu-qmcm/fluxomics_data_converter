"""Tests for the INCA MATLAB model-script parser."""

import shutil
import tempfile
import warnings
from pathlib import Path

import pytest

from fluxomics_data_converter.core.core import FluxomicsData
from fluxomics_data_converter.io import (
    INCAParser,
    parse,
    parse_fluxml_file,
    parse_inca,
    write_fluxml,
)
from fluxomics_data_converter.io._matlab import (
    UNKNOWN,
    evaluate,
    read_statements,
    split_call,
)

BENCHMARK_DIR = Path(__file__).parent.parent.parent / "data" / "benchmark"
INCA_DIR = BENCHMARK_DIR / "INCA"
EC_SCRIPT = INCA_DIR / "EC.m"
SYN_SCRIPT = INCA_DIR / "Syn.m"
POOLSIZES_SCRIPT = INCA_DIR / "EC_poolsizes.m"


@pytest.fixture(autouse=True)
def _quiet_tracer_warnings():
    """Silence the natural-abundance remainder warning from Tracers."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Label fractions sum to", category=UserWarning
        )
        yield


class TestMatlabLexer:
    """The MATLAB statement layer."""

    def test_joins_continuations(self):
        statements = read_statements("a = [1,...\n     2,...\n     3];")
        assert len(statements) == 1
        assert statements[0].text == "a = [1,      2,      3]"

    def test_strips_comments_but_not_inside_strings(self):
        statements = read_statements("r = {'A -> B'};  % 100% done\n")
        assert len(statements) == 1
        assert statements[0].text == "r = {'A -> B'}"

    def test_percent_inside_string_is_preserved(self):
        statements = read_statements("s = 'a % b';")
        assert statements[0].text == "s = 'a % b'"

    def test_doubled_quote_is_an_escape(self):
        statements = read_statements("s = 'it''s';\nt = 2;")
        assert [s.text for s in statements] == ["s = 'it''s'", "t = 2"]

    def test_separators_inside_brackets_are_kept(self):
        statements = read_statements("r = reaction({'A -> B'; 'B -> C'});")
        assert len(statements) == 1

    def test_tracks_original_line_numbers(self):
        source = "a = 1;\n% comment\n\nb = 2;\n"
        statements = read_statements(source)
        assert [(s.text, s.line) for s in statements] == [
            ("a = 1", 1),
            ("b = 2", 4),
        ]

    def test_continuation_spacer_lines_collapse(self):
        source = "r = reaction({...\n...\n... % header\n  'A -> B';...\n});"
        statements = read_statements(source)
        assert len(statements) == 1
        assert "'A -> B'" in statements[0].text


class TestMatlabEvaluator:
    """The MATLAB literal value grammar."""

    def test_numbers_and_scientific_notation(self):
        assert evaluate("1e-7") == pytest.approx(1e-7)
        assert evaluate("10^-6") == pytest.approx(1e-6)
        assert evaluate("8/60/60") == pytest.approx(8 / 3600)

    def test_vectors(self):
        assert evaluate("[0.75,0.25]") == [0.75, 0.25]
        assert evaluate("[1, 2, 3,]") == [1.0, 2.0, 3.0]
        assert evaluate("[1 2 3]") == [1.0, 2.0, 3.0]

    def test_range_with_step(self):
        assert evaluate("0:2:6") == [0.0, 2.0, 4.0, 6.0]
        assert evaluate("0:samp:(samp*2)", {"samp": 0.5}) == [0.0, 0.5, 1.0]

    def test_booleans_and_strings(self):
        assert evaluate("true") is True
        assert evaluate("false") is False
        assert evaluate("'h'") == "h"

    def test_member_access_and_broadcasting(self):
        env = {"f": {"val": [10.0, 20.0]}}
        assert evaluate("f.val/20", env) == [0.5, 1.0]

    def test_unknown_calls_do_not_raise(self):
        assert evaluate("normrnd(a,b)") is UNKNOWN
        assert evaluate("mod2stoich(m)'") is UNKNOWN
        assert evaluate("table2array(readtable('x.csv'))") is UNKNOWN

    def test_unknown_is_falsy(self):
        assert not UNKNOWN

    def test_split_call(self):
        assert split_call("reaction({'A -> B'})") == ("reaction", "{'A -> B'}")
        assert split_call("f(1) + g(2)") is None
        assert split_call("not_a_call") is None


class TestEquationGrammar:
    """Equation shapes taken from the benchmark corpus."""

    def _parse(self, equation, **kwargs):
        script = f"r = reaction({{'{equation}'}});\nm = model(r);\n"
        directory = Path(tempfile.mkdtemp())
        try:
            path = directory / "case.m"
            path.write_text(script + kwargs.get("extra", ""))
            return parse_inca(path)
        finally:
            shutil.rmtree(directory)

    def test_no_space_before_parenthesis(self):
        model = self._parse("Ru5P (abcde) + CO2(f) -> G3P (cde) + G3P (fba)")
        assert model.model.reactions["R1"].reactants == ["Ru5P", "CO2"]
        assert model.model.metabolites["CO2"].atoms == 1

    def test_no_space_around_plus(self):
        model = self._parse("2.039 OAA+  1.233 AKG -> Biomass")
        stoichiometry = model.model.reactions["R1"].stoichiometry_dict
        assert stoichiometry["OAA"] == pytest.approx(-2.039)
        assert stoichiometry["AKG"] == pytest.approx(-1.233)

    def test_unmapped_term_beside_mapped_terms(self):
        model = self._parse("OAA (abcd) + Dummy_Fum.ex -> Fum (abcd)")
        reaction = model.model.reactions["R1"]
        assert reaction.reactants == ["OAA", "Dummy_Fum.ex"]
        assert "R1" in model.model.atom_mappings
        assert model.model.metabolites["Dummy_Fum.ex"].atoms is None

    def test_wholly_unmapped_reaction_gets_no_atom_map(self):
        model = self._parse("Dummy -> Dummy.ext")
        assert "R1" not in model.model.atom_mappings
        assert len(model.model.reactions) == 1

    def test_repeated_metabolite_uses_instances(self):
        model = self._parse(
            "Pyr (abc) + Pyr (def) + Glu (ghijk) "
            "-> Val (abcef) + CO2 (d) + AKG (ghijk)"
        )
        notation = model.model.atom_mappings["R1"].to_letter_notation()
        # Two distinct Pyr instances contribute different letters.
        assert notation.count("Pyr(") == 2
        assert model.model.reactions["R1"].reactants.count("Pyr") == 2

    def test_dotted_metabolite_identifiers_kept_verbatim(self):
        model = self._parse("Gluc.ext (abcdef) -> G6P (abcdef)")
        assert "Gluc.ext" in model.model.metabolites
        assert model.model.metabolites["Gluc.ext"].compartment is None

    def test_reversibility_from_arrow(self):
        reversible = self._parse("A (a) <-> B (a)").model.reactions["R1"]
        irreversible = self._parse("A (a) -> B (a)").model.reactions["R1"]
        assert reversible.reversibility
        assert not irreversible.reversibility

    def test_malformed_equation_raises(self):
        with pytest.raises(ValueError, match="no arrow"):
            self._parse("A (a) B (a)")

    def test_conflicting_atom_counts_raise(self):
        script = (
            "r = reaction({'A (abc) -> B (abc)'; 'A (ab) -> C (ab)'});\n"
            "m = model(r);\n"
        )
        directory = Path(tempfile.mkdtemp())
        try:
            path = directory / "bad.m"
            path.write_text(script)
            with pytest.raises(ValueError, match="atoms"):
                parse_inca(path)
        finally:
            shutil.rmtree(directory)


@pytest.fixture(scope="module")
def parsed_ec():
    """Parse EC.m once for the whole module."""
    parser = INCAParser()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = parser.parse(EC_SCRIPT)
    return model, parser


@pytest.fixture(scope="module")
def parsed_syn():
    """Parse Syn.m once for the whole module."""
    parser = INCAParser()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = parser.parse(SYN_SCRIPT)
    return model, parser


@pytest.mark.skipif(not EC_SCRIPT.exists(), reason="EC.m benchmark not found")
class TestECBenchmark:
    """The E. coli benchmark model."""

    def test_network_counts(self, parsed_ec):
        model, _ = parsed_ec
        assert isinstance(model, FluxomicsData)
        # 68 reactions; biomass (v67) and the dummy drain (v68) carry no
        # atom transitions.
        assert len(model.model.reactions) == 68
        assert len(model.model.metabolites) == 66
        assert len(model.model.atom_mappings) == 66

    def test_reaction_ids_are_positional(self, parsed_ec):
        model, _ = parsed_ec
        assert model.model.reactions[0].id == "R1"
        assert model.model.ordered_reaction_ids[-1] == "R68"

    def test_metabolites_in_ascii_order(self, parsed_ec):
        model, _ = parsed_ec
        ids = model.model.ordered_metabolite_ids
        assert ids == sorted(ids)
        assert ids[:4] == ["AKG", "Ac", "Ac.ext", "AcCoA"]

    def test_flux_vector_split_into_net_and_exchange(self, parsed_ec):
        model, _ = parsed_ec
        values = model.experiments[0].simulation.variables.flux_values
        # 68 net values plus one exchange per reversible reaction.
        assert len(values) == 92
        r1 = {v.type: v for v in values if v.flux == "R1"}
        assert r1["net"].value == pytest.approx(
            77.5311428582956 - 17.1878684051923
        )
        assert r1["xch"].value == pytest.approx(17.1878684051923)
        # R2 is irreversible and gets no exchange flux.
        assert [v.type for v in values if v.flux == "R2"] == ["net"]

    def test_directional_fluxes_kept_as_annotations(self, parsed_ec):
        model, _ = parsed_ec
        annotations = {
            a.name: a.content for a in model.model.reactions["R1"].annotations
        }
        assert float(annotations["inca_flux_forward"]) == pytest.approx(
            77.5311428582956
        )
        assert float(annotations["inca_flux_backward"]) == pytest.approx(
            17.1878684051923
        )
        assert "G6P (abcdef) <-> F6P (abcdef)" in annotations["inca_equation"]

    def test_pool_sizes_zip_with_ascii_order(self, parsed_ec):
        model, _ = parsed_ec
        pools = model.experiments[0].simulation.variables.metabolitesize_values
        assert len(pools) == 66
        assert pools[0].metabolite == "AKG"
        assert pools[0].value == pytest.approx(8.41e-5)
        assert pools[3].metabolite == "AcCoA"
        assert pools[3].value == pytest.approx(1.80e-4)

    def test_ms_fragments(self, parsed_ec):
        model, _ = parsed_ec
        groups = model.experiments[0].measurement.model.labeling_measurement
        expressions = {g.id: g.expression.textual for g in groups.groups}
        assert len(expressions) == 33
        assert expressions["Ala2"] == "Ala[2,3]"
        assert expressions["AKG5"] == "AKG[1-5]"
        # Same fragment measured on two derivatives: both are kept.
        assert expressions["Asp2a"] == expressions["Asp2b"] == "Asp[1,2]"

    def test_flux_measurements(self, parsed_ec):
        model, _ = parsed_ec
        measurement = model.experiments[0].measurement
        ids = [f.id for f in measurement.model.flux_measurement.net_fluxes]
        assert ids == ["R61", "R68", "R64", "R66", "R67", "R62", "R65"]
        first = measurement.data.data[0]
        assert first.id == "R61"
        assert first.value == pytest.approx(92.5864)
        # f.std = f.val/20
        assert first.stddev == pytest.approx(92.5864 / 20)

    def test_tracers(self, parsed_ec):
        model, _ = parsed_ec
        tracers = model.experiments[0].tracers
        assert len(tracers) == 1
        assert tracers[0].metabolite == "Gluc.ext"
        patterns = {
            label.labeled_pattern: label.fraction for label in tracers[0].labels
        }
        assert patterns == {"100000": 0.75, "111111": 0.25}

    def test_symmetry_expanded_into_variants(self, parsed_ec):
        model, _ = parsed_ec
        variants = {
            rxn_id: sorted(transition.maps)
            for rxn_id, transition in model.model.atom_mappings.items()
            if len(transition.maps) > 1
        }
        # Every reaction touching the symmetric Suc or Fum pools.
        assert set(variants) == {
            "R24",
            "R25",
            "R26",
            "R38",
            "R47",
            "R50",
            "R57",
        }
        assert variants["R24"] == ["R24___1", "R24___2"]
        assert model.model.reactions["R24"].atom_transition_ids == [
            "R24___1",
            "R24___2",
        ]
        assert model.model.reactions["R24"].is_variant_reaction
        symmetric = {
            m.id for m in model.model.metabolites if m.involved_in_variants
        }
        assert symmetric == {"Suc", "Fum"}

    def test_instationary_options(self, parsed_ec):
        model, _ = parsed_ec
        experiment = model.experiments[0]
        # m.options.sim_ss = false with a non-empty int_tspan.
        assert experiment.stationary is False
        groups = experiment.measurement.model.labeling_measurement.groups
        assert groups[0].times is not None
        assert len(groups[0].times.split(",")) == 9

    def test_opaque_calls_are_reported_not_applied(self, parsed_ec):
        _, parser = parsed_ec
        joined = " | ".join(parser.warnings)
        assert "normrnd" in joined
        assert "mod2stoich" in joined

    def test_benchmark_harness_is_skipped(self, parsed_ec):
        _, parser = parsed_ec
        assert parser.skipped
        assert all("evalc" not in s.text for s in parser.skipped[:1] or [])


@pytest.mark.skipif(not SYN_SCRIPT.exists(), reason="Syn.m benchmark not found")
class TestSynBenchmark:
    """The Synechocystis benchmark model."""

    def test_network_counts(self, parsed_syn):
        model, _ = parsed_syn
        assert len(model.model.reactions) == 42
        assert len(model.model.metabolites) == 38
        # Only the biomass equation carries no atom transitions.
        assert len(model.model.atom_mappings) == 41

    def test_flux_and_pool_vector_lengths(self, parsed_syn):
        model, _ = parsed_syn
        variables = model.experiments[0].simulation.variables
        # 42 reactions + 22 reversible.
        assert len(variables.flux_values) == 64
        assert len(variables.metabolitesize_values) == 38

    def test_only_the_bound_msdata_is_imported(self, parsed_syn):
        model, _ = parsed_syn
        groups = model.experiments[0].measurement.model.labeling_measurement
        # The script defines d_min, d_sim and d_real but binds only d_min
        # via `x.data_ms = d_min`; the other two are dead code.
        assert len(groups.groups) == 1
        assert groups.groups[0].id == "G3P3"

    def test_tracer_on_unknown_metabolite_warns(self, parsed_syn):
        model, parser = parsed_syn
        joined = " | ".join(parser.warnings)
        assert "GO2.ex" in joined
        assert "GO2.ex" not in model.model.metabolites
        # The three well-formed tracers still come through.
        assert {t.metabolite for t in model.experiments[0].tracers} == {
            "CO2.ex",
            "CO20.ex",
            "CO21.ex",
        }

    def test_assignment_to_undefined_variable_warns(self, parsed_syn):
        _, parser = parsed_syn
        assert any(
            "undefined variable 'd'" in message for message in parser.warnings
        )


@pytest.mark.skipif(
    not POOLSIZES_SCRIPT.exists(), reason="EC_poolsizes.m not found"
)
class TestPoolsizesVariant:
    """The pool-size sweep variant of the E. coli model."""

    def test_matches_the_base_model_network(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = parse_inca(POOLSIZES_SCRIPT)
        assert len(model.model.reactions) == 68
        assert len(model.model.metabolites) == 66

    def test_external_csv_is_not_resolved(self):
        parser = INCAParser()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser.parse(POOLSIZES_SCRIPT)
        # `samples = table2array(readtable(...))` is recorded, not resolved.
        assert parser.skipped or parser.warnings


class TestParserErrors:
    """Failure modes that must not pass silently."""

    def _write(self, directory, body):
        path = Path(directory) / "case.m"
        path.write_text(body)
        return path

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            parse_inca("does_not_exist.m")

    def test_script_without_model_call(self, tmp_path):
        path = self._write(tmp_path, "r = reaction({'A (a) -> B (a)'});\n")
        with pytest.raises(ValueError, match="No model"):
            parse_inca(path)

    def test_wrong_flux_vector_length_raises(self, tmp_path):
        path = self._write(
            tmp_path,
            "r = reaction({'A (a) <-> B (a)'; 'B (a) -> C (a)'});\n"
            "m = model(r);\n"
            "m.rates.flx.val = [1, 2];\n",
        )
        with pytest.raises(ValueError, match="wrong reactions"):
            parse_inca(path)

    def test_wrong_pool_vector_length_raises(self, tmp_path):
        path = self._write(
            tmp_path,
            "r = reaction({'A (a) -> B (a)'});\n"
            "m = model(r);\n"
            "m.states.val = [1, 2, 3];\n",
        )
        with pytest.raises(ValueError, match="wrong metabolites"):
            parse_inca(path)


@pytest.mark.skipif(not EC_SCRIPT.exists(), reason="EC.m benchmark not found")
class TestFormatIntegration:
    """Entry-point wiring and cross-format conversion."""

    def test_extension_dispatch(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = parse(EC_SCRIPT)
        assert isinstance(model, FluxomicsData)
        assert model.metadata.name == "EC"

    def test_inca_to_fluxml_preserves_the_network(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            original = parse_inca(EC_SCRIPT)
            target = tmp_path / "EC_from_inca.fml"
            write_fluxml(original, target)
            reparsed = parse_fluxml_file(target)

        assert set(reparsed.model.ordered_metabolite_ids) == set(
            original.model.ordered_metabolite_ids
        )
        # Every INCA reaction survives.  The FluxML writer additionally
        # emits `<metabolite>_out` drains for sink pools, which FluxML
        # needs and INCA leaves implicit.
        assert set(original.model.reaction_ids) <= set(
            reparsed.model.reaction_ids
        )
        added = set(reparsed.model.reaction_ids) - set(
            original.model.reaction_ids
        )
        assert all(rxn_id.endswith("_out") for rxn_id in added)

        for rxn_id, transition in original.model.atom_mappings.items():
            assert rxn_id in reparsed.model.atom_mappings
            if len(transition.maps) == 1:
                assert len(reparsed.model.atom_mappings[rxn_id].maps) == 1

    @pytest.mark.xfail(
        strict=False,
        reason="FluxMLWriter._add_product_variants emits whole-reaction "
        "atom-map variants once per product, so reparsing recombines them "
        "combinatorially (R38: 2 variants x 4 products -> 16). A FluxML "
        "writer defect, independent of the INCA reader.",
    )
    def test_symmetry_variant_counts_survive_fluxml(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            original = parse_inca(EC_SCRIPT)
            target = tmp_path / "EC_variants.fml"
            write_fluxml(original, target)
            reparsed = parse_fluxml_file(target)

        # R38 produces Arg, AKG, Fum and Ac, and Fum is symmetric, so the
        # reaction has exactly two whole-reaction atom maps.
        assert len(original.model.atom_mappings["R38"].maps) == 2
        assert len(reparsed.model.atom_mappings["R38"].maps) == 2
