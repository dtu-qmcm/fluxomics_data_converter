"""Tests for FluxML XML format parser."""

import pytest
from pathlib import Path

from fluxomics_data_converter.io import FluxMLParser, parse_fluxml_file
from fluxomics_data_converter.core.core import FluxomicsData


SIMPLE_MODEL = (
    Path(__file__).parent.parent.parent
    / "data"
    / "simple_model"
    / "FluxML"
    / "simple_model.fml"
)
EC_BENCHMARK = (
    Path(__file__).parent.parent.parent
    / "data"
    / "benchmark"
    / "FluxML"
    / "EC.fml"
)


class TestFluxMLParserBasic:
    """Basic FluxML parser functionality tests."""

    @pytest.mark.skipif(
        not SIMPLE_MODEL.exists(), reason="Simple model FML not found"
    )
    def test_parse_returns_fluxomics_data(self):
        model = parse_fluxml_file(str(SIMPLE_MODEL))
        assert isinstance(model, FluxomicsData)

    @pytest.mark.skipif(
        not SIMPLE_MODEL.exists(), reason="Simple model FML not found"
    )
    def test_parser_instance(self):
        parser = FluxMLParser()
        model = parser.parse(str(SIMPLE_MODEL))
        assert isinstance(model, FluxomicsData)

    @pytest.mark.skipif(
        not SIMPLE_MODEL.exists(), reason="Simple model FML not found"
    )
    def test_parse_file_alias(self):
        parser = FluxMLParser()
        model = parser.parse_file(str(SIMPLE_MODEL))
        assert isinstance(model, FluxomicsData)

    @pytest.mark.skipif(
        not SIMPLE_MODEL.exists(), reason="Simple model FML not found"
    )
    def test_metadata(self):
        model = parse_fluxml_file(str(SIMPLE_MODEL))
        assert model.metadata is not None
        assert model.metadata.date is not None
        assert "Freeflux" in model.metadata.comment

    @pytest.mark.skipif(
        not SIMPLE_MODEL.exists(), reason="Simple model FML not found"
    )
    def test_missing_file(self):
        with pytest.raises(ValueError):
            parse_fluxml_file("nonexistent.fml")


class TestFluxMLParserReactions:
    """Tests for reaction parsing from FluxML files."""

    @pytest.mark.skipif(
        not SIMPLE_MODEL.exists(), reason="Simple model FML not found"
    )
    def test_reaction_count(self):
        model = parse_fluxml_file(str(SIMPLE_MODEL))
        assert len(model.model.reactions) == 8

    @pytest.mark.skipif(
        not SIMPLE_MODEL.exists(), reason="Simple model FML not found"
    )
    def test_reaction_ids(self):
        model = parse_fluxml_file(str(SIMPLE_MODEL))
        ids = list(model.model.reactions.ids)
        assert ids == [
            "R1", "R2", "R3", "R4", "R5", "R6", "E_out", "F_out"
        ]

    @pytest.mark.skipif(
        not SIMPLE_MODEL.exists(), reason="Simple model FML not found"
    )
    def test_reaction_reversibility(self):
        model = parse_fluxml_file(str(SIMPLE_MODEL))
        for rxn in model.model.reactions:
            assert rxn.reversibility is False

    @pytest.mark.skipif(
        not SIMPLE_MODEL.exists(), reason="Simple model FML not found"
    )
    def test_reaction_reactants_products(self):
        model = parse_fluxml_file(str(SIMPLE_MODEL))
        r1 = model.model.reactions.get_by_id("R1")
        assert r1.reactants == ["A"]
        assert r1.products == ["B"]

        r5 = model.model.reactions.get_by_id("R5")
        assert r5.reactants == ["B", "C"]
        assert r5.products == ["D", "E", "E"]

    @pytest.mark.skipif(
        not SIMPLE_MODEL.exists(), reason="Simple model FML not found"
    )
    def test_atom_mappings(self):
        model = parse_fluxml_file(str(SIMPLE_MODEL))
        assert len(model.model.atom_mappings) > 0
        assert "R1" in model.model.atom_mappings

        r1_mapping = model.model.atom_mappings["R1"]
        assert r1_mapping.reactants == ["A"]
        assert r1_mapping.products == ["B"]

    @pytest.mark.skipif(
        not SIMPLE_MODEL.exists(), reason="Simple model FML not found"
    )
    def test_metabolites(self):
        model = parse_fluxml_file(str(SIMPLE_MODEL))
        met_ids = list(model.model.metabolites.ids)
        assert met_ids == ["A", "B", "D", "C", "E", "F"]

        a = model.model.metabolites.get_by_id("A")
        assert a.atoms == 3


class TestFluxMLParserConstraints:
    """Tests for constraint parsing from FluxML files."""

    @pytest.mark.skipif(
        not SIMPLE_MODEL.exists(), reason="Simple model FML not found"
    )
    def test_net_constraints(self):
        model = parse_fluxml_file(str(SIMPLE_MODEL))
        assert model.constraints is not None
        assert model.constraints.net is not None
        formulas = model.constraints.net.formulas
        assert len(formulas) == 16

    @pytest.mark.skipif(
        not SIMPLE_MODEL.exists(), reason="Simple model FML not found"
    )
    def test_exchange_constraints(self):
        model = parse_fluxml_file(str(SIMPLE_MODEL))
        assert model.constraints is not None
        assert model.constraints.xch is None


class TestFluxMLParserExperiments:
    """Tests for experiment parsing from FluxML files."""

    @pytest.mark.skipif(
        not SIMPLE_MODEL.exists(), reason="Simple model FML not found"
    )
    def test_experiment_count(self):
        model = parse_fluxml_file(str(SIMPLE_MODEL))
        assert len(model.experiments) == 1

    @pytest.mark.skipif(
        not SIMPLE_MODEL.exists(), reason="Simple model FML not found"
    )
    def test_experiment_name(self):
        model = parse_fluxml_file(str(SIMPLE_MODEL))
        assert model.experiments[0].name == "freeflux"
        assert model.experiments[0].stationary is True

    @pytest.mark.skipif(
        not SIMPLE_MODEL.exists(), reason="Simple model FML not found"
    )
    def test_tracers(self):
        model = parse_fluxml_file(str(SIMPLE_MODEL))
        exp = model.experiments[0]
        assert len(exp.tracers) == 1

        tracer = exp.tracers[0]
        assert tracer.metabolite == "A"
        assert tracer.type == "isotopomer"
        assert len(tracer.labels) == 1
        label = tracer.labels[0]
        assert label.labeled_pattern == "010"
        assert label.fraction == 1.0
        assert label.purity == 1.0

    @pytest.mark.skipif(
        not SIMPLE_MODEL.exists(), reason="Simple model FML not found"
    )
    def test_simulation(self):
        model = parse_fluxml_file(str(SIMPLE_MODEL))
        exp = model.experiments[0]
        assert exp.simulation is not None
        flux_values = exp.simulation.variables.flux_values
        assert len(flux_values) == 8

        r1_flux = next(f for f in flux_values if f.flux == "R1")
        assert r1_flux.type == "net"
        assert r1_flux.value == 100.0

    @pytest.mark.skipif(
        not SIMPLE_MODEL.exists(), reason="Simple model FML not found"
    )
    def test_measurement(self):
        model = parse_fluxml_file(str(SIMPLE_MODEL))
        exp = model.experiments[0]
        assert exp.measurement is not None

        lm = exp.measurement.model.labeling_measurement
        assert lm is not None
        assert len(lm.groups) == 1

        group = lm.groups[0]
        assert group.id == "F_123"
        assert group.expression.textual == "F[1,2,3]#M0,1,2,3"

    @pytest.mark.skipif(
        not SIMPLE_MODEL.exists(), reason="Simple model FML not found"
    )
    def test_flux_measurement(self):
        model = parse_fluxml_file(str(SIMPLE_MODEL))
        exp = model.experiments[0]
        fm = exp.measurement.model.flux_measurement
        assert fm is not None
        assert len(fm.net_fluxes) == 1
        assert fm.net_fluxes[0].id == "R1"

    @pytest.mark.skipif(
        not SIMPLE_MODEL.exists(), reason="Simple model FML not found"
    )
    def test_measurement_data(self):
        model = parse_fluxml_file(str(SIMPLE_MODEL))
        exp = model.experiments[0]
        data = exp.measurement.data.data
        assert len(data) == 5

        f_m0 = next(
            d for d in data if d.id == "F_123" and d.weight == "0"
        )
        assert f_m0.value == pytest.approx(6.14943422e-05)

        r1_datum = next(d for d in data if d.id == "R1")
        assert r1_datum.value == pytest.approx(100.0)
        assert r1_datum.stddev == pytest.approx(0.1)


class TestFluxMLParserBenchmark:
    """Tests using the larger EC benchmark file."""

    @pytest.mark.skipif(
        not EC_BENCHMARK.exists(), reason="EC benchmark FML not found"
    )
    def test_parse_ec(self):
        model = parse_fluxml_file(str(EC_BENCHMARK))
        assert isinstance(model, FluxomicsData)
        assert len(model.model.reactions) > 0
        assert len(model.model.metabolites) > 0
        assert len(model.experiments) > 0

    @pytest.mark.skipif(
        not EC_BENCHMARK.exists(), reason="EC benchmark FML not found"
    )
    def test_ec_has_atom_mappings(self):
        model = parse_fluxml_file(str(EC_BENCHMARK))
        assert len(model.model.atom_mappings) > 0

    @pytest.mark.skipif(
        not EC_BENCHMARK.exists(), reason="EC benchmark FML not found"
    )
    def test_ec_has_constraints(self):
        model = parse_fluxml_file(str(EC_BENCHMARK))
        assert model.constraints is not None
