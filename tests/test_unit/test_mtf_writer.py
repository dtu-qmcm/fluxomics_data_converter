"""Tests for MTF (influx_si) format writer."""

import pytest
from pathlib import Path
import tempfile
import shutil

from fluxomics_data_converter.io import (
    parse_mtf,
    write_mtf,
    parse_fluxml_file,
    parse_freeflux,
)
from fluxomics_data_converter.core.core import FluxomicsData


SIMPLE_MODEL_MTF = (
    Path(__file__).parent.parent.parent
    / "data"
    / "simple_model"
    / "MTF"
    / "simple_model"
)
MTF_TOY = (
    Path(__file__).parent.parent.parent
    / "data"
    / "MTF_tests"
    / "toy_test"
    / "toy"
)
FREEFLUX_TOY = (
    Path(__file__).parent.parent.parent / "data" / "freeflux_tests" / "toy"
)
SIMPLE_MODEL_FML = (
    Path(__file__).parent.parent.parent
    / "data"
    / "simple_model"
    / "FluxML"
    / "simple_model.fml"
)


class TestMTFWriter:
    """Test MTF writer output."""

    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    @pytest.mark.skipif(
        not MTF_TOY.with_suffix(".netw").exists(),
        reason="MTF toy test data not found",
    )
    def test_write_creates_files(self, temp_dir):
        model = parse_mtf(str(MTF_TOY))
        base = temp_dir / "output"
        write_mtf(model, str(base), experiment_name="toy")

        assert base.with_suffix(".netw").exists()
        assert base.with_suffix(".linp").exists()
        assert base.with_suffix(".cnstr").exists()

    @pytest.mark.skipif(
        not MTF_TOY.with_suffix(".netw").exists(),
        reason="MTF toy test data not found",
    )
    def test_write_netw_content(self, temp_dir):
        model = parse_mtf(str(MTF_TOY))
        base = temp_dir / "output"
        write_mtf(model, str(base), experiment_name="toy")

        netw = base.with_suffix(".netw").read_text()
        assert "Network definition" in netw
        assert "V1" in netw

    @pytest.mark.skipif(
        not MTF_TOY.with_suffix(".netw").exists(),
        reason="MTF toy test data not found",
    )
    def test_write_linp_content(self, temp_dir):
        model = parse_mtf(str(MTF_TOY))
        base = temp_dir / "output"
        write_mtf(model, str(base), experiment_name="toy")

        linp = base.with_suffix(".linp").read_text()
        assert "A" in linp

    @pytest.mark.skipif(
        not MTF_TOY.with_suffix(".netw").exists(),
        reason="MTF toy test data not found",
    )
    def test_write_cnstr_content(self, temp_dir):
        model = parse_mtf(str(MTF_TOY))
        base = temp_dir / "output"
        write_mtf(model, str(base), experiment_name="toy")

        cnstr = base.with_suffix(".cnstr").read_text()
        assert "NET" in cnstr or "XCH" in cnstr

    @pytest.mark.skipif(
        not MTF_TOY.with_suffix(".netw").exists(),
        reason="MTF toy test data not found",
    )
    def test_write_multiple_experiments_requires_name(self, temp_dir):
        from fluxomics_data_converter.core.core import (
            MetabolicNetworkModel,
            LabelingExperiments,
        )
        from fluxomics_data_converter.model.metabolite import Metabolite
        from fluxomics_data_converter.model.reaction import Reaction

        m = MetabolicNetworkModel(
            metabolites=[Metabolite(id="A"), Metabolite(id="B")],
            reactions=[Reaction(id="r1", reactants=["A"], products=["B"])],
        )
        dm = FluxomicsData(
            model=m,
            experiments=[
                LabelingExperiments(name="exp1"),
                LabelingExperiments(name="exp2"),
            ],
        )
        with pytest.raises(ValueError):
            write_mtf(dm, str(temp_dir / "output"))


class TestMTFWriterFromFluxML:
    """Test writing MTF from a FluxML-parsed model."""

    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    @pytest.mark.skipif(
        not SIMPLE_MODEL_FML.exists(),
        reason="Simple model FML not found",
    )
    def test_fluxml_to_mtf(self, temp_dir):
        model = parse_fluxml_file(str(SIMPLE_MODEL_FML))
        base = temp_dir / "output"
        exp_name = model.experiments[0].name if model.experiments else None
        write_mtf(model, str(base), experiment_name=exp_name)
        assert base.with_suffix(".netw").exists()


class TestMTFWriterFromFreeflux:
    """Test writing MTF from a Freeflux-parsed model."""

    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    @pytest.mark.skipif(
        not FREEFLUX_TOY.exists(),
        reason="FreeFlux toy test data not found",
    )
    def test_freeflux_to_mtf(self, temp_dir):
        model = parse_freeflux(FREEFLUX_TOY)
        base = temp_dir / "output"
        exp_name = model.experiments[0].name if model.experiments else None
        write_mtf(model, str(base), experiment_name=exp_name)
        assert base.with_suffix(".netw").exists()


class TestMTFRoundTrip:
    """Test MTF -> MTF round-trip."""

    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    @pytest.mark.skipif(
        not MTF_TOY.with_suffix(".netw").exists(),
        reason="MTF toy test data not found",
    )
    def test_roundtrip_reactions(self, temp_dir):
        original = parse_mtf(str(MTF_TOY))
        base = temp_dir / "output"
        write_mtf(original, str(base), experiment_name="toy")
        converted = parse_mtf(str(base))

        # The writer adds drain reactions for sink metabolites, so
        # the original reactions must be a subset of the result.
        orig_ids = set(original.model.reactions.ids)
        conv_ids = set(converted.model.reactions.ids)
        assert orig_ids <= conv_ids

    @pytest.mark.skipif(
        not MTF_TOY.with_suffix(".netw").exists(),
        reason="MTF toy test data not found",
    )
    def test_roundtrip_metabolites(self, temp_dir):
        original = parse_mtf(str(MTF_TOY))
        base = temp_dir / "output"
        write_mtf(original, str(base), experiment_name="toy")
        converted = parse_mtf(str(base))

        # The writer adds _ext metabolites for drains.
        orig_ids = set(original.model.metabolites.ids)
        conv_ids = set(converted.model.metabolites.ids)
        assert orig_ids <= conv_ids

    @pytest.mark.skipif(
        not MTF_TOY.with_suffix(".netw").exists(),
        reason="MTF toy test data not found",
    )
    def test_roundtrip_atom_mappings(self, temp_dir):
        original = parse_mtf(str(MTF_TOY))
        base = temp_dir / "output"
        write_mtf(original, str(base), experiment_name="toy")
        converted = parse_mtf(str(base))

        # Drain reactions create extra atom mappings.
        assert len(original.model.atom_mappings) <= len(
            converted.model.atom_mappings
        )

    @pytest.mark.skipif(
        not MTF_TOY.with_suffix(".netw").exists(),
        reason="MTF toy test data not found",
    )
    def test_roundtrip_tracer_count(self, temp_dir):
        original = parse_mtf(str(MTF_TOY))
        base = temp_dir / "output"
        write_mtf(original, str(base), experiment_name="toy")
        converted = parse_mtf(str(base))

        orig_tracers = len(original.experiments[0].tracers)
        conv_tracers = len(converted.experiments[0].tracers)
        assert orig_tracers == conv_tracers

    @pytest.mark.skipif(
        not SIMPLE_MODEL_MTF.with_suffix(".netw").exists(),
        reason="Simple model MTF not found",
    )
    def test_roundtrip_simple_model_reactions(self, temp_dir):
        original = parse_mtf(str(SIMPLE_MODEL_MTF))
        base = temp_dir / "output"
        exp_name = (
            original.experiments[0].name if original.experiments else None
        )
        write_mtf(original, str(base), experiment_name=exp_name)
        converted = parse_mtf(str(base))

        assert set(original.model.reactions.ids) == set(
            converted.model.reactions.ids
        )
