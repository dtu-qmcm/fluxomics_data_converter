"""Tests for FluxML XML writer."""

import pytest
from pathlib import Path
import tempfile
import shutil

from fluxomics_data_converter.io import (
    parse_fluxml_file,
    write_fluxml,
    parse_mtf,
    parse_freeflux,
)


SIMPLE_MODEL_FML = (
    Path(__file__).parent.parent.parent
    / "data"
    / "simple_model"
    / "FluxML"
    / "simple_model.fml"
)
SIMPLE_MODEL_MTF = (
    Path(__file__).parent.parent.parent
    / "data"
    / "simple_model"
    / "MTF"
    / "simple_model"
)
FREEFLUX_TOY = (
    Path(__file__).parent.parent.parent / "data" / "freeflux_tests" / "toy"
)
EC_BENCHMARK = (
    Path(__file__).parent.parent.parent
    / "data"
    / "benchmark"
    / "FluxML"
    / "EC.fml"
)


class TestFluxMLWriter:
    """Test FluxML writer output."""

    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    @pytest.mark.skipif(
        not SIMPLE_MODEL_FML.exists(),
        reason="Simple model FML not found",
    )
    def test_write_creates_file(self, temp_dir):
        model = parse_fluxml_file(str(SIMPLE_MODEL_FML))
        out = temp_dir / "output.fml"
        write_fluxml(model, str(out))
        assert out.exists()

    @pytest.mark.skipif(
        not SIMPLE_MODEL_FML.exists(),
        reason="Simple model FML not found",
    )
    def test_write_xml_structure(self, temp_dir):
        model = parse_fluxml_file(str(SIMPLE_MODEL_FML))
        out = temp_dir / "output.fml"
        write_fluxml(model, str(out))

        content = out.read_text()
        assert "<fluxml" in content
        assert "reactionnetwork" in content
        assert "configuration" in content

    @pytest.mark.skipif(
        not SIMPLE_MODEL_FML.exists(),
        reason="Simple model FML not found",
    )
    def test_write_preserves_reactions(self, temp_dir):
        original = parse_fluxml_file(str(SIMPLE_MODEL_FML))
        out = temp_dir / "output.fml"
        write_fluxml(original, str(out))
        converted = parse_fluxml_file(str(out))

        assert set(original.model.reactions.ids) == set(
            converted.model.reactions.ids
        )

    @pytest.mark.skipif(
        not SIMPLE_MODEL_FML.exists(),
        reason="Simple model FML not found",
    )
    def test_write_preserves_metabolites(self, temp_dir):
        original = parse_fluxml_file(str(SIMPLE_MODEL_FML))
        out = temp_dir / "output.fml"
        write_fluxml(original, str(out))
        converted = parse_fluxml_file(str(out))

        assert set(original.model.metabolites.ids) == set(
            converted.model.metabolites.ids
        )

    @pytest.mark.skipif(
        not SIMPLE_MODEL_FML.exists(),
        reason="Simple model FML not found",
    )
    def test_write_preserves_atom_mappings(self, temp_dir):
        original = parse_fluxml_file(str(SIMPLE_MODEL_FML))
        out = temp_dir / "output.fml"
        write_fluxml(original, str(out))
        converted = parse_fluxml_file(str(out))

        assert len(original.model.atom_mappings) == len(
            converted.model.atom_mappings
        )

    @pytest.mark.skipif(
        not SIMPLE_MODEL_FML.exists(),
        reason="Simple model FML not found",
    )
    def test_write_preserves_simulation(self, temp_dir):
        original = parse_fluxml_file(str(SIMPLE_MODEL_FML))
        out = temp_dir / "output.fml"
        write_fluxml(original, str(out))
        converted = parse_fluxml_file(str(out))

        orig_fluxes = {
            f.flux: f.value
            for f in original.experiments[0].simulation.variables.flux_values
        }
        conv_fluxes = {
            f.flux: f.value
            for f in converted.experiments[0].simulation.variables.flux_values
        }
        assert orig_fluxes == conv_fluxes

    @pytest.mark.skipif(
        not SIMPLE_MODEL_FML.exists(),
        reason="Simple model FML not found",
    )
    def test_write_preserves_tracers(self, temp_dir):
        original = parse_fluxml_file(str(SIMPLE_MODEL_FML))
        out = temp_dir / "output.fml"
        write_fluxml(original, str(out))
        converted = parse_fluxml_file(str(out))

        orig_tracers = {t.metabolite for t in original.experiments[0].tracers}
        conv_tracers = {t.metabolite for t in converted.experiments[0].tracers}
        assert orig_tracers == conv_tracers

    @pytest.mark.skipif(
        not SIMPLE_MODEL_FML.exists(),
        reason="Simple model FML not found",
    )
    def test_write_preserves_measurement_groups(self, temp_dir):
        original = parse_fluxml_file(str(SIMPLE_MODEL_FML))
        out = temp_dir / "output.fml"
        write_fluxml(original, str(out))
        converted = parse_fluxml_file(str(out))

        orig_groups = {
            g.id
            for g in original.experiments[
                0
            ].measurement.model.labeling_measurement.groups
        }
        conv_groups = {
            g.id
            for g in converted.experiments[
                0
            ].measurement.model.labeling_measurement.groups
        }
        assert orig_groups == conv_groups


class TestFluxMLWriterFromMTF:
    """Test writing FluxML from an MTF-parsed model."""

    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    @pytest.mark.skipif(
        not SIMPLE_MODEL_MTF.with_suffix(".netw").exists(),
        reason="Simple model MTF not found",
    )
    def test_mtf_to_fluxml(self, temp_dir):
        model = parse_mtf(str(SIMPLE_MODEL_MTF))
        out = temp_dir / "from_mtf.fml"
        write_fluxml(model, str(out))
        assert out.exists()

        converted = parse_fluxml_file(str(out))
        assert len(converted.model.reactions) > 0
        assert len(converted.model.metabolites) > 0


class TestFluxMLWriterFromFreeflux:
    """Test writing FluxML from a Freeflux-parsed model."""

    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    @pytest.mark.skipif(
        not FREEFLUX_TOY.exists(),
        reason="FreeFlux toy test data not found",
    )
    def test_freeflux_to_fluxml(self, temp_dir):
        model = parse_freeflux(FREEFLUX_TOY)
        out = temp_dir / "from_freeflux.fml"
        write_fluxml(model, str(out))
        assert out.exists()

        converted = parse_fluxml_file(str(out))
        assert len(converted.model.reactions) > 0


class TestFluxMLRoundTrip:
    """Test FluxML -> FluxML round-trip."""

    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    @pytest.mark.skipif(
        not SIMPLE_MODEL_FML.exists(),
        reason="Simple model FML not found",
    )
    def test_roundtrip_preserves_reaction_count(self, temp_dir):
        original = parse_fluxml_file(str(SIMPLE_MODEL_FML))
        out = temp_dir / "roundtrip.fml"
        write_fluxml(original, str(out))
        converted = parse_fluxml_file(str(out))
        assert len(converted.model.reactions) == len(original.model.reactions)

    @pytest.mark.skipif(
        not SIMPLE_MODEL_FML.exists(),
        reason="Simple model FML not found",
    )
    def test_roundtrip_preserves_experiment_name(self, temp_dir):
        original = parse_fluxml_file(str(SIMPLE_MODEL_FML))
        out = temp_dir / "roundtrip.fml"
        write_fluxml(original, str(out))
        converted = parse_fluxml_file(str(out))
        assert converted.experiments[0].name == original.experiments[0].name

    @pytest.mark.skipif(
        not EC_BENCHMARK.exists(), reason="EC benchmark FML not found"
    )
    def test_roundtrip_ec_metabolites(self, temp_dir):
        original = parse_fluxml_file(str(EC_BENCHMARK))
        out = temp_dir / "ec_roundtrip.fml"
        write_fluxml(original, str(out))
        converted = parse_fluxml_file(str(out))

        orig_mets = set(original.model.metabolites.ids)
        conv_mets = set(converted.model.metabolites.ids)
        assert orig_mets == conv_mets


# ---------------------------------------------------------------------------
# 13CFLUX compatibility: variants, flux values, purity, flux fill
# ---------------------------------------------------------------------------

import warnings  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime  # noqa: E402
from math import prod  # noqa: E402

FML_NS = "{http://www.13cflux.net/fluxml}"
SYN_BENCHMARK = EC_BENCHMARK.parent / "Syn.fml"

# A TCA fragment with the glyoxylate shunt. ICL gives a symmetric succinate
# next to a glyoxylate with a single mapping; SDH turns symmetric succinate
# into symmetric fumarate; FH turns it into malate, which is not symmetric.
# The tracer enters through an uptake reaction, because 13CFLUX does not
# allow an input pool next to an internal pool on a reaction's educt side.
SYMMETRY_REACTIONS = [
    "#reaction_ID\treactant_IDs(atom)\tproduct_IDs(atom)\treversibility",
    "ACup\tAcEx(ab)\tAcCoA(ab)\t0",
    "CS\tAcCoA(ab) + Oxa(cdef)\tIsoCit(fedbac)\t0",
    "ICL\tIsoCit(abcdef)\tGlx(ab) + Suc(fcde,edcf)\t0",
    "SDH\tSuc(abcd,dcba)\tFum(abcd,dcba)\t1",
    "FH\tFum(abcd,dcba)\tMal(abcd)\t1",
    "MALS\tGlx(ab) + AcCoA(cd)\tMal(abdc)\t0",
    "MDH\tMal(abcd)\tOxa(abcd)\t1",
    "OAAout\tOxa(abcd)\tOxaEx(abcd)\t0",
]
SYMMETRY_FLUXES = [
    "#flux_ID\tvalue",
    "ACup\t20",
    "CS\t10",
    "ICL\t10",
    "SDH_f\t12",
    "SDH_b\t2",
    "FH_f\t11",
    "FH_b\t1",
    "MALS\t10",
    "MDH_f\t20",
    "MDH_b\t0",
    "OAAout\t10",
]
SYMMETRY_MDVS = [
    "#fragment_ID\tmean\tsd",
    "Mal_1234\t0.4, 0.3, 0.2, 0.05, 0.05\t0.01, 0.01, 0.01, 0.01, 0.01",
]
TRACER_NO_PURITY = [
    "#metabolite_ID\tlabeling_pattern\tpercentage",
    "AcEx\t'11'\t1",
]

# Two parallel irreversible reactions B -> C: the split between them is not
# determined by the mass balance, everything downstream is.
BRANCH_REACTIONS = [
    "#reaction_ID\treactant_IDs(atom)\tproduct_IDs(atom)\treversibility",
    "v1\tA(ab)\tB(ab)\t0",
    "v2\tB(ab)\tC(ab)\t0",
    "v3\tB(ab)\tC(ba)\t0",
    "v4\tC(ab)\tD(ab)\t0",
]
BRANCH_TRACER = ["#metabolite_ID\tlabeling_pattern\tpercentage", "A\t'10'\t1"]


def _freeflux_model(tmp_path, reactions, fluxes, label_input, mdvs=None):
    """Write a small Freeflux model to *tmp_path* and parse it."""
    src = tmp_path / "model"
    src.mkdir()
    files = {
        "reactions.tsv": reactions,
        "fluxes.tsv": fluxes,
        "label_input.tsv": label_input,
    }
    if mdvs:
        files["measured_MDVs.tsv"] = mdvs
    for name, lines in files.items():
        (src / name).write_text("\n".join(lines) + "\n")
    return parse_freeflux(src)


def _write(model, path, **kwargs):
    """Write FluxML, returning the parsed XML root and warning messages."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_fluxml(model, str(path), **kwargs)
    return ET.parse(path).getroot(), [str(w.message) for w in caught]


def _reaction(root, base_id):
    """The <reaction> element whose id list belongs to *base_id*."""
    for rxn in root.iter(f"{FML_NS}reaction"):
        if rxn.get("id").split()[0].split("___")[0] == base_id:
            return rxn
    raise KeyError(base_id)


def _participant(rxn, tag, compound):
    return next(
        p for p in rxn.findall(f"{FML_NS}{tag}") if p.get("id") == compound
    )


def _variant_cfgs(elem):
    return [v.get("cfg") for v in elem.findall(f"{FML_NS}variant")]


def _fluxvalues(root):
    return {
        (fv.get("flux"), fv.get("type")): float(fv.text)
        for fv in root.iter(f"{FML_NS}fluxvalue")
    }


def _transitions(model, reaction_id):
    """A reaction's atom maps as a multiset of comparable transitions."""
    maps = model.model.atom_mappings[reaction_id].maps.values()
    return Counter(
        frozenset(
            ((p.mol, p.instance, p.index), (s.mol, s.instance, s.index))
            for p, s in atom_map.mapping.items()
        )
        for atom_map in maps
    )


@pytest.fixture
def symmetric_model(tmp_path):
    return _freeflux_model(
        tmp_path,
        SYMMETRY_REACTIONS,
        SYMMETRY_FLUXES,
        TRACER_NO_PURITY,
        SYMMETRY_MDVS,
    )


class TestVariantReactions:
    """13CFLUX reads <variant> lists as a cross product over participants."""

    def test_id_count_equals_variant_cross_product(
        self, symmetric_model, tmp_path
    ):
        root, _ = _write(symmetric_model, tmp_path / "out.fml")
        for rxn in root.iter(f"{FML_NS}reaction"):
            counts = [
                max(1, len(_variant_cfgs(p)))
                for p in rxn
                if p.tag in (f"{FML_NS}reduct", f"{FML_NS}rproduct")
            ]
            assert prod(counts) == len(rxn.get("id").split()), rxn.get("id")

    def test_icl_symmetric_product_and_plain_coproduct(
        self, symmetric_model, tmp_path
    ):
        root, _ = _write(symmetric_model, tmp_path / "out.fml")
        icl = _reaction(root, "ICL")
        assert icl.get("id").split() == ["ICL___1", "ICL___2"]
        glx = _participant(icl, "rproduct", "Glx")
        assert glx.get("cfg") == "C#1@1 C#2@1"
        assert _variant_cfgs(glx) == []
        suc = _participant(icl, "rproduct", "Suc")
        assert len(set(_variant_cfgs(suc))) == 2

    def test_sdh_symmetric_to_symmetric(self, symmetric_model, tmp_path):
        root, _ = _write(symmetric_model, tmp_path / "out.fml")
        sdh = _reaction(root, "SDH")
        assert len(sdh.get("id").split()) == 4
        suc = _participant(sdh, "reduct", "Suc")
        fum = _participant(sdh, "rproduct", "Fum")
        assert len(set(_variant_cfgs(suc))) == 2
        assert len(set(_variant_cfgs(fum))) == 2

    def test_fluxvalues_name_reaction_ids(self, symmetric_model, tmp_path):
        root, _ = _write(symmetric_model, tmp_path / "out.fml")
        reaction_ids = {
            i
            for rxn in root.iter(f"{FML_NS}reaction")
            for i in rxn.get("id").split()
        }
        values = _fluxvalues(root)
        assert {flux for flux, _ in values} <= reaction_ids
        # SDH: net 10 and xch 2, split over its four variant reactions
        for i in range(1, 5):
            assert values[(f"SDH___{i}", "net")] == pytest.approx(2.5)
            assert values[(f"SDH___{i}", "xch")] == pytest.approx(0.5)
        assert values[("ICL___1", "net")] == pytest.approx(5.0)
        assert ("SDH", "net") not in values

    def test_variants_round_trip(self, symmetric_model, tmp_path):
        out = tmp_path / "out.fml"
        _write(symmetric_model, out)
        back = parse_fluxml_file(str(out))
        for rxn_id in ("ICL", "SDH", "FH", "CS", "MALS"):
            assert _transitions(back, rxn_id) == _transitions(
                symmetric_model, rxn_id
            ), rxn_id

    @pytest.mark.skipif(
        not SYN_BENCHMARK.exists(), reason="Syn benchmark FML not found"
    )
    def test_reduct_variants_round_trip(self, tmp_path):
        """Syn.fml puts variants on reducts (sdh, fum); they must survive."""
        original = parse_fluxml_file(str(SYN_BENCHMARK))
        # fum: Fum(2 reduct variants) -> Mal is two distinct transitions
        assert len(_transitions(original, "fum")) == 2
        out = tmp_path / "syn.fml"
        _write(original, out, natural_abundance=True)
        back = parse_fluxml_file(str(out))
        for rxn_id in original.model.atom_mappings:
            assert _transitions(back, rxn_id) == _transitions(
                original, rxn_id
            ), rxn_id


class TestPurityAndNaturalAbundance:
    """purity="1.0" switches natural abundance on in 13CFLUX."""

    def _labels(self, root):
        return list(root.iter(f"{FML_NS}label"))

    def test_no_purity_in_input_no_attribute(self, symmetric_model, tmp_path):
        tracer = symmetric_model.experiments[0].tracers[0]
        assert tracer.labels[0].purity is None
        root, _ = _write(symmetric_model, tmp_path / "out.fml")
        assert all(lab.get("purity") is None for lab in self._labels(root))

    def test_purity_one_not_written_by_default(self, tmp_path):
        model = _freeflux_model(
            tmp_path,
            SYMMETRY_REACTIONS,
            SYMMETRY_FLUXES,
            [
                "#metabolite_ID\tlabeling_pattern\tpercentage\tpurity",
                "AcEx\t'11'\t1\t1",
            ],
        )
        root, messages = _write(model, tmp_path / "out.fml")
        assert all(lab.get("purity") is None for lab in self._labels(root))
        assert not any("purity" in m.lower() for m in messages)

    def test_natural_abundance_writes_purity(self, symmetric_model, tmp_path):
        root, _ = _write(
            symmetric_model, tmp_path / "out.fml", natural_abundance=True
        )
        assert [lab.get("purity") for lab in self._labels(root)] == ["1.0"]

    def test_impure_label_dropped_with_warning(self, tmp_path):
        model = _freeflux_model(
            tmp_path,
            SYMMETRY_REACTIONS,
            SYMMETRY_FLUXES,
            [
                "#metabolite_ID\tlabeling_pattern\tpercentage\tpurity",
                "AcEx\t'11'\t1\t0.99",
            ],
        )
        root, messages = _write(model, tmp_path / "out.fml")
        assert all(lab.get("purity") is None for lab in self._labels(root))
        assert any("purity below 1 dropped" in m for m in messages)
        root, _ = _write(model, tmp_path / "na.fml", natural_abundance=True)
        assert [lab.get("purity") for lab in self._labels(root)] == ["0.99"]

    def test_na_correction_without_natural_abundance_warns(
        self, symmetric_model, tmp_path
    ):
        _, messages = _write(
            symmetric_model, tmp_path / "out.fml", apply_na_correction=True
        )
        assert any("natural_abundance=True" in m for m in messages)


class TestFluxFill:
    """Missing net fluxes are derived only when the mass balance fixes them."""

    def test_drain_flux_derived(self, symmetric_model, tmp_path):
        root, messages = _write(symmetric_model, tmp_path / "out.fml")
        assert _fluxvalues(root)[("OxaEx_out", "net")] == pytest.approx(10.0)
        assert not any("not determined" in m for m in messages)

    def test_underdetermined_flux_not_written(self, tmp_path):
        model = _freeflux_model(
            tmp_path,
            BRANCH_REACTIONS,
            ["#flux_ID\tvalue", "v1\t10"],
            BRANCH_TRACER,
        )
        root, messages = _write(model, tmp_path / "out.fml")
        values = _fluxvalues(root)
        assert ("v2", "net") not in values
        assert ("v3", "net") not in values
        assert values[("v4", "net")] == pytest.approx(10.0)
        assert values[("D_out", "net")] == pytest.approx(10.0)
        assert any(
            "not determined" in m and "'v2'" in m and "'v3'" in m
            for m in messages
        )

    def test_negative_irreversible_flux_raises(self, tmp_path):
        model = _freeflux_model(
            tmp_path,
            BRANCH_REACTIONS,
            ["#flux_ID\tvalue", "v1\t10", "v2\t15"],
            BRANCH_TRACER,
        )
        with pytest.raises(ValueError, match="negative.*v3"):
            write_fluxml(model, str(tmp_path / "out.fml"))


class TestInputsAndMetadata:
    def test_unlabelled_source_gets_input(self, tmp_path):
        model = _freeflux_model(
            tmp_path,
            [
                "#reaction_ID\treactant_IDs(atom)\tproduct_IDs(atom)"
                "\treversibility",
                "v1\tA(ab)\tB(ab)\t0",
                "v2\tB(ab) + CO2(c)\tC(abc)\t0",
            ],
            ["#flux_ID\tvalue", "v1\t1", "v2\t1"],
            BRANCH_TRACER,
        )
        root, messages = _write(model, tmp_path / "out.fml")
        inputs = {
            i.get("pool"): [lab.get("cfg") for lab in i]
            for i in root.iter(f"{FML_NS}input")
        }
        assert inputs == {"A": ["10"], "CO2": ["0"]}
        assert any("unlabelled inputs" in m and "CO2" in m for m in messages)

    def test_dates_use_metadata_date(self, symmetric_model, tmp_path):
        date = datetime(2024, 1, 2, 3, 4, 5)
        model = symmetric_model.model_copy(
            update={
                "metadata": symmetric_model.metadata.model_copy(
                    update={"date": date}
                )
            }
        )
        root, _ = _write(model, tmp_path / "out.fml")
        dates = {d.text for d in root.iter(f"{FML_NS}date")}
        assert dates == {"2024-01-02 03:04:05"}


class TestX3cflux:
    """Parse the written file with 13CFLUX itself (Linux only)."""

    def test_x3cflux_reads_written_file(self, symmetric_model, tmp_path):
        x3cflux = pytest.importorskip("x3cflux")
        out = tmp_path / "out.fml"
        write_fluxml(symmetric_model, str(out))
        data = x3cflux.FluxMLParser().parse(str(out))
        x3cflux.create_simulator_from_data(
            data.network_data, data.configurations[0]
        )
