"""Tests for the record's model copy (model <-> record) and equivalence.

Skipped as a whole when the optional ``linkml`` extra (``fluxolink``) is
not installed.
"""

import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

pytest.importorskip("fluxolink")

import yaml  # noqa: E402
from fluxolink.datamodel import fluxolink as fl  # noqa: E402

from fluxomics_data_converter.core.common import TextualOrMath  # noqa: E402
from fluxomics_data_converter.core.core import (  # noqa: E402
    FluxomicsData,
    LabelingExperiments,
    MetabolicNetworkModel,
)
from fluxomics_data_converter.experiment.measurement import (  # noqa: E402
    Datum,
    Group,
    LabelingMeasurement,
    Measurement,
    MeasurementData,
    MeasurementModel,
)
from fluxomics_data_converter.experiment.tracer import (  # noqa: E402
    LabelComposition,
    Tracers,
)
from fluxomics_data_converter.link import (  # noqa: E402
    from_fluxo_dataset,
    load_record,
    save_record,
    to_fluxo_dataset,
)
from fluxomics_data_converter.link.equivalence import (  # noqa: E402
    canonical_constraint,
    compare,
)
from fluxomics_data_converter.io import (  # noqa: E402
    parse_fluxml_file,
    parse_freeflux,
    parse_inca,
    parse_mtf,
    write_fluxml,
    write_freeflux,
    write_mtf,
)
from fluxomics_data_converter.model.atom_mapping import (  # noqa: E402
    AtomAddress,
    AtomMap,
)
from fluxomics_data_converter.model.metabolite import Metabolite  # noqa: E402
from fluxomics_data_converter.model.reaction import Reaction  # noqa: E402

REPO = Path(__file__).parent.parent.parent
DATA = REPO / "data"
FLUXML = DATA / "benchmark" / "FluxML"
EC_FML = FLUXML / "EC.fml"
SCHEMA = (
    REPO.parent
    / "fluxomics_linkml"
    / "src"
    / "fluxolink"
    / "schema"
    / "fluxolink.yaml"
)

#: Sidecar used for the published EC benchmark dataset.
EC_SIDECAR = {
    "investigation": {
        "id": "ec_fluxml_benchmark.investigation",
        "name": "E. coli 13C-MFA benchmark (INCA 2.3 model)",
        "description": (
            "13C-MFA benchmark on the Escherichia coli network model that "
            "ships with INCA 2.3 (EC.fml <info><comment>: 'E. Coli Model "
            "Shipped with INCA 2.3'), with four FluxML configurations "
            "(a_INST, b_INST, a_STAT, b_STAT) for a 75 % [1-13C] / 25 % "
            "[U-13C] glucose tracer."
        ),
        "investigation_type": "MFA",
    }
}


def _quiet(func, *args, **kwargs):
    """Call a parser or the adapter with its expected warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args, **kwargs)


def _round_trip(data, tmp_path, name, source_path):
    """Materialise, dump, re-load with Pydantic and hydrate."""
    ds = _quiet(
        to_fluxo_dataset,
        data,
        dataset_id=name,
        source_path=source_path,
        data_origin="synthetic",
    )
    path = save_record(ds, tmp_path / f"{name}.yaml")
    reloaded = load_record(path)
    assert isinstance(reloaded, fl.FluxoDataset)
    return ds, from_fluxo_dataset(reloaded)


def _only(data, name):
    """Copy of a model with a single experiment."""
    return data.model_copy(update={"experiments": [data.get_experiments(name)]})


@pytest.fixture(scope="module")
def ec():
    return _quiet(parse_fluxml_file, str(EC_FML))


# -- round trips ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["EC", "linea", "Syn"])
def test_fluxml_round_trip(name, tmp_path):
    path = FLUXML / f"{name}.fml"
    original = _quiet(parse_fluxml_file, str(path))
    _, back = _round_trip(original, tmp_path, name, path)
    report = compare(original, back)
    assert report.equivalent, report.summary()


@pytest.mark.parametrize("name", ["EC", "linea", "Syn"])
def test_fluxml_writer_round_trip(name, tmp_path):
    """write_fluxml of the hydrated model re-parses to the same problem.

    The differences allowed are FluxMLWriter's own behaviour, not losses
    of the adapter: it adds drain reactions for sink metabolites (EC3 in
    EC.fml), and it does not write experiment-level <constraints> or
    <poolsizemeasurement> (Syn.fml). A ``purity`` attribute on a label
    switches natural abundance on, so the faithful write passes
    ``natural_abundance=True`` exactly when the source labels carry one
    (EC.fml, Syn.fml) and keeps the default ideal inputs otherwise
    (linea.fml).
    """
    path = FLUXML / f"{name}.fml"
    original = _quiet(parse_fluxml_file, str(path))
    _, back = _round_trip(original, tmp_path, name, path)
    natural_abundance = any(
        label.purity is not None
        for experiment in original.experiments
        for tracer in experiment.tracers
        for label in tracer.labels
    )
    out = tmp_path / f"{name}_written.fml"
    write_fluxml(back, str(out), natural_abundance=natural_abundance)
    again = _quiet(parse_fluxml_file, str(out))
    report = compare(original, again)
    if name == "EC":
        assert [(d.kind, d.key) for d in report.differences] == [
            ("reaction", "EC3_out")
        ]
    elif name == "linea":
        assert report.equivalent, report.summary()
    else:
        assert set(report.counts()) <= {"constraint", "pool_size_measurement"}
        assert compare(
            original, again, ignore=("constraint", "pool_size_measurement")
        ).equivalent


@pytest.mark.parametrize(
    "prefix",
    [
        DATA / "simple_model" / "MTF" / "simple_model",
        DATA / "MTF_tests" / "toy_test" / "toy",
    ],
    ids=["simple_model", "toy"],
)
def test_mtf_round_trip(prefix, tmp_path):
    original = _quiet(parse_mtf, str(prefix))
    ds, back = _round_trip(original, tmp_path, prefix.name, prefix)
    report = compare(original, back)
    if prefix.name == "toy":
        # The MTF parser drops "X ()" participants, so V17-V19 arrive with
        # no reactants; NetworkReaction needs one, so they are skipped.
        assert [(d.kind, d.key) for d in report.differences] == [
            ("reaction", "V17"),
            ("reaction", "V18"),
            ("reaction", "V19"),
        ]
        # toy.tvar has 8 columns; the MTF parser reads 6 by position and
        # gets no variables, so there is no simulation run to publish.
        assert ds.simulation_runs is None
    else:
        assert report.equivalent, report.summary()
        run = ds.simulation_runs[0]
        prefix_option = {o.key: o.value for o in run.solver_options}
        assert prefix_option["--prefix"] == prefix.name
        assert run.software.software_name == "influx_si"
        assert {v.variable_type for v in run.variables} == {
            "free",
            "dependent",
        }
        # The MTF parser's -1000..1000 placeholder bounds are not data.
        assert all(v.lower_bound is None for v in run.variables)
    roles = {Path(f.path).suffix: f.role for f in ds.source_files}
    assert roles[".netw"] == ["network"]
    assert roles[".miso"] == ["mid_measurements"]
    assert roles[".tvar"] == ["variable_specification"]
    assert roles[".opt"] == ["configuration"]


def test_freeflux_round_trip(tmp_path):
    path = DATA / "simple_model" / "freeflux"
    original = _quiet(parse_freeflux, str(path))
    ds, back = _round_trip(original, tmp_path, "freeflux", path)
    assert compare(original, back).equivalent
    assert ds.source_files[0].format == "freeflux_tsv"
    assert ds.simulation_runs[0].software.software_name == "freeflux"


def test_inca_round_trip(tmp_path):
    path = DATA / "benchmark" / "INCA" / "EC.m"
    original = _quiet(parse_inca, str(path))
    ds, back = _round_trip(original, tmp_path, "inca", path)
    report = compare(original, back)
    assert report.equivalent, report.summary()
    assert ds.experiments[0].time_unit == "h"
    assert "driver_script" in ds.source_files[0].role
    # INCA keeps "Pyr (abc) + Pyr (def)" as two participants.
    r51 = next(r for r in ds.networks[0].reactions if r.id == "R51")
    assert [(p.metabolite_id, p.instance) for p in r51.reactants][:2] == [
        ("Pyr", 1),
        ("Pyr", 2),
    ]


def test_hydrated_model_drives_writers(ec, tmp_path):
    _, back = _round_trip(ec, tmp_path, "ec", EC_FML)
    write_mtf(back, str(tmp_path / "mtf" / "EC"), experiment_name="a_STAT")
    write_freeflux(back, str(tmp_path / "ff"), experiment_name="a_INST")
    assert (tmp_path / "mtf" / "EC.netw").exists()
    assert (tmp_path / "ff" / "reactions.tsv").exists()


# -- dataset content -----------------------------------------------------------


def test_ec_dataset_content(ec):
    ds = _quiet(
        to_fluxo_dataset,
        ec,
        dataset_id="ec",
        source_path=EC_FML,
        source_root=REPO,
        data_origin="synthetic",
        metadata=EC_SIDECAR,
    )
    net = ds.networks[0]
    assert net.description == "E. Coli Model Shipped with INCA 2.3"
    assert net.model_date == "2025-08-05 09:28:21"
    assert not ds.conditions
    source = ds.source_files[0]
    assert source.path == "data/benchmark/FluxML/EC.fml"
    assert source.checksum.startswith("sha256:")
    assert source.file_size_bytes == EC_FML.stat().st_size
    assert {"network", "constraints", "tracer", "mid_measurements"} <= set(
        source.role
    )

    r25 = next(r for r in net.reactions if r.id == "R25")
    assert [v.variant_id for v in r25.atom_transition_variants] == [
        f"R25___{i}" for i in range(1, 5)
    ]
    # Uniform weights are the parser's default, not data.
    assert all(v.ratio is None for v in r25.atom_transition_variants)
    r23 = next(r for r in net.reactions if r.id == "R23")
    first = r23.atom_transition_variants[0].transitions[0]
    assert (first.reactant_metabolite_id, first.reactant_atom_index) == (
        "AKG",
        2,
    )
    assert (first.product_metabolite_id, first.product_atom_index) == (
        "SucCoA",
        1,
    )
    assert first.element == "C"
    assert {c.flux_type for c in net.flux_constraints} == {"net"}
    assert net.flux_constraints[0].constraint_type == "equality"

    exps = {e.configuration_name: e for e in ds.experiments}
    assert set(exps) == {"a_INST", "b_INST", "a_STAT", "b_STAT"}
    inst = exps["a_INST"]
    assert inst.stationary is False and inst.time_unit == "s"
    assert inst.investigation_id == "ec_fluxml_benchmark.investigation"
    mid = next(m for m in inst.mid_measurements if m.id.endswith("ms_group_1"))
    assert mid.id == "ec.a_INST.ms_group_1"
    assert mid.measurement_spec == "Ala[2,3]#M0,1,2"
    assert (mid.metabolite_id, mid.atom_positions) == ("Ala", "2,3")
    assert len(mid.time_points) == 9
    fp1 = next(f for f in inst.flux_measurements if f.id.endswith("fp_1"))
    assert fp1.flux_expression == "R63-R63b" and fp1.reaction_id is None
    runs = {r.configuration_name: r for r in ds.simulation_runs}
    assert runs["a_STAT"].fluxomics_method == "IST_MFA"
    assert runs["a_INST"].fluxomics_method == "INST_MFA"
    assert runs["a_INST"].software.software_name == "13CFLUX2"
    kinds = {v.variable_kind for v in runs["a_INST"].variables}
    assert kinds == {"net_flux", "exchange_flux", "pool_size"}
    assert {v.variable_type for v in runs["a_INST"].variables} == {"free"}


def test_dump_omits_schema_defaults(ec, tmp_path):
    ds = _quiet(to_fluxo_dataset, ec, dataset_id="ec", source_path=EC_FML)
    text = save_record(ds, tmp_path / "ec.yaml", header="EC").read_text()
    assert text.startswith("# EC\n")
    assert "is_pseudo" not in text and "instance: 1\n" not in text
    doc = yaml.safe_load(text)
    assert doc["networks"][0]["model_date"] == "2025-08-05 09:28:21"


def test_ec_dataset_validates(ec, tmp_path):
    ds = _quiet(
        to_fluxo_dataset,
        ec,
        dataset_id="ec_fluxml_benchmark",
        source_path=EC_FML,
        source_root=REPO,
        dataset_type="benchmark",
        data_origin="synthetic",
        metadata=EC_SIDECAR,
    )
    path = save_record(ds, tmp_path / "ec_fluxml_benchmark.yaml")
    cli = shutil.which("linkml-validate") or str(
        Path(sys.executable).parent / "linkml-validate"
    )
    if not Path(cli).exists() or not SCHEMA.exists():
        pytest.skip("linkml-validate or the FluxoLink schema not found")
    result = subprocess.run(
        [cli, "-s", str(SCHEMA), str(path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    validate = pytest.importorskip("fluxolink.validate")
    issues = validate.validate_dataset(yaml.safe_load(path.read_text()))
    errors = [str(i) for i in issues if i.severity == "error"]
    assert not errors, errors


def test_sidecar_rules(ec):
    with pytest.raises(ValueError, match="Unknown metadata sidecar"):
        to_fluxo_dataset(ec, dataset_id="x", metadata={"organism": "E. coli"})
    ds = _quiet(
        to_fluxo_dataset,
        ec.model_copy(
            update={
                "metadata": ec.metadata.model_copy(update={"strain": "K-12"})
            }
        ),
        dataset_id="x",
        metadata={
            "investigation": {"name": "inv"},
            "contributors": [{"id": "c1", "name": "A. Person"}],
            "publication": {"doi": "doi:10.1/x"},
            "conditions": [{"id": "cond", "organism_name": "E. coli"}],
        },
    )
    inv = ds.investigations[0]
    assert inv.id == "x.investigation1"
    assert inv.publication_doi == "doi:10.1/x"
    assert inv.contributors[0].name == "A. Person"
    assert ds.conditions[0].strain == "K-12"
    assert {e.condition for e in ds.experiments} == {"cond"}
    # No sidecar condition: the strain is not published anywhere.
    bare = _quiet(to_fluxo_dataset, ec, dataset_id="y")
    assert not bare.conditions


def test_experiment_without_tracer_is_kept_without_inventing_one():
    """FreeFlux EC keeps its tracer in the driver script: no tracer, no error.

    The experiment is written without labeled_sources, and the benchmark gate
    (not the converter) reports what a benchmark would still need.
    """
    validate = pytest.importorskip("fluxolink.validate")
    data = _quiet(parse_freeflux, str(DATA / "benchmark" / "FreeFlux" / "EC"))
    ds = _quiet(to_fluxo_dataset, data, dataset_id="ff", dataset_type="catalog")
    assert ds.experiments
    assert all(not e.labeled_sources for e in ds.experiments)
    doc = ds.model_dump(mode="json", exclude_unset=True, exclude_none=True)
    assert validate.schema_issues(doc) == []
    errors = [
        i for i in validate.validate_dataset(doc) if i.severity == "error"
    ]
    assert errors == []
    gate = validate.validate_dataset(doc, benchmark=True)
    assert any("labeled_sources" in i.message for i in gate)


def test_non_uniform_weights_become_ratios(ec, tmp_path):
    transition = ec.model.atom_mappings["R24"]
    skewed = transition.model_copy(
        update={"weights": {"R24___1": 0.7, "R24___2": 0.3}}
    )
    data = _replace_mapping(ec, "R24", skewed)
    ds = _quiet(to_fluxo_dataset, data, dataset_id="w")
    r24 = next(r for r in ds.networks[0].reactions if r.id == "R24")
    assert [v.ratio for v in r24.atom_transition_variants] == [0.7, 0.3]
    back = from_fluxo_dataset(ds)
    assert back.model.atom_mappings["R24"].weights == {
        "R24___1": 0.7,
        "R24___2": 0.3,
    }
    assert compare(data, back).equivalent
    assert compare(ec, back).of_kind("atom_transition")


def test_nmr_measurements_round_trip():
    exp = LabelingExperiments(
        name="nmr",
        tracers=[
            Tracers(
                metabolite="A",
                labels=[LabelComposition(labeled_pattern="11", fraction=1.0)],
            )
        ],
        measurement=Measurement(
            model=MeasurementModel(
                labeling_measurement=LabelingMeasurement(
                    groups=[
                        Group(
                            id="cnmr",
                            expression=TextualOrMath(textual="A#S1,DL2"),
                        )
                    ]
                )
            ),
            data=MeasurementData(
                data=[
                    Datum(id="cnmr", value=0.4, stddev=0.01, pos=1, type="S"),
                    Datum(id="cnmr", value=0.6, stddev=0.01, pos=2, type="DL"),
                ]
            ),
        ),
    )
    original = FluxomicsData(
        model=MetabolicNetworkModel(
            metabolites=[Metabolite(id="A", atoms=2), Metabolite(id="B")],
            reactions=[Reaction(id="v1", reactants=["A"], products=["B"])],
        ),
        experiments=[exp],
    )
    ds = _quiet(to_fluxo_dataset, original, dataset_id="nmr")
    nmr = ds.experiments[0].nmr_measurements[0]
    assert nmr.metabolite_id == "A"
    assert [(p.atom_position, p.multiplet_type) for p in nmr.data_points] == [
        (1, "S"),
        (2, "DL"),
    ]
    assert compare(original, from_fluxo_dataset(ds)).equivalent


# -- equivalence ---------------------------------------------------------------


def _replace_mapping(data, reaction_id, transition):
    mappings = dict(data.model.atom_mappings)
    mappings[reaction_id] = transition
    model = data.model.model_copy(update={"atom_mappings": mappings})
    return data.model_copy(update={"model": model})


def _replace_experiment(data, name, **changes):
    exps = [
        e.model_copy(update=changes) if e.name == name else e
        for e in data.experiments
    ]
    return data.model_copy(update={"experiments": exps})


def test_compare_flags_one_atom_transition(ec):
    transition = ec.model.atom_mappings["R1"]
    ((key, atom_map),) = transition.maps.items()
    mapping = dict(atom_map.mapping)
    one = AtomAddress("F6P", 1, "C", 1)
    two = AtomAddress("F6P", 2, "C", 1)
    mapping[one], mapping[two] = mapping[two], mapping[one]
    perturbed = transition.model_copy(
        update={"maps": {key: AtomMap(mapping=mapping)}}
    )
    report = compare(ec, _replace_mapping(ec, "R1", perturbed))
    assert [(d.kind, d.key) for d in report.differences] == [
        ("atom_transition", "R1")
    ]


def test_compare_flags_one_tracer_fraction(ec):
    exp = ec.get_experiments("a_STAT")
    tracers = list(exp.tracers)
    glucose = tracers[0]
    labels = [glucose.labels[0].model_copy(update={"fraction": 0.74})]
    tracers[0] = glucose.model_copy(
        update={"labels": labels + list(glucose.labels[1:])}
    )
    report = compare(ec, _replace_experiment(ec, "a_STAT", tracers=tracers))
    assert [(d.kind, d.key) for d in report.differences] == [
        ("tracer", "a_STAT/Gluc_ext")
    ]


@pytest.mark.parametrize("delta,flagged", [(1e-3, True), (1e-12, False)])
def test_compare_flags_one_mid_value(ec, delta, flagged):
    exp = ec.get_experiments("a_STAT")
    data = list(exp.measurement.data.data)
    index = next(i for i, d in enumerate(data) if d.weight is not None)
    data[index] = data[index].model_copy(
        update={"value": data[index].value + delta}
    )
    measurement = exp.measurement.model_copy(
        update={"data": MeasurementData(data=data)}
    )
    report = compare(
        ec, _replace_experiment(ec, "a_STAT", measurement=measurement)
    )
    if flagged:
        assert [d.kind for d in report.differences] == ["mid_measurement"]
        assert report.differences[0].key.startswith("a_STAT/AKG")
    else:
        assert report.equivalent


def test_compare_ignores_variant_names_and_partition(ec):
    transition = ec.model.atom_mappings["R25"]
    renamed = transition.model_copy(
        update={
            "maps": {
                f"x{i}": m for i, m in enumerate(transition.maps.values())
            },
            "weights": None,
        }
    )
    data = _replace_mapping(ec, "R25", renamed)
    exp = data.get_experiments("a_STAT")
    variables = exp.simulation.variables.model_copy(
        update={"flux_values": exp.simulation.variables.flux_values[:3]}
    )
    data = _replace_experiment(
        data,
        "a_STAT",
        simulation=exp.simulation.model_copy(update={"variables": variables}),
    )
    assert compare(ec, data).equivalent


def test_compare_id_map_and_ignore(ec):
    renamed = ec.model_copy(
        update={
            "experiments": [
                e.model_copy(update={"name": e.name.lower()})
                for e in ec.experiments
            ]
        }
    )
    assert compare(ec, renamed).counts() == {"experiment": 8}
    id_map = {e.name.lower(): e.name for e in ec.experiments}
    assert compare(ec, renamed, id_map=id_map).equivalent
    assert compare(ec, renamed, ignore=["experiment"]).equivalent
    with pytest.raises(ValueError, match="Unknown difference kind"):
        compare(ec, ec, ignore=["colour"])


def test_canonical_constraints():
    def same(x, y):
        return canonical_constraint(x, False, str) == canonical_constraint(
            y, False, str
        )

    assert same("R1 >= 0.0", "0 <= R1")
    assert same("0.488*R67=bmAla", "bmAla == 0.488 * R67")
    assert same("V4_1 - V4_2 = 0", "V4_2 = V4_1")
    assert not same("R1 >= 0", "R1 <= 0")
    assert same("Gluc.ext >= 1", "1 <= Gluc.ext")


# -- cross-format -------------------------------------------------------------


def test_cross_format_ec_differences(ec, tmp_path, capsys):
    """Record how the EC benchmark differs between the four formats.

    Not an equivalence assertion: the files are independent encodings and
    genuinely differ. The counts are printed for the record.
    """
    influx = DATA / "benchmark" / "influx_si"
    for source, target in [
        ("EC.netw", "EC.netw"),
        ("EC.linp", "EC.linp"),
        ("EC.cnstr", "EC.cnstr"),
        ("EC_stat.tvar", "EC.tvar"),
        ("EC_a_stat.miso", "EC.miso"),
    ]:
        shutil.copy(influx / source, tmp_path / target)
    mtf = _quiet(parse_mtf, str(tmp_path / "EC"))
    inca = _quiet(parse_inca, str(DATA / "benchmark" / "INCA" / "EC.m"))
    freeflux = _quiet(
        parse_freeflux, str(DATA / "benchmark" / "FreeFlux" / "EC")
    )
    dotted = {
        m: m.replace(".", "_")
        for d in (inca, freeflux)
        for m in d.metabolite_ids
        if "." in m
    }
    results = {}
    for label, a, b in [
        ("influx_si a_stat", _only(ec, "a_STAT"), mtf),
        ("INCA", _only(ec, "a_INST"), inca),
        ("FreeFlux", _only(ec, "a_INST"), freeflux),
    ]:
        report = compare(a, b, id_map=dotted)
        assert not report.equivalent
        results[label] = report
    with capsys.disabled():
        print("\nEC.fml vs other formats (difference counts):")
        for label, report in results.items():
            print(f"  {label}: {len(report.differences)} {report.counts()}")
