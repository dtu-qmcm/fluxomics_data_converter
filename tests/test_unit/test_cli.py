"""Tests for the ``fluxo`` command-line interface."""

import re
import shutil
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fluxolink")

from fluxomics_data_converter.cli import (  # noqa: E402
    clean_sidecar,
    main,
    missing_sidecar_fields,
)

DATA = Path(__file__).parents[2] / "data"
EC_FML = DATA / "benchmark" / "FluxML" / "EC.fml"
SIMPLE_MTF = DATA / "simple_model" / "MTF" / "simple_model.netw"
SIMPLE_RES = DATA / "simple_model" / "MTF" / "simple_model_res"
ROUNDTRIP = DATA / "simple_model" / "roundtrip_test"
FF_TOY = DATA / "freeflux_tests" / "toy"


def _filled_sidecar(tmp_path: Path) -> Path:
    template = tmp_path / "meta.yaml"
    assert main(["meta", "init", str(EC_FML), "-o", str(template)]) == 0
    text = template.read_text()
    assert "a_INST, b_INST, a_STAT, b_STAT" in text
    text = text.replace(
        "data_origin: ~            # experimental | synthetic | simulated",
        "data_origin: synthetic",
    ).replace("  name: ~                 # required", "  name: EC benchmark")
    filled = tmp_path / "meta_filled.yaml"
    filled.write_text(text)
    return filled


def test_benchmark_ingest_without_sidecar_lists_missing_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "ec.yaml"
    code = main(["ingest", str(EC_FML), "--tier", "benchmark", "-o", str(out)])
    assert code == 2
    assert not out.exists()
    err = capsys.readouterr().err
    assert "investigation.name" in err
    assert "data_origin" in err


def test_benchmark_ingest_with_sidecar_validates(tmp_path: Path) -> None:
    out = tmp_path / "ec.yaml"
    code = main(
        [
            "ingest",
            str(EC_FML),
            "--tier",
            "benchmark",
            "--meta",
            str(_filled_sidecar(tmp_path)),
            "-o",
            str(out),
        ]
    )
    assert code == 0
    doc = yaml.safe_load(out.read_text())
    assert doc["dataset_type"] == "benchmark"
    assert [i["name"] for i in doc["investigations"]] == ["EC benchmark"]
    assert "conditions" not in doc, "an unfilled condition must not be kept"
    assert {e["data_origin"] for e in doc["experiments"]} == {"synthetic"}
    assert main(["validate", "-q", "--benchmark", str(out)]) == 0


def test_clean_sidecar_drops_blank_entries() -> None:
    raw = {
        "data_origin": None,
        "investigation": {"name": None, "investigation_type": "MFA"},
        "contributors": [{"id": "C1", "name": None}],
        "conditions": [{"id": "COND1", "growth_rate_unit": "1/h"}],
        "experiment_conditions": {},
    }
    assert clean_sidecar(raw) == {}


def test_missing_sidecar_fields_for_experimental_data() -> None:
    doc = {
        "investigations": [{"id": "I", "name": "x"}],
        "conditions": [{"id": "C", "organism_name": "E. coli"}],
        "experiments": [
            {"id": "E", "data_origin": "experimental", "condition": "C"}
        ],
    }
    missing = missing_sidecar_fields(doc)
    assert "conditions[C].strain" in missing
    assert "conditions[C].specific_growth_rate" in missing
    assert "conditions[C].organism_name" not in missing


def test_catalog_ingest_with_influx_results(tmp_path: Path) -> None:
    out = tmp_path / "run.yaml"
    code = main(
        [
            "ingest",
            str(SIMPLE_MTF),
            "--results",
            str(SIMPLE_RES),
            "-o",
            str(out),
        ]
    )
    assert code == 0
    doc = yaml.safe_load(out.read_text())
    runs = doc["simulation_runs"]
    fitted = [r for r in runs if r.get("flux_solution")]
    assert len(fitted) == 1
    assert fitted[0]["software"]["software_name"] == "influx_si"
    assert "forward_simulation" in fitted[0]["run_mode"]
    assert fitted[0]["network_model_id"] == doc["networks"][0]["id"]


def test_compare_results_across_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = []
    for name in ("MTF_from_mtf", "MTF_from_fml"):
        out = tmp_path / f"{name}.yaml"
        res = ROUNDTRIP / name / "simple_model_res"
        code = main(
            [
                "ingest",
                str(ROUNDTRIP / name / "simple_model.netw"),
                "--results",
                str(res),
                "--dataset-id",
                name,
                "-o",
                str(out),
            ]
        )
        assert code == 0
        files.append(str(out))
    capsys.readouterr()
    assert main(["compare", *files]) == 0
    table = capsys.readouterr().out
    assert table.splitlines()[0].startswith("reaction\tflux_type")
    assert "largest disagreement" in table


def test_compare_inputs_and_export(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "ec.yaml"
    meta = _filled_sidecar(tmp_path)
    args = ["ingest", str(EC_FML), "--meta", str(meta), "-o", str(out)]
    assert main(args) == 0
    capsys.readouterr()
    assert main(["compare", "--inputs", str(out), str(out)]) == 0
    assert "equivalent" in capsys.readouterr().out

    mtf = tmp_path / "mtf" / "EC"
    assert main(["export", str(out), "--to", "mtf", "-o", str(mtf)]) == 1
    code = main(
        [
            "export",
            str(out),
            "--to",
            "mtf",
            "--experiment",
            "a_STAT",
            "-o",
            str(mtf),
        ]
    )
    assert code == 0
    assert (tmp_path / "mtf" / "EC.netw").exists()
    doc = yaml.safe_load(out.read_text())
    (check,) = doc["equivalence_checks"]
    paths = {f["id"]: f["path"] for f in doc["source_files"]}
    assert "mtf/EC.netw" in {paths[i] for i in check["files"]}
    fml = tmp_path / "EC_rt.fml"
    assert main(["export", str(out), "--to", "fluxml", "-o", str(fml)]) == 0
    assert fml.read_text().lstrip().startswith("<?xml")


def test_convert_links_inputs_and_results(tmp_path: Path) -> None:
    work = tmp_path / "src"
    shutil.copytree(SIMPLE_MTF.parent, work)
    record = tmp_path / "simple.yaml"
    convert = [
        "convert",
        str(work / "simple_model.netw"),
        "--to",
        "fluxml",
        "freeflux",
        "-o",
        str(tmp_path / "runs"),
        "--link",
        str(record),
    ]
    assert main(convert) == 0
    doc = yaml.safe_load(record.read_text())
    generated = [f for f in doc["source_files"] if f.get("generated_by")]
    assert {Path(f["path"]).parts[1] for f in generated} == {
        "fluxml",
        "freeflux",
    }
    sources = {f["id"] for f in doc["source_files"]} - {
        f["id"] for f in generated
    }
    assert all(set(f["derived_from"]) == sources for f in generated)
    assert len(doc["equivalence_checks"]) == 2

    assert (
        main(["add-results", str(record), str(work / "simple_model_res")]) == 0
    )
    doc = yaml.safe_load(record.read_text())
    (done,) = [r for r in doc["simulation_runs"] if r.get("flux_solution")]
    assert done["software"]["software_name"] == "influx_si"
    assert done["output_files"]

    # converting again reuses the recorded source instead of adding it twice
    assert main(convert) == 0
    again = yaml.safe_load(record.read_text())
    assert len(again["source_files"]) == len(doc["source_files"])


def test_add_input_records_aliases(tmp_path: Path) -> None:
    source = tmp_path / "toy"
    shutil.copytree(FF_TOY, source)
    (source / "measured_inst_MDVs.tsv").unlink()
    record = tmp_path / "toy.yaml"
    assert main(["ingest", str(source), "-o", str(record)]) == 0

    renamed = tmp_path / "renamed"
    shutil.copytree(source, renamed)
    for table in renamed.glob("*.tsv"):
        text = table.read_text()
        table.write_text(re.sub(r"^v3\t", "glu_synthesis\t", text, flags=re.M))
    assert main(["add-input", str(record), str(renamed)]) == 0
    doc = yaml.safe_load(record.read_text())
    (v3,) = [r for r in doc["networks"][0]["reactions"] if r["id"] == "v3"]
    assert v3["aliases"] == ["freeflux:glu_synthesis"]
    checks = {c["id"]: c for c in doc["equivalence_checks"]}
    assert any(c["equivalent"] for c in checks.values())
