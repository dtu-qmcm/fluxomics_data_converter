"""Tests for linking tool inputs, aliases and results in a record."""

from __future__ import annotations

import re
import shutil
import warnings
from pathlib import Path

import pytest

pytest.importorskip("fluxolink")

from fluxolink.validate import ERROR, schema_issues, validate_dataset  # noqa: E402

from fluxomics_data_converter.io import (  # noqa: E402
    parse_fluxml_file,
    parse_freeflux,
    parse_mtf,
)
from fluxomics_data_converter.link import (  # noqa: E402
    ToolInput,
    add_results,
    alias_map,
    check_tool_input,
    from_fluxo_dataset,
    infer_aliases,
    record_document,
    record_tool_input,
    result_files,
    to_fluxo_dataset,
    write_tool_input,
)
from fluxomics_data_converter.link.results import (  # noqa: E402
    read_influx_si_results,
)

DATA = Path(__file__).parents[2] / "data"
FF_TOY = DATA / "freeflux_tests" / "toy"
SIMPLE_FML = DATA / "simple_model" / "FluxML" / "simple_model.fml"
SIMPLE_MTF = DATA / "simple_model" / "MTF"


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _stationary_toy(target: Path) -> Path:
    """Copy the FreeFlux toy without its non-stationary data."""
    shutil.copytree(FF_TOY, target)
    (target / "measured_inst_MDVs.tsv").unlink()
    return target


def _errors(ds) -> list[str]:
    doc = record_document(ds)
    issues = schema_issues(doc) or validate_dataset(doc)
    return [str(i) for i in issues if i.severity == ERROR]


def _record(data, source: Path, root: Path):
    return to_fluxo_dataset(
        data,
        dataset_id="t",
        source_path=source,
        source_root=root,
        dataset_type="catalog",
        data_origin="synthetic",
    )


def test_drains_and_unlabelled_inputs_are_by_design(tmp_path: Path) -> None:
    data = parse_freeflux(str(FF_TOY))
    ti = write_tool_input(data, "fluxml", tmp_path / "toy.fml")
    check = check_tool_input(data, ti)
    assert check.equivalent, check.differences
    assert "[reaction] Glu_out: drain added for terminal pool Glu" in (
        check.expected_differences
    )
    assert any(
        "unlabelled input declared for source pool AcCoA" in line
        for line in check.expected_differences
    )


def test_real_losses_are_reported(tmp_path: Path) -> None:
    data = parse_fluxml_file(str(SIMPLE_FML))
    ti = write_tool_input(data, "mtf", tmp_path / "simple")
    check = check_tool_input(data, ti)
    # sink pools for the effluxes are by design ...
    assert "[metabolite] E_ext: sink pool added by the mtf writer" in (
        check.expected_differences
    )
    assert any(
        line.startswith("[atom_transition] E_out")
        for line in check.expected_differences
    )
    # ... but the MTF writer drops the tracer purity, which changes the
    # problem (natural abundance), so the check must say so
    assert not check.equivalent
    assert any(line.startswith("[tracer]") for line in check.differences)


def test_record_tool_input_links_files_check_and_run(tmp_path: Path) -> None:
    source = _stationary_toy(tmp_path / "src")
    data = parse_freeflux(str(source))
    ds = _record(data, source, tmp_path)
    source_ids = [f.id for f in ds.source_files]

    ti = write_tool_input(data, "fluxml", tmp_path / "runs" / "toy.fml")
    check = check_tool_input(from_fluxo_dataset(ds), ti)
    ds = record_tool_input(
        ds, ti, check, root=tmp_path, generated=True, derived_from=source_ids
    )
    assert _errors(ds) == []

    generated = [f for f in ds.source_files if f.path == "runs/toy.fml"]
    assert len(generated) == 1
    entry = generated[0]
    assert entry.generated_by == "fluxomics_data_converter"
    assert entry.derived_from == source_ids
    assert entry.checksum.startswith("sha256:")

    (chk,) = ds.equivalence_checks
    assert chk.files == [entry.id]
    assert chk.equivalent == check.equivalent

    runs = [r for r in ds.simulation_runs if r.input_files == [entry.id]]
    assert len(runs) == 1
    assert runs[0].run_status == "planned"
    assert runs[0].software.software_name == "13CFLUX2"


def test_mtf_set_is_one_check_over_all_files(tmp_path: Path) -> None:
    data = parse_fluxml_file(str(SIMPLE_FML))
    ds = _record(data, SIMPLE_FML, tmp_path)
    ti = write_tool_input(data, "mtf", tmp_path / "mtf" / "simple")
    ds = record_tool_input(
        ds, ti, check_tool_input(data, ti), root=tmp_path, generated=True
    )
    (chk,) = ds.equivalence_checks
    paths = {f.id: f.path for f in ds.source_files}
    assert {paths[i] for i in chk.files} >= {
        "mtf/simple.netw",
        "mtf/simple.linp",
        "mtf/simple.miso",
    }
    assert chk.experiment_id
    assert _errors(ds) == []


def test_rewritten_file_of_a_finished_run_gets_a_new_id(tmp_path: Path) -> None:
    data = parse_fluxml_file(str(SIMPLE_FML))
    ds = _record(data, SIMPLE_FML, tmp_path)
    out = tmp_path / "simple.fml"
    ti = write_tool_input(data, "fluxml", out)
    ds = record_tool_input(
        ds, ti, check_tool_input(data, ti), root=tmp_path, generated=True
    )
    (run,) = [r for r in ds.simulation_runs if r.id.endswith(".fluxml.run")]
    first_id = run.input_files[0]
    run.run_status = "completed"

    out.write_text(out.read_text() + "<!-- edited -->\n")
    ti = ToolInput.locate(out)
    ds = record_tool_input(
        ds, ti, check_tool_input(data, ti), root=tmp_path, generated=False
    )
    ids = [f.id for f in ds.source_files if f.path == "simple.fml"]
    assert first_id in ids and len(ids) == 2
    finished = [r for r in ds.simulation_runs if r.run_status == "completed"]
    assert finished[0].input_files == [first_id]


def test_hand_written_input_ids_become_aliases(tmp_path: Path) -> None:
    source = _stationary_toy(tmp_path / "src")
    data = parse_freeflux(str(source))
    ds = _record(data, source, tmp_path)
    reference = from_fluxo_dataset(ds)

    renamed = _stationary_toy(tmp_path / "renamed")
    for table in renamed.glob("*.tsv"):
        text = re.sub(
            r"^v3\t", "glu_synthesis\t", table.read_text(), flags=re.M
        )
        table.write_text(re.sub(r"\bGlu(?=[(\t_])", "GLU", text))

    ti = ToolInput.locate(renamed)
    match = infer_aliases(reference, ti.read())
    assert match.reactions == {"glu_synthesis": "v3"}
    assert match.metabolites == {"GLU": "Glu"}
    check = check_tool_input(reference, ti, id_map=match.id_map)
    assert not any("v3" in line for line in check.differences)

    ds = record_tool_input(
        ds, ti, check, root=tmp_path, generated=False, aliases=match
    )
    aliases = alias_map(ds, "freeflux")
    assert aliases == {"glu_synthesis": "v3", "GLU": "Glu"}
    assert _errors(ds) == []


def test_add_results_fills_the_planned_run(tmp_path: Path) -> None:
    work = tmp_path / "MTF"
    shutil.copytree(SIMPLE_MTF, work)
    data = parse_mtf(str(work / "simple_model"))
    ds = _record(data, work / "simple_model", tmp_path)
    planned = [r for r in ds.simulation_runs if r.run_status == "planned"]
    assert len(planned) == 1

    res = work / "simple_model_res"
    run = read_influx_si_results(res)
    ds = add_results(
        ds, run, output_files=result_files(res, "influx_si"), root=tmp_path
    )
    (filled,) = [r for r in ds.simulation_runs if r.id == planned[0].id]
    assert filled.run_status != "planned"
    assert filled.input_files == planned[0].input_files
    assert filled.flux_solution is not None
    assert filled.flux_solution.id.startswith(filled.id)
    outputs = {f.id: f for f in ds.source_files}
    assert filled.output_files
    assert all("results" in outputs[i].role for i in filled.output_files)
    assert _errors(ds) == []
