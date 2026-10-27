"""Tests for the tool results readers (influx_si, INCA, FreeFlux, x3cflux).

Every produced run is wrapped in a catalog-tier dataset and checked with
``fluxolink-validate`` (schema + semantic checks), which must report no
error.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fluxolink")
scipy_io = pytest.importorskip("scipy.io")
from scipy.stats import chi2  # noqa: E402

from fluxomics_data_converter.link.results._common import (  # noqa: E402
    checked_interval,
)
from fluxomics_data_converter.link import save_record  # noqa: E402
from fluxomics_data_converter.link.results import (  # noqa: E402
    from_freeflux_result,
    from_x3cflux_fit,
    parse_influx_log,
    read_inca_results,
    read_influx_si_results,
    read_miso_sim,
    wrap_in_dataset,
)

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
LINKML = REPO.parent / "fluxomics_linkml"
SIMPLE_RES = DATA / "simple_model" / "MTF" / "simple_model_res"
ROUNDTRIP = DATA / "simple_model" / "roundtrip_test"
INFLUX_DIRS = [
    SIMPLE_RES,
    ROUNDTRIP / "MTF_from_fml" / "simple_model_res",
    ROUNDTRIP / "MTF_from_freeflux" / "simple_model_res",
    ROUNDTRIP / "MTF_from_mtf" / "simple_model_res",
]
TOY = DATA / "MTF_tests" / "toy_test"
INCA_DOWNLOADS = Path(
    "/private/tmp/claude-1573474093/"
    "-Users-techen-Yo-Fluxomics-data-model-fluxomics-linkml/"
    "662fdf48-be17-4c0f-8b76-d736f815ff83/scratchpad/dl_inca_mat"
)


# -- helpers ------------------------------------------------------------------


def _validator_command() -> list[str] | None:
    local = Path(sys.executable).parent / "fluxolink-validate"
    if local.exists():
        return [str(local)]
    found = shutil.which("fluxolink-validate")
    if found:
        return [found]
    uv = shutil.which("uv")
    if uv and LINKML.exists():
        return [uv, "run", "--project", str(LINKML), "fluxolink-validate"]
    return None


def validate(runs, tmp_path: Path) -> None:
    """Wrap each run in a catalog dataset and require 0 validation errors."""
    command = _validator_command()
    if command is None:
        pytest.skip("fluxolink-validate is not available")
    files = []
    for i, run in enumerate(runs):
        ds = wrap_in_dataset(run, dataset_id=f"ds{i}")
        files.append(save_record(ds, tmp_path / f"ds{i}.yaml"))
    proc = subprocess.run(
        command + [str(f) for f in files],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summaries = [ln for ln in proc.stdout.splitlines() if "error(s)" in ln]
    assert len(summaries) == len(files), proc.stdout
    assert all(": 0 error(s)" in ln for ln in summaries), proc.stdout


def flux(run, reaction, flux_type="net"):
    matches = [
        f
        for f in run.flux_solution.flux_results
        if f.reaction_id == reaction and f.flux_type == flux_type
    ]
    assert len(matches) == 1, (reaction, flux_type, matches)
    return matches[0]


def options(run):
    return {o.key: o for o in run.solver_options or []}


# -- influx_si: real result directories ---------------------------------------


@pytest.mark.parametrize("res_dir", INFLUX_DIRS, ids=lambda p: p.parent.name)
def test_influx_si_real_results(res_dir):
    run = read_influx_si_results(res_dir)

    assert run.run_status == "completed"
    assert run.run_mode == ["forward_simulation"]
    assert run.fluxomics_method == "IST_MFA"
    assert run.flux_solution.solution_role == "reference"
    assert run.software.software_name == "influx_si"
    assert run.software.version == "7.4.3"
    assert run.command_line.startswith('"')
    assert run.command_line.endswith(
        '"--noopt" "--clownr" "1e-5" "--prefix" "simple_model"'
    )
    assert run.isotopic_state_space == "cumomer"
    assert run.optimization_method is None  # --noopt: nothing optimized
    opts = options(run)
    assert opts["--noopt"].value is None
    assert opts["--clownr"].value == "1e-5"  # the log, not the .opt (1.e-3)
    assert opts["--zc"].value == "0.0001"
    assert ".opt" in opts["--zc"].description
    assert opts["--prefix"].value == "simple_model"
    assert run.start_time.startswith("2026-05-12T23:1")
    assert run.runtime_seconds == 2.0

    sol = run.flux_solution
    assert len(sol.flux_results) == 16
    r2 = flux(run, "R2")
    assert r2.value == 110.0
    assert r2.structural_identifiability == "yes"
    xch = flux(run, "R2", "exchange")
    assert xch.value == 0.0 and xch.variable_type == "constrained"
    assert xch.derivation.startswith("xch = xch01 / (1 - xch01)")
    assert "influx_si xch01 = 0" in xch.derivation  # the row as written
    assert r2.derivation is None

    gof = run.goodness_of_fit
    # .stat says 1.17771751823129e-08, the lower tail; stored as upper tail
    assert gof.chi_square_p_value == pytest.approx(1 - 1.17771751823129e-08)
    assert gof.fit_accepted is True
    assert gof.chi_square_confidence_level == 0.95
    assert gof.degrees_of_freedom == 1
    assert gof.number_of_measurements == 5
    assert gof.number_of_parameters == 4
    assert gof.wssr < 1e-15

    mids = sol.simulated_mids
    assert len(mids) == 1
    assert mids[0].scaled is True  # stationary .miso.sim values are scaled
    assert mids[0].fragment_name == "MS_F_"
    assert mids[0].metabolite_id == "F"
    assert mids[0].atom_positions == "1,2,3"  # joined to the input .miso
    m1 = next(
        f
        for f in sol.measurement_fits
        if f.measurement_id == "MS_F_" and f.isotopologue_index == 1
    )
    assert m1.measured_value == 0.800761905
    assert m1.residual == pytest.approx(m1.simulated_value - 0.800761905)
    assert m1.weighted_residual == pytest.approx(
        m1.residual / m1.standard_deviation, rel=1e-6
    )
    assert m1.derivation is None  # measured value read from the input
    r1 = next(f for f in sol.measurement_fits if f.measurement_id == "R1")
    assert r1.isotopologue_index is None
    assert r1.measured_value == 100.0 and r1.residual == 0.0
    assert len(sol.scale_factors) == 1
    assert sol.scale_factors[0].measurement_id.startswith("1:mass;F;111;")


def test_influx_si_values_differ_per_directory():
    """The four runs are not copies: check file-specific values."""
    mtf = read_influx_si_results(SIMPLE_RES)
    fml = read_influx_si_results(INFLUX_DIRS[1])
    assert flux(mtf, "R2").variable_type == "free"
    assert flux(mtf, "R2").standard_deviation == 12.6867790062193
    assert flux(fml, "R2").variable_type == "dependent"
    assert flux(fml, "R2").standard_deviation == 12.6867790062349
    assert mtf.goodness_of_fit.wssr == 2.17872364785429e-16
    assert mtf.goodness_of_fit.chi_square_p_value == pytest.approx(
        1 - 1.17771751823129e-08
    )
    assert fml.goodness_of_fit.wssr == 2.17859091132409e-16
    assert fml.flux_solution.scale_factors[0].measurement_id == (
        "1:mass;F;111;108"
    )
    assert fml.flux_solution.scale_factors[0].value == 1.0000000001909
    assert mtf.start_time == "2026-05-12T23:19:01"
    assert mtf.end_time == "2026-05-12T23:19:03"


def test_influx_si_real_results_validate(tmp_path):
    validate([read_influx_si_results(d) for d in INFLUX_DIRS], tmp_path)


def test_influx_si_measured_value_without_input(tmp_path):
    res = tmp_path / "simple_model_res"
    shutil.copytree(SIMPLE_RES, res)
    run = read_influx_si_results(res)  # tmp_path holds no input .miso
    m0 = next(
        f
        for f in run.flux_solution.measurement_fits
        if f.isotopologue_index == 0
    )
    # recovered as simulated - Residual * SD
    assert m0.measured_value == pytest.approx(6.34920635e-05, rel=1e-12)
    assert m0.derivation.startswith(
        "measured_value = simulated_value - weighted_residual * SD"
    )
    r1 = next(
        f
        for f in run.flux_solution.measurement_fits
        if f.measurement_id == "R1"
    )
    assert r1.measured_value == 100.0 and r1.derivation is not None
    assert run.flux_solution.simulated_mids[0].atom_positions is None


# -- influx_si: toy files (.tvar with free XCH, toy_calc.miso) ----------------


@pytest.fixture
def toy_res(tmp_path):
    res = tmp_path / "toy_res"
    res.mkdir()
    shutil.copy(TOY / "toy.tvar", res / "toy.tvar.sim")  # .tvar.sim layout
    shutil.copy(TOY / "toy_calc.miso", res / "toy.miso.sim")
    shutil.copy(TOY / "toy.mflux", res / "toy.mflux.sim")  # empty Residual
    return res


def test_read_miso_sim_toy_calc():
    mids, fits = read_miso_sim(
        TOY / "toy_calc.miso", input_miso=TOY / "toy.miso"
    )
    assert len(fits) == 53
    lab = next(f for f in fits if f.measurement_id == "LAB_E_1xx")
    assert lab.isotopologue_index is None
    assert lab.simulated_value == 0.724538895599788
    # NA measured value, NA Residual and Pvalue
    assert lab.measured_value is None
    assert lab.residual is None and lab.weighted_residual is None
    assert lab.p_value is None
    by_name = {m.fragment_name: m for m in mids}
    assert set(by_name) >= {"MS_B_", "MS_C_1", "MS_C_", "MS_E+B_", "MS_I_"}
    assert by_name["MS_C_1"].atom_positions == "1"
    assert by_name["MS_C_"].atom_positions == "1,2,3"
    assert by_name["MS_H_"].atom_positions == "1,2,3,4,5"
    pooled = by_name["MS_E+B_"]
    assert pooled.metabolite_id is None and pooled.measurement_spec == "E+B"
    b2 = next(
        p for p in by_name["MS_B_"].data_points if p.isotopologue_index == 2
    )
    assert b2.value == -5.55111512312578e-17  # tiny negative kept
    assert len(by_name["MS_I_"].data_points) == 7


def test_influx_si_toy_xch01_and_pools(toy_res, tmp_path):
    run = read_influx_si_results(toy_res, input_dir=TOY)
    assert options(run)["--prefix"].value == "toy"
    assert run.run_mode == ["estimation"]  # no log: not a --noopt run
    v3 = flux(run, "V3", "exchange")
    x01, sd01 = 0.860915383556858, 0.0826712020315524
    assert v3.value == pytest.approx(x01 / (1 - x01))
    assert v3.standard_deviation == pytest.approx(sd01 / (1 - x01) ** 2)
    assert v3.variable_type == "free"
    assert v3.derivation == (
        "xch = xch01 / (1 - xch01); SD by the delta method; MC bounds "
        f"transformed exactly; influx_si xch01 = {x01} (SD {sd01!r})"
    )
    assert not any(k.startswith("xch01:") for k in options(run))
    assert flux(run, "V1").variable_type == "constrained"
    pools = run.flux_solution.pool_size_results
    assert len(pools) == 13
    assert {p.variable_type for p in pools} == {"constrained"}
    biomass = next(
        f
        for f in run.flux_solution.measurement_fits
        if f.measurement_id == "BIOMASS_YIELD"
    )
    assert biomass.measured_value == 0.113703411305323  # from toy.mflux
    assert biomass.residual == 0.0
    assert run.goodness_of_fit is None  # no .stat
    validate([run], tmp_path)


# -- influx_si: synthetic Monte Carlo / multistart / --wkvh run ---------------

_TVAR_MC = """\
Id\tComment\tName\tKind\tType\tValue\tSD\tStruct_identif\tLow_mc\tUp_mc
\t\tR1\tNET\tF\t100\t0.1\tyes\t99.8\t100.2
\t\tR2\tNET\tD\t40\t2\tyes\t41\t45
\t\tR2\tXCH\tF\t0.5\t0.01\tyes\t0.4\t0.6
\t\tR3\tNET\tF\tNA\t20000\tno\t1\t2
\t\tA\tMETAB\tF\t1.5\t0.2\tyes\t1.2\t1.9
"""
_LOG_MC = (
    '"/opt/influx_s.py" "--noopt" "--prefix" "old"\n'
    "code gen: 2026-01-01 10:00:00\n"
    "end     : 2026-01-01 10:00:01\n"
    '"/opt/bin/influx_s.py" "--sens" "mc=20" "--np" "4" "--emu" '
    '"--wkvh" "--clownr=1e-4" "--prefix" "mc"\n'
    "code gen: 2026-01-02 09:00:00\n"
    'executing: --vanilla --slave < mc.R"\n'
    "calcul  : 2026-01-02 09:00:01\n"
    "***Warning: inverse of covariance matrix is numerically singular.\n"
    "Starting point V1\n"
    "end     : 2026-01-02 09:01:30.5\n"
)
_STAT_BAD = (
    "chi2_value\t30\nchi2/df\t3\nnumber_of_measurements\t14\n"
    "number_of_parameters\t4\ndegrees_of_freedom\t10\n"
    "p-value\t0.999\nconclusion\tAt level of 95% confidence, the model does "
    "not fit the data good enough with respect to the provided measurement "
    "SD\n"
)
_KVH = (
    "influx\n\tversion\t7.9.9\n"
    "optimization process information\n\tfitted parameters\n"
    "\t\tf.n.R1\t100\n\titeration number\t17\n\texit message\t\n"
)


def test_influx_si_synthetic_mc_multistart(tmp_path):
    res = tmp_path / "mc_res"
    (res / "tmp").mkdir(parents=True)
    for start in ("V1", "V2"):
        (res / f"mc.{start}.tvar.sim").write_text(_TVAR_MC)
        (res / f"mc.{start}.stat").write_text(_STAT_BAD)
    (res / "mc.log").write_text(_LOG_MC)
    (res / "mc.err").write_text("")
    (res / "tmp" / "mc.pres.csv").write_text(
        "row_col\tV1\tV2\ncost\t35\t30\nf.n.R1\t100\t100\n"
        "1:mass;F;10;13\t1.1\t0.9\n"
    )
    (res / "tmp" / "mc_res.V2.kvh").write_text(_KVH)

    with pytest.warns(UserWarning) as record:
        run = read_influx_si_results(res)  # best start V2 (lowest cost)
    messages = [str(w.message) for w in record]
    assert len(messages) == 1 and "NET R3 has no value" in messages[0]

    assert run.command_line.startswith('"/opt/bin/influx_s.py"')  # last run
    assert run.run_mode == [
        "estimation",
        "confidence_interval_estimation",
        "monte_carlo",
    ]
    assert run.monte_carlo_runs == 20
    assert run.multistart_count == 2
    assert run.isotopic_state_space == "EMU"
    assert run.optimization_method == "nlsic"
    assert run.flux_solution.solution_role == "estimated"
    opts = options(run)
    assert opts["--sens"].value == "mc=20"
    assert opts["--np"].value == "4"
    assert opts["--emu"].value is None
    assert opts["--clownr"].value == "1e-4"
    assert run.start_time == "2026-01-02T09:00:00"
    assert run.end_time == "2026-01-02T09:01:30.5"
    assert run.runtime_seconds == pytest.approx(90.5)
    assert run.iteration_count == 17
    assert run.converged is True
    assert run.software.version == "7.9.9"  # from the .kvh
    assert "numerically singular" in run.solver_status

    r1 = flux(run, "R1")
    assert (r1.lower_ci, r1.upper_ci) == (99.8, 100.2)
    assert r1.confidence_level == 0.95
    assert r1.confidence_interval_method == "monte_carlo"
    r2 = flux(run, "R2")  # an MC interval may exclude the estimate: kept
    assert (r2.value, r2.lower_ci, r2.upper_ci) == (40.0, 41.0, 45.0)
    xch = flux(run, "R2", "exchange")
    assert xch.value == pytest.approx(1.0)
    assert xch.lower_ci == pytest.approx(0.4 / 0.6)
    assert xch.upper_ci == pytest.approx(1.5)
    assert xch.standard_deviation == pytest.approx(0.04)
    assert "influx_si xch01 = 0.5 (SD 0.01)" in xch.derivation
    pool = run.flux_solution.pool_size_results[0]
    assert (pool.metabolite_id, pool.lower_ci, pool.upper_ci) == ("A", 1.2, 1.9)
    scale = run.flux_solution.scale_factors[0]
    assert (scale.measurement_id, scale.value) == ("1:mass;F;10;13", 0.9)

    gof = run.goodness_of_fit
    assert gof.fit_accepted is False
    # 0.999 in the .stat is the lower tail; the record holds the upper
    assert gof.chi_square_p_value == pytest.approx(1 - 0.999)
    assert gof.reduced_chi_square == 3.0

    with pytest.warns(UserWarning, match="R3 has no value"):
        v1 = read_influx_si_results(res, start="V1")
    assert v1.flux_solution.scale_factors[0].value == 1.1
    validate([run, v1], tmp_path)


_MISO_SIM_INST = """\
Id\tComment\tSpecie\tFragment\tDataset\tIsospecies\tValue\tSD\tTime\tResidual\tPvalue
\t\tF\t\tMS_F_\tM0\t0.9\t0.02\t1\t0.5\t0.6
\t\tF\t\tMS_F_\tM1\t0.1\t0.02\t1\t-0.5\t0.6
\t\tF\t\tMS_F_\tM0\t0.7\t0.02\t2\tNA\tNA
\t\tF\t\tMS_F_\tM1\t0.3\t0.02\t2\t1\t0.3
"""


def test_read_miso_sim_instationary(tmp_path):
    path = tmp_path / "inst.miso.sim"
    path.write_text(_MISO_SIM_INST)
    mids, fits = read_miso_sim(path)
    assert len(mids) == 1 and mids[0].scaled is False  # unscaled for INST
    assert len(mids[0].data_points) == 4
    first = fits[0]
    assert first.time_point == 1.0
    assert first.measured_value is None  # not recoverable from unscaled
    assert first.residual == pytest.approx(0.5 * 0.02)
    assert first.derivation.startswith("residual = weighted_residual * SD")
    assert fits[2].residual is None and fits[2].derivation is None


def test_checked_interval_rules():
    with pytest.warns(UserWarning, match="does not contain"):
        assert checked_interval(1.0, 2.0, 3.0, label="x") == (None, None)
    assert checked_interval(1.0, 2.0, 3.0, label="x", method="monte_carlo") == (
        2.0,
        3.0,
    )
    with pytest.warns(UserWarning, match="lower > upper"):
        assert checked_interval(
            1.0, 3.0, 2.0, label="x", method="bootstrap"
        ) == (None, None)
    assert checked_interval(1.0, "NA", None, label="x") == (None, None)


def test_parse_influx_log_flags():
    log = parse_influx_log(
        '"influx_s.py" "--TIMEIT" "--seed=3333" "--ffguess" "--zc" "-1e-4" '
        '"model.ftbl"\n'
    )
    assert log.options == [
        ("--TIMEIT", None),
        ("--seed", "3333"),
        ("--ffguess", None),
        ("--zc", "-1e-4"),
    ]


# -- INCA ---------------------------------------------------------------------


def _structs(records):
    arr = np.zeros((len(records),), dtype=[(k, object) for k in records[0]])
    for i, rec in enumerate(records):
        for key, value in rec.items():
            arr[i][key] = value
    return arr


def _par(type_, id_, val, std, free, lb=None, ub=None):
    empty = np.zeros((0, 0))
    return {
        "type": type_,
        "id": id_,
        "eqn": "",
        "val": val,
        "std": std,
        "lb": empty if lb is None else lb,
        "ub": empty if ub is None else ub,
        "free": free,
        "unit": "",
        "alf": 0.05,
    }


def _res(id_, type_, peak, time, data, std, fit):
    return {
        "id": id_,
        "type": type_,
        "peak": peak,
        "time": time,
        "data": data,
        "std": std,
        "fit": fit,
        "val": (fit - data) / std,
        "expt": "exp1",
    }


def make_inca_mat(path, *, continuation=True, stationary=True):
    """Write an INCA-like results file mirroring the real layout.

    Returns:
        The WSSR and degrees of freedom written to ``f``.
    """
    times = [0.0] if stationary else [0.5, 1.0]
    ms_res = []
    for t_i, t in enumerate(times):
        for peak, data, fit in (
            ("M0", 0.10, 0.11),
            ("M1", 0.70, 0.69),
            ("M2", 0.20, 0.18),
            ("M3", 0.0, 0.02),
        ):
            ms_res.append(
                _res(f"exp1_F1_{t_i}_0_1", "MS", peak, t, data, 0.01, fit)
            )
    flux_res = [_res("R1", "Flux", "", math.inf, 10.0, 0.1, 10.05)]
    mnt = _structs(
        [
            {
                "id": "R1",
                "type": "Flux",
                "expt": "exp1",
                "res": _structs(flux_res),
                "sres": 0.25,
            },
            {
                "id": "F1",
                "type": "MS",
                "expt": "exp1",
                "res": _structs(ms_res),
                "sres": 0.0,
            },
        ]
    )
    cont = continuation
    pars = [
        _par(
            "Net flux",
            "R1",
            10.05,
            0.1,
            0,
            9.8 if cont else None,
            10.3 if cont else None,
        ),
        _par(
            "Net flux",
            "R2 net",
            6.0,
            0.07,
            1,
            5.9 if cont else None,
            6.2 if cont else None,
        ),
        _par(
            "Exch flux",
            "R2 exch",
            6.6,
            0.33,
            1,
            6.0 if cont else None,
            7.4 if cont else None,
        ),
    ]
    if not stationary:
        pars.append(
            _par(
                "Pool",
                "B pool",
                2.0,
                math.inf,
                1,
                1.0 if cont else None,
                math.inf if cont else None,
            )
        )
    for t_i, _ in enumerate(times):
        pars.append(
            _par("Norm", f"exp1 F1 exp1_F1_{t_i}_0_1 norm", 0.98, 0.016, 1)
        )
    wssr = sum(r["val"] ** 2 for r in flux_res + ms_res)
    n_meas = len(flux_res) + len(ms_res)
    dof = n_meas - sum(p["free"] for p in pars)
    f = {
        "alf": 0.05,
        "chi2": wssr,
        "dof": dof,
        "Echi2": chi2.ppf([0.025, 0.975], dof),
        "mnt": mnt,
        "par": _structs(pars),
    }
    m = {
        "options": {
            "cont_alpha": 0.05,
            "fit_reltol": 0.01,
            "fit_starts": 5,
            "int_reltol": 0.001,
            "int_tspan": np.array([0.0, 1.0]),
            "mc_n": 100,
            "sim_ss": 1 if stationary else 0,
            "sim_tunit": "h",
        },
        "expts": {
            "id": "exp1",
            "data_ms": _structs([{"id": "F1", "state": "F", "atoms": "1 2 3"}]),
        },
    }
    scipy_io.savemat(str(path), {"f": f, "m": m})
    return wssr, dof


def test_inca_synthetic_continuation(tmp_path):
    wssr, dof = make_inca_mat(tmp_path / "fit.mat")
    run = read_inca_results(tmp_path / "fit.mat", software_version="2.3")

    assert run.id == "fit.inca"
    assert run.run_mode == ["estimation", "confidence_interval_estimation"]
    assert run.fluxomics_method == "IST_MFA"
    assert run.software.software_name == "INCA"
    assert run.multistart_count == 5
    assert run.integration_timespan is None  # stationary
    r2 = flux(run, "R2")
    assert (r2.value, r2.lower_ci, r2.upper_ci) == (6.0, 5.9, 6.2)
    assert r2.confidence_interval_method == "parameter_continuation"
    assert r2.confidence_level == pytest.approx(0.95)
    assert r2.variable_type == "free"
    assert flux(run, "R1").variable_type is None  # free = 0: dependent/fixed
    assert flux(run, "R2", "exchange").value == 6.6  # absolute in INCA
    scale = run.flux_solution.scale_factors[0]
    assert scale.measurement_id == "exp1 F1 exp1_F1_0_0_1 norm"
    assert scale.time_point is None

    fits = run.flux_solution.measurement_fits
    m2 = next(f for f in fits if f.isotopologue_index == 2)
    assert m2.measurement_id == "exp1_F1_0_0_1"
    assert m2.residual == pytest.approx(0.18 - 0.20)
    assert m2.weighted_residual == pytest.approx(-2.0)
    assert m2.time_point is None
    mid = run.flux_solution.simulated_mids[0]
    assert (mid.metabolite_id, mid.atom_positions) == ("F", "1,2,3")
    assert mid.fragment_name == "F1"
    assert mid.scaled is True

    gof = run.goodness_of_fit
    assert gof.wssr == pytest.approx(wssr)
    assert gof.degrees_of_freedom == dof == 2
    assert gof.number_of_measurements == 5
    assert gof.number_of_parameters == 3
    assert gof.chi_square_p_value == pytest.approx(chi2.sf(wssr, dof))
    assert gof.chi_square_confidence_level == pytest.approx(0.95)
    assert gof.fit_accepted is bool(
        chi2.ppf(0.025, dof) <= wssr <= chi2.ppf(0.975, dof)
    )
    opts = options(run)
    assert opts["fit_starts"].value == "5"
    assert opts["int_tspan"].value == "0.0 1.0"
    validate([run], tmp_path)


def test_inca_synthetic_monte_carlo_inst(tmp_path):
    make_inca_mat(tmp_path / "inst.mat", continuation=False, stationary=False)
    n_par = 6  # R1, R2 net, R2 exch, B pool, 2 x Norm
    samples = np.random.default_rng(0).normal(1.0, 0.01, (40, n_par))
    ci = np.array(
        [[9.9, 5.8, 6.8, 1.5, 0.9, 0.9], [10.2, 6.3, 7.0, 2.5, 1.1, 1.1]]
    )
    scipy_io.savemat(str(tmp_path / "inst_mc.mat"), {"K": samples, "CI": ci})
    # [6.8, 7.0] excludes R2 exch = 6.6: a Monte Carlo interval is kept
    run = read_inca_results(
        tmp_path / "inst.mat", mc_path=tmp_path / "inst_mc.mat"
    )
    assert run.run_mode == [
        "estimation",
        "confidence_interval_estimation",
        "monte_carlo",
    ]
    assert run.monte_carlo_runs == 40
    assert run.fluxomics_method == "INST_MFA"
    assert run.integration_timespan == [0.0, 1.0]
    assert run.time_unit == "h"
    assert options(run)["int_reltol"].value == "0.001"
    r2 = flux(run, "R2")
    assert (r2.lower_ci, r2.upper_ci) == (5.8, 6.3)
    assert r2.confidence_interval_method == "monte_carlo"
    assert flux(run, "R2", "exchange").lower_ci == 6.8
    pool = run.flux_solution.pool_size_results[0]
    assert (pool.metabolite_id, pool.value) == ("B", 2.0)
    assert pool.standard_deviation is None  # Inf dropped
    assert (pool.lower_ci, pool.upper_ci) == (1.5, 2.5)
    times = sorted(
        {
            f.time_point
            for f in run.flux_solution.measurement_fits
            if f.isotopologue_index is not None
        }
    )
    assert times == [0.5, 1.0]
    scales = run.flux_solution.scale_factors
    assert [s.time_point for s in scales] == [0.5, 1.0]
    assert [(s.lower_ci, s.upper_ci) for s in scales] == [(0.9, 1.1)] * 2
    assert len(run.flux_solution.simulated_mids) == 2
    validate([run], tmp_path)


def test_inca_without_f_raises(tmp_path):
    scipy_io.savemat(str(tmp_path / "x.mat"), {"m": {"a": 1}})
    with pytest.raises(ValueError, match="no INCA fit"):
        read_inca_results(tmp_path / "x.mat")


@pytest.mark.skipif(
    not INCA_DOWNLOADS.exists(),
    reason="downloaded INCA example files not available",
)
def test_inca_real_files(tmp_path):
    quick = read_inca_results(INCA_DOWNLOADS / "simple_model_quikstart.mat")
    r2 = flux(quick, "R2")
    assert r2.value == 6.084230283997268
    assert r2.lower_ci == 5.947701736974323
    assert quick.goodness_of_fit.wssr == 136.49091035304212
    assert quick.goodness_of_fit.fit_accepted is False
    assert quick.goodness_of_fit.degrees_of_freedom == 1

    mc = read_inca_results(
        INCA_DOWNLOADS / "simple_model_mc_tutorial.mat",
        mc_path=INCA_DOWNLOADS / "simple_model_mc_tutorial_mc.mat",
    )
    assert mc.monte_carlo_runs == 800
    assert flux(mc, "R2").confidence_interval_method == "monte_carlo"
    assert flux(mc, "R2").lower_ci == pytest.approx(5.952221572842056)

    inst = read_inca_results(INCA_DOWNLOADS / "simple_model_inst_fitting.mat")
    gof = inst.goodness_of_fit
    assert (
        gof.number_of_measurements,
        gof.number_of_parameters,
        gof.degrees_of_freedom,
    ) == (54, 23, 31)
    assert gof.fit_accepted is True
    assert gof.chi_square_p_value == pytest.approx(0.2681653574018469)
    assert inst.fluxomics_method == "INST_MFA"
    assert len(inst.flux_solution.pool_size_results) == 6
    assert len(inst.flux_solution.scale_factors) == 15
    validate([quick, mc, inst], tmp_path)


# -- FreeFlux -----------------------------------------------------------------


def fake_freeflux(inst=False):
    net = pd.Series({"R1": 100.0, "R2": 110.0, "R3": 50.0, "R4": 20.0})
    total = pd.Series(
        {"R1": 100.0, "R2_f": 130.0, "R2_b": 20.0, "R3": 50.0, "R4": 20.0}
    )
    sim_f = np.array([0.0001, 0.80, 0.19, 0.0099])
    meas_f = [
        np.array([0.0002, 0.79, 0.20, 0.0098]),
        np.array([0.001, 0.02, 0.02, 0.001]),
    ]
    res = SimpleNamespace(
        _opt_net_fluxes=net,
        _opt_total_fluxes=total,
        simulated_fluxes={"R1": 100.0},
        measured_fluxes={"R1": [100.2, 0.1]},
        n_meas=5,
        n_params=3,
        dof=2,
        is_success=True,
    )
    resid = [(s - m) / sd for s, m, sd in zip(sim_f, *meas_f)]
    resid.append((100.0 - 100.2) / 0.1)
    res._opt_obj = float(np.sum(np.square(resid)))
    if inst:
        res._opt_concs = pd.Series({"F": 1.5, "B": 0.7})
        res.simulated_inst_MDVs = {
            "F_123": {0.5: sim_f * 0.5, 1.0: sim_f},
            "B_12": {1.0: np.array([0.4, 0.6])},
        }
        res.measured_inst_MDVs = {"F_123": {1.0: meas_f}}
    else:
        res.simulated_MDVs = {"F_123": sim_f}
        res.measured_MDVs = {"F_123": meas_f}
    return res


def test_freeflux_fit_linearized_ci(tmp_path):
    res = fake_freeflux()
    ci = {"R1": [99.9, 100.1], "R2": [100.0, 120.0], "R3": [51.0, 60.0]}
    with pytest.warns(UserWarning, match="net R3"):
        run = from_freeflux_result(
            res,
            run_id="sm.freeflux",
            fitter_meta={
                "solver": "slsqp",
                "tol": 1e-6,
                "max_iters": 400,
                "version": "0.3.8",
                "runtime_seconds": 1.25,
            },
            ci=ci,
        )
    assert run.run_mode == ["estimation", "confidence_interval_estimation"]
    assert run.fluxomics_method == "IST_MFA"
    assert run.isotopic_state_space == "EMU"
    assert run.software.software_name == "freeflux"
    assert run.software.version == "0.3.8"
    assert run.optimization_method == "slsqp"
    assert run.convergence_tolerance == 1e-6
    assert run.runtime_seconds == 1.25
    assert run.converged is True
    assert {o.key: o.value for o in run.solver_options} == {
        "solver": "slsqp",
        "tol": "1e-06",
        "max_iters": "400",
    }
    r2 = flux(run, "R2")
    assert (r2.lower_ci, r2.upper_ci) == (100.0, 120.0)
    assert r2.confidence_interval_method == "linearized"
    assert r2.confidence_level == 0.95
    assert flux(run, "R3").lower_ci is None  # [51, 60] excludes 50
    assert flux(run, "R2", "forward").value == 130.0
    assert flux(run, "R2", "backward").value == 20.0
    xch = flux(run, "R2", "exchange")
    assert xch.value == 20.0
    assert xch.derivation == "xch = min(fwd, bwd)"
    assert flux(run, "R2", "forward").derivation is None
    with pytest.raises(AssertionError):
        flux(run, "R1", "exchange")  # irreversible: no exchange

    fits = run.flux_solution.measurement_fits
    m1 = next(f for f in fits if f.isotopologue_index == 1)
    assert m1.measurement_id == "F_123"
    assert m1.residual == pytest.approx(0.80 - 0.79)
    assert m1.weighted_residual == pytest.approx(0.01 / 0.02)
    r1 = next(f for f in fits if f.isotopologue_index is None)
    assert r1.residual == pytest.approx(-0.2)
    mid = run.flux_solution.simulated_mids[0]
    assert (mid.metabolite_id, mid.atom_positions) == ("F", "1,2,3")
    assert mid.scaled is False  # FreeFlux fits no scale factors

    gof = run.goodness_of_fit
    assert gof.wssr == pytest.approx(res._opt_obj)
    assert gof.degrees_of_freedom == 2
    assert gof.chi_square_p_value == pytest.approx(chi2.sf(res._opt_obj, 2))
    assert gof.chi_square_confidence_level == 0.999
    assert gof.fit_accepted is bool(
        chi2.ppf(0.0005, 2) <= res._opt_obj <= chi2.ppf(0.9995, 2)
    )
    validate([run], tmp_path)


def test_freeflux_monte_carlo_list_and_object(tmp_path):
    res = fake_freeflux()
    rng = np.random.default_rng(1)
    runs = [
        pd.Series(
            {
                "R1": 100.0,
                "R2": 110 + rng.normal(0, 2),
                "R3": 50 + rng.normal(0, 1),
                "R4": 20 + rng.normal(0, 1),
            }
        )
        for _ in range(200)
    ]
    run = from_freeflux_result(res, run_id="mc", mc_runs=runs)
    assert run.run_mode == [
        "estimation",
        "confidence_interval_estimation",
        "monte_carlo",
    ]
    assert run.monte_carlo_runs == 200
    r2 = flux(run, "R2")
    assert r2.confidence_interval_method == "monte_carlo"
    assert r2.lower_ci < 110 < r2.upper_ci
    assert flux(run, "R1").lower_ci == 100.0  # fixed flux: zero-width CI

    class FakeMC:
        net_fluxes_set = runs[:10]
        total_fluxes_set = runs[:10]

        def estimate_confidence_intervals(self, which, confidence_level):
            if which == "net":
                return {"R2": [105.0, 115.0]}
            return {"R2_f": [120.0, 140.0]}

    run2 = from_freeflux_result(res, run_id="mc2", mc_runs=FakeMC())
    assert run2.monte_carlo_runs == 10
    assert (flux(run2, "R2").lower_ci, flux(run2, "R2").upper_ci) == (
        105.0,
        115.0,
    )
    assert flux(run2, "R2", "forward").upper_ci == 140.0
    validate([run, run2], tmp_path)


def test_freeflux_inst(tmp_path):
    res = fake_freeflux(inst=True)
    run = from_freeflux_result(
        res,
        run_id="inst",
        fitter_meta={"timepoints": [0.5, 1.0], "time_unit": "s"},
        ci={"conc": {"F": [1.0, 2.0]}, "net": {"R2": [100, 120]}},
    )
    assert run.fluxomics_method == "INST_MFA"
    assert run.integration_timespan == [0.5, 1.0]
    assert run.time_unit == "s"
    pools = {p.metabolite_id: p for p in run.flux_solution.pool_size_results}
    assert pools["F"].value == 1.5
    assert (pools["F"].lower_ci, pools["F"].upper_ci) == (1.0, 2.0)
    fits = run.flux_solution.measurement_fits
    mid_fits = [f for f in fits if f.isotopologue_index is not None]
    assert {f.time_point for f in mid_fits} == {1.0}
    mids = {m.fragment_name: m for m in run.flux_solution.simulated_mids}
    assert len(mids["F_123"].data_points) == 8  # two time points
    assert mids["B_12"].measurement_id is None  # simulated only
    validate([run], tmp_path)


def test_freeflux_requires_net_fluxes():
    with pytest.raises(ValueError):
        from_freeflux_result(SimpleNamespace(), run_id="x")


# -- x3cflux ------------------------------------------------------------------


class _Classification(SimpleNamespace):
    pass


class FakeParameterSpace:
    def __init__(self, pools=False):
        self.parameter_names = ["R1.n", "R2.n", "R3.n", "R2.x"]
        self.free_parameter_names = ["R2.n", "R2.x"]
        self.net_flux_classification = _Classification(
            names=["R1", "R2", "R3"],
            free=[1],
            constraint=[0],
            dependent=[(2, "formula")],
            quasi_constraint=[],
        )
        self.exchange_flux_classification = _Classification(
            names=["R2"],
            free=[0],
            constraint=[],
            dependent=[],
            quasi_constraint=[],
        )
        if pools:
            self.parameter_names.append("B")
            self.free_parameter_names.append("B")
            self.pool_size_classification = _Classification(
                names=["B"],
                free=[0],
                constraint=[],
                dependent=[],
                quasi_constraint=[],
            )

    def compute_parameters(self, free):
        free = np.asarray(free, float)
        values = [10.0, free[0], 10.0 - free[0], free[1]]
        if len(free) > 2:
            values.append(free[2])
        return np.array(values)


class MSMeasurement(SimpleNamespace):
    pass


class FluxMeasurement(SimpleNamespace):
    pass


def make_simulator(cls_name, *, times, scale=1.25):
    unscaled = np.array([0.1, 0.6, 0.25, 0.05])
    vectors = len(times) or 1
    data = [unscaled * scale + 0.01 * (k + 1) for k in range(vectors)]
    sds = [np.full(4, 0.02) for _ in range(vectors)]
    space = FakeParameterSpace(pools=cls_name.startswith("NonStationary"))

    def compute_measurements(x):
        return [unscaled.copy() for _ in range(vectors)], [10.0]

    def compute_scaled_measurements(x):
        return [unscaled * scale for _ in range(vectors)], [10.0]

    cls = type(cls_name, (), {})
    sim = cls()
    sim.parameter_space = space
    sim.measurement_names = (["ms_F"], ["flux_R1"])
    sim.measurement_data = (data, [10.2])
    sim.measurement_standard_deviations = (sds, [0.1])
    sim.measurement_time_stamps = [list(times)]
    sim.compute_measurements = compute_measurements
    sim.compute_scaled_measurements = compute_scaled_measurements
    sim.configurations = [
        SimpleNamespace(
            name="default",
            stationary=not times,
            measurements=[
                MSMeasurement(
                    name="ms_F",
                    auto_scalable=True,
                    metabolite_name="F",
                    specification=SimpleNamespace(weights=[0, 1, 2, 3]),
                ),
                FluxMeasurement(name="flux_R1"),
            ],
        )
    ]
    wssr = (
        sum(
            float(np.sum(((unscaled * scale - d) / s) ** 2))
            for d, s in zip(data, sds)
        )
        + ((10.0 - 10.2) / 0.1) ** 2
    )
    return sim, wssr


def test_x3cflux_stationary(tmp_path):
    sim, wssr = make_simulator("StationaryCumomerSimulator", times=[])
    cov = np.array([[0.04, 0.0], [0.0, 1e6]])
    run = from_x3cflux_fit(
        sim,
        [6.0, 2.0],
        wssr,
        run_id="m.x3cflux",
        cis=[(5.5, 6.5), (1.0, 3.0)],
        ci_names=["R2.n", "R2.x"],
        free_covariance=(cov, np.array([1])),
        multistart_objectives=[wssr, wssr + 1, wssr + 2],
        optimizer_options={"tol": 1e-8, "max_iter": 1000},
        software_version="3.0.0",
    )
    assert run.fluxomics_method == "IST_MFA"
    assert run.isotopic_state_space == "cumomer"
    assert run.software.software_name == "13CFLUX3"
    assert run.optimization_method == "ipopt"
    assert run.multistart_count == 3
    assert run.configuration_name == "default"
    assert run.run_mode == ["estimation", "confidence_interval_estimation"]
    r2 = flux(run, "R2")
    assert (r2.value, r2.lower_ci, r2.upper_ci) == (6.0, 5.5, 6.5)
    assert r2.confidence_interval_method == "profile_likelihood"
    assert r2.standard_deviation == pytest.approx(0.2)
    assert r2.variable_type == "free"
    assert r2.structural_identifiability == "yes"
    xch = flux(run, "R2", "exchange")
    assert xch.structural_identifiability == "no"
    assert xch.standard_deviation is None
    assert flux(run, "R1").variable_type == "constrained"
    assert flux(run, "R3").variable_type == "dependent"
    assert flux(run, "R3").value == 4.0

    scales = run.flux_solution.scale_factors
    assert len(scales) == 1 and scales[0].value == pytest.approx(1.25)
    assert scales[0].derivation == "scaled / unscaled simulated measurement"
    fits = run.flux_solution.measurement_fits
    assert len(fits) == 5
    m1 = next(f for f in fits if f.isotopologue_index == 1)
    assert m1.simulated_value == pytest.approx(0.75)
    flux_fit = next(f for f in fits if f.measurement_id == "flux_R1")
    assert flux_fit.isotopologue_index is None
    mid = run.flux_solution.simulated_mids[0]
    assert mid.metabolite_id == "F" and len(mid.data_points) == 4
    assert mid.scaled is True

    gof = run.goodness_of_fit
    assert gof.wssr == pytest.approx(wssr)
    assert (
        gof.number_of_measurements,
        gof.number_of_parameters,
        gof.degrees_of_freedom,
    ) == (5, 3, 2)
    assert gof.fit_accepted is None  # x3cflux has no verdict
    assert gof.degrees_of_freedom_convention == (
        "n_measurements - n_free_parameters - n_auto_scaled_groups"
    )
    assert "degrees_of_freedom_convention" not in options(run)
    assert options(run)["max_iter"].value == "1000"
    validate([run], tmp_path)


def test_x3cflux_nonstationary_with_pools(tmp_path):
    sim, wssr = make_simulator("NonStationaryEMUSimulator", times=[1.0, 2.0])
    run = from_x3cflux_fit(
        sim,
        [6.0, 2.0, 0.8],
        wssr,
        run_id="inst.x3cflux",
        cis={"B": (0.5, 1.2)},
    )
    assert run.fluxomics_method == "INST_MFA"
    assert run.isotopic_state_space == "EMU"
    pool = run.flux_solution.pool_size_results[0]
    assert (pool.metabolite_id, pool.value) == ("B", 0.8)
    assert pool.variable_type == "free"
    assert (pool.lower_ci, pool.upper_ci) == (0.5, 1.2)
    scales = run.flux_solution.scale_factors
    assert [s.time_point for s in scales] == [1.0, 2.0]
    gof = run.goodness_of_fit
    # 2 x 4 MS values + 1 flux; 3 free parameters + 2 scale factors
    assert (gof.number_of_measurements, gof.degrees_of_freedom) == (9, 4)
    times = {
        f.time_point
        for f in run.flux_solution.measurement_fits
        if f.isotopologue_index is not None
    }
    assert times == {1.0, 2.0}
    validate([run], tmp_path)


def test_x3cflux_list_cis_need_names():
    pytest.importorskip("scipy")
    try:
        import x3cflux  # noqa: F401
    except ImportError:
        pass
    else:  # pragma: no cover - x3cflux installed (Linux only)
        pytest.skip("x3cflux provides the default names")
    sim, wssr = make_simulator("StationaryEMUSimulator", times=[])
    with pytest.raises(ValueError, match="ci_names"):
        from_x3cflux_fit(sim, [6.0, 2.0], wssr, run_id="x", cis=[(1, 2)])
