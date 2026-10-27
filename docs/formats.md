# Supported Formats

| Format | Software | Read | Write | Notes |
|--------|----------|------|-------|-------|
| FluxML | 13CFlux2, 13CFlux3 | ✅ | ✅ | Fully implemented |
| MTF | influx_si | ✅ | ✅ | Cumomer, `.mmet`, METAB vars supported |
| CSV/TSV/XLSX | FreeFlux | ✅ | 🔲 | No tracers/constraints from files |
| MATLAB `.m` | INCA | ✅ | 🔲 | Model scripts; write not planned |
| XML | SBML | 🔲 | 🔲 | Planned (dependency ready) |

## FluxML

XML-based format used by 13CFlux2/13CFlux3. Supports the richest
feature set including multiple experiments, NMR data, error models,
and variant atom transitions.

```python
from fluxomics_data_converter.io import parse_fluxml_file, write_fluxml

model = parse_fluxml_file("model.fml")
write_fluxml(model, "output.fml")
```

### What `write_fluxml` writes

**Natural abundance.** In FluxML the `purity` attribute of an input
`<label>` also switches natural abundance: with the attribute present —
even `purity="1.0"` — 13CFLUX gives every *unlabelled* input carbon its
natural ¹³C abundance (~1.1 %); without it the input is the ideal
isotopomer.

```xml
<label cfg="110000" purity="1.0">1.0</label>  <!-- carbons 3-6 get ~1.1 % 13C each -->
<label cfg="110000">1.0</label>               <!-- 100 % 110000 -->
```

| `natural_abundance` | Written |
|---------------------|---------|
| `False` (default) | No `purity` attribute: ideal inputs. A label purity below 1 cannot be expressed and is dropped with a warning. |
| `True` | `purity` on every label: the tracer's purity, or `1.0` when unknown. |

Pass `natural_abundance=True` to round-trip a FluxML file whose labels
carry `purity` (e.g. `EC.fml`, `Syn.fml`), and together with
`apply_na_correction=True`.

**Symmetric metabolites.** 13CFLUX reads a reaction's `<variant>` lists as
a cross product over all reducts and rproducts, and needs exactly one
reaction id per combination. The writer gives a `<variant>` list only to a
participant whose mapping really differs between the reaction's atom
maps (a symmetric product as rproduct variants, a symmetric substrate as
reduct variants); everything else gets a plain `cfg`:

```xml
<reaction id="ICL___1 ICL___2" bidirectional="false">
  <reduct id="IsoCit" cfg="C#1@1 C#2@1 C#3@1 C#4@1 C#5@1 C#6@1"/>
  <rproduct id="Glx" cfg="C#1@1 C#2@1"/>
  <rproduct id="Suc">
    <variant cfg="C#6@1 C#3@1 C#4@1 C#5@1"/>
    <variant cfg="C#5@1 C#4@1 C#3@1 C#6@1"/>
  </rproduct>
</reaction>
```

A flux given for the whole reaction (`ICL`) is split evenly over its
variant ids, as 13CFLUX's own constraints between variants require.

**Inputs and drains.** 13CFLUX balances every pool that is not an input.

- Every metabolite that no reaction can produce and that has atoms
  becomes an unlabelled `<input>` when no tracer covers it (with a
  warning).
- Every pool that is produced but never consumed gets an irreversible
  `<pool>_out` drain reaction, unless its id ends in `_ext` (the
  convention for external boundary pools) or `add_drains=False`.
- A pool that never appears as a product (e.g. one filled only by the
  backward direction of reversible reactions) is rejected by 13CFLUX;
  the writer warns about it. Write those reactions in the direction of
  their net flux.

**Flux values.** Stored flux values are written as given. Net fluxes
without a value (MTF `D` fluxes, the `_out` drains) are derived from the
steady-state mass balance, but only where the stored values determine
them uniquely; the others are left out with a warning, as FluxML allows.
A derived negative net flux on an irreversible reaction raises
`ValueError`.

## MTF (influx_si)

Multi-file text format. Each model consists of files sharing a
basename:

| File | Description |
|------|-------------|
| `.netw` | Network with reactions and atom transitions |
| `.linp` | Tracer specifications |
| `.miso` | MS/NMR isotopomer measurements |
| `.mflux` | Flux measurements |
| `.mmet` | Metabolite concentration measurements |
| `.cnstr` | Constraints (NET and XCH) |
| `.tvar` | Variable types and starting values |
| `.opt` | Options |

```python
from fluxomics_data_converter.io import parse_mtf, write_mtf

model = parse_mtf("path/to/model")  # reads all files
write_mtf(model, "path/to/output", experiment_name="exp1")
```

## FreeFlux

Tabular format (TSV/CSV/XLSX). Parsed files:

- `reactions.{tsv,csv,xlsx}` — network definition (required)
- `fluxes.{tsv,csv,xlsx}` — flux values
- `concentrations.{tsv,csv,xlsx}` — metabolite pool sizes
- `measured_MDVs.{tsv,csv,xlsx}` — steady-state MDVs
- `measured_fluxes.{tsv,csv,xlsx}` — measured fluxes
- `measured_inst_MDVs.{tsv,csv,xlsx}` — time-course MDVs

!!! note
    FreeFlux specifies tracers and constraints via Python API, not
    files. Parsed models will have empty tracers and no constraints.

```python
from fluxomics_data_converter.io import parse_freeflux

model = parse_freeflux("path/to/freeflux_dir")
```

## Linking the formats: FluxoLink records

FluxoLink is not a format in the table above: no tool reads it, and no
translation goes through it. Every translation goes straight from one tool's
file to `FluxomicsData` and on to another tool's file. A
[FluxoLink](../../fluxomics_linkml) record (YAML) ties one study's files
together:

- the reference copy of the model (network, atom maps, tracers,
  measurements), which every tool input is checked against;
- every tool input actually used, with its checksum, which file it was
  generated from (`derived_from`), and the result of checking that it
  encodes the same problem (`equivalence_checks`);
- each tool's ids mapped onto the canonical ones (`aliases`);
- each tool's results, in the run that names its exact input files.

```bash
fluxo convert EC.fml --to mtf freeflux -o runs/               # translate only
fluxo convert EC.fml --to mtf freeflux -o runs/ --link ec.yaml  # ... and record
fluxo add-input ec.yaml my_EC.netw --experiment a_STAT  # hand-written input: check + aliases
fluxo add-results ec.yaml runs/mtf/EC_a_STAT_res        # into the planned influx_si run
fluxo compare ec.yaml                                   # every run's fluxes, by canonical id
```

The check parses each written input back and compares it with the record.
Differences a writer makes by design are listed apart (`expected_differences`):
a drain for a pool nothing consumes, a sink pool as an efflux's product, an
explicit unlabelled input for a source pool. Anything else is a difference
(`equivalent: false`), for example the succinate symmetry or the tracer
purity the MTF writer drops.

The code lives in `link/` and needs the optional `linkml` extra
(`uv sync --extra linkml`), except `compare`:

| Module | Role |
|--------|------|
| `link/record.py` | Read and write record YAML (`load_record`, `save_record`) |
| `link/record_from_model.py` | Model → the record's model copy (`to_fluxo_dataset`) |
| `link/model_from_record.py` | The record's model copy → model (`from_fluxo_dataset`) |
| `link/tool_inputs.py` | Write a tool's input, check it against the record, record it (`write_tool_input`, `check_tool_input`, `record_tool_input`) |
| `link/files.py` | One `SourceFile` per physical file, with roles and checksum |
| `link/aliases.py` | Tool ids ↔ canonical ids (`infer_aliases`, `alias_map`) |
| `link/runs.py` | Put a tool's results into its planned run (`add_results`) |
| `link/results/` | Read tool output: influx_si `<prefix>_res/`, INCA `.mat`, FreeFlux and 13CFLUX v3 result objects |
| `link/equivalence.py` | `compare(a, b)`: do two models describe the same problem? Works without the extra |
| `io/format_conventions.py` | Format quirks shared by the parsers, writers and the link layer |
| `cli.py` | The `fluxo` command |

The model copy holds only what the source said: converter defaults (uniform
variant weights, placeholder bounds, `scale="auto"`) are left out, and
`from_fluxo_dataset` puts them back.

## INCA (MATLAB model scripts)

INCA models are MATLAB programs that build the model with constructor
calls and then mutate it with property assignments. The parser replays
that subset of MATLAB; it does not require MATLAB to be installed.

```python
from fluxomics_data_converter.io import parse_inca

model = parse_inca("data/benchmark/INCA/EC.m")
```

### Derived identifiers

INCA scripts contain no reaction or metabolite identifiers — INCA derives
them, and so does the parser:

| Item | Rule |
|------|------|
| Reaction IDs | `R1 … Rn` in source order |
| `m.rates.flx.val` | One entry per irreversible reaction, two (forward then reverse) per reversible reaction, in reaction order |
| `m.states.val` | One entry per metabolite, in ASCII order of the identifier |

Both vector lengths are asserted at parse time. The values are unlabeled,
so a wrong ordering rule would silently attach every flux and pool size to
the wrong reaction or metabolite.

### Representation choices

- **Identifiers are verbatim.** `Gluc.ext` stays `Gluc.ext`; the `.ext` and
  `.pre` suffixes are an INCA naming convention, not compartments.
- **Fluxes become net and exchange** (`net = f - b`, `xch = min(f, b)`) for
  comparability with the FluxML and MTF readers. The raw directional values
  are preserved as `inca_flux_forward` / `inca_flux_backward` annotations on
  each reaction, alongside the original equation as `inca_equation`.
- **Symmetric metabolites** declared with `m.mets{'X'}.sym` are expanded into
  atom-map variants named `<rxn>___1`, `<rxn>___2`, … matching the FluxML
  parser's convention.

### Diagnostics

`INCAParser` records what it could not reproduce rather than failing or
dropping it silently:

```python
from fluxomics_data_converter.io import INCAParser

parser = INCAParser()
model = parser.parse("Syn.m")
parser.warnings  # tracers on unknown pools, unevaluable calls, ...
parser.skipped   # statements the parser did not recognise
```

### Not recovered

- **Measured data.** Benchmark scripts define MS fragments and simulate;
  they contain no intensities, so groups come through and data does not.
- **Values behind opaque calls.** `mod2stoich(m)'` and `normrnd(...)` need
  MATLAB. The literal values assigned beforehand are kept and a warning is
  recorded.
- **External data files** read with `readtable`.
- **Solver configuration.** Only `sim_ss` and `int_tspan` carry model
  semantics; the rest of INCA's 33 options are informational.
