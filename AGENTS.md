# AGENTS.md — AI Assistant Guide for DFTK.jl

**Read [`README.md`](README.md) first** — it has the one-line project pitch, a minimal
end-to-end usage example, and links to the docs (https://docs.dftk.org). This file assumes
you've seen it and focuses on things a contributor/AI needs that aren't in the README.

## Project Overview

**DFTK** (Density-Functional Toolkit) is a Julia library for plane-wave density-functional
theory (DFT). It is small, research-focused, and MIT-licensed; Julia ≥ 1.10 is required.

## Design Philosophy (read this before making changes)

DFTK exists to *"bundle the research efforts of all involved communities"* — mathematicians,
physicists, chemists, and HPC/computer scientists working on DFT — in one accessible code base.
Its founding paper ([Herbst, Levitt, Cancès, JuliaCon 2021](https://doi.org/10.21105/jcon.00069))
states the goal plainly: keep the entrance barrier low *"by designing the code in line with the
mathematical and physical structure of the DFT problem"*, so DFTK can *"rapidly prototype new
physical models or support the mathematical analysis of DFT methods"* — while still scaling to
realistic systems (1000+ electrons, GPU, MPI).

Three values, in priority order, that should guide every change here:

1. **Code structure mirrors the math.** Types and functions should map onto the objects a
   physicist/mathematician reasons about (`Model`, `PlaneWaveBasis`, `Term`, `Hamiltonian`,
   `ρ`, `ψ`, `εF`). If a change makes the code diverge from the mathematical structure to save
   a few lines or cycles, it is usually the wrong trade for this project.
2. **Clean, readable code so tinkering is easy.** The audience includes researchers who will
   fork a function to try a new algorithm. Prefer transparent, hackable implementations over
   clever or heavily abstracted ones. Readability beats consistency and beats brevity.
3. **Simplicity over feature count.** DFTK deliberately stays small rather than chasing the
   full feature list of ABINIT/QE/VASP. Be cautious about adding surface area, special cases,
   or configuration knobs; a feature that complicates the core for a niche case is usually not
   worth it. When in doubt, keep the core simple and push specialization to the edges (`ext/`,
   keyword arguments with sane defaults, separate helper functions).

Because the code itself is meant to be read, **the code is the primary documentation** — we
lean on it rather than on heavy prose. In practice this means: keep functions short so a
reader can follow the whole implementation without getting lost; don't reflexively attach a
large docstring to every function (the public API and anything non-obvious to a *user* is
documented, but internal helpers often need none); and use **inline comments to explain the
non-obvious steps** — the physics/math reasoning, a subtle normalization, a workaround —
rather than restating what the code plainly does. Prefer making the code clearer over
compensating for unclear code with more text.

---

## Repository Layout

```
DFTK.jl/
├── src/                    # Core library
│   ├── DFTK.jl             # Module entry point: all exports and includes
│   ├── Model.jl            # Physical model data structure
│   ├── PlaneWaveBasis.jl   # Basis set discretization
│   ├── Kpoint.jl           # k-point representation
│   ├── SymOp.jl            # Symmetry operations
│   ├── Smearing.jl         # Electronic temperature/occupation
│   ├── standard_models.jl  # Convenience model constructors (model_DFT, etc.)
│   ├── elements.jl         # Atomic element types
│   ├── symmetry.jl         # Crystal symmetry handling
│   ├── common/             # Shared utilities (MPI, threading, math, types)
│   ├── eigen/              # Eigenvalue solvers (LOBPCG, etc.)
│   ├── scf/                # SCF algorithms and mixing strategies
│   ├── terms/              # Energy term implementations
│   ├── postprocess/        # Band structure, forces, stresses, DOS, phonons
│   ├── response/           # Linear response (chi0, Hessian, GMRES)
│   ├── pseudo/             # Pseudopotential parsers (GTH, UPF)
│   ├── external/           # AtomsBase, Wannier90, calculator interface
│   ├── gpu/                # GPU array utilities
│   └── workarounds/        # FFT and ForwardDiff workarounds
├── ext/                    # Julia package extensions (loaded on demand)
│   ├── DFTKCUDAExt.jl      # CUDA GPU support
│   ├── DFTKAMDGPUExt.jl    # AMD GPU support
│   ├── DFTKJLD2Ext.jl      # JLD2 serialization
│   ├── DFTKJSON3Ext.jl     # JSON serialization
│   ├── DFTKPlotsExt.jl     # Visualization
│   ├── DFTKWannierExt.jl   # Wannier functions
│   └── ...                 # Other optional extensions
├── test/                   # Test suite
│   ├── runtests.jl         # Main test entry point (supports arguments)
│   ├── runtests_runner.jl  # Test runner (used directly for MPI)
│   ├── runtests_parser.jl  # Test argument parsing
│   └── testcases.jl        # Parameterized test systems (Silicon, Magnesium)
├── docs/                   # Documentation source (Literate.jl + Documenter.jl)
│   ├── src/developer/      # Developer guides (conventions, style, data structures)
│   └── src/examples/       # Worked examples as .jl files
├── examples/               # Standalone runnable scripts
├── benchmark/              # Performance benchmarks
├── data/                   # Data files (pseudopotentials, etc.)
├── Project.toml            # Package manifest and dependencies
└── .github/workflows/      # CI/CD (ci.yaml, documentation.yaml, etc.)
```

---

## Core Architecture

### Key Data Structures

1. **`Model`** (`src/Model.jl`) — Physical model specification
   - Contains: lattice, atoms, positions, spin polarization, temperature, number of electrons, list of energy terms (`term_types`)
   - Created via convenience constructors: `model_DFT()`, `model_atomic()`, `model_HF()`

2. **`PlaneWaveBasis`** (`src/PlaneWaveBasis.jl`) — Discretized basis set
   - Created from a `Model` with cutoff energy `Ecut` and k-point grid `kgrid`
   - Manages k-point sampling, FFT grids, symmetry reduction

3. **`Kpoint`** (`src/Kpoint.jl`) — Single k-point with its plane-wave G-vectors

4. **`Hamiltonian`** (`src/scf/Hamiltonian.jl`) — Assembled Hamiltonian for the current density

5. **`scfres`** — Named tuple returned by `self_consistent_field()`, containing: `basis`, `ψ` (wavefunctions), `ρ` (density), `eigenvalues`, `εF` (Fermi level), `energies`, `occupation`, convergence info

The end-to-end flow (`Model` → `PlaneWaveBasis` → `self_consistent_field` → post-processing)
is shown in the README example; the `## Key Public API` section below lists the entry points.

---

## Development Workflows

### Setting Up for Development

```bash
git clone https://github.com/JuliaMolSim/DFTK.jl /some/path/
julia --project=/some/path/   # REPL against the source (or Pkg.develop("/some/path/") into another env)
```
Always start the session with `using Revise, DFTK` (Revise first) so source edits hot-reload.

**Do not launch a fresh `julia` process for every change.** Each cold start pays package load
+ compilation (minutes), so iterating that way is painfully slow. Instead keep **one persistent
Julia session** alive and let Revise hot-reload your edits. If you (as an AI agent) don't already
have such a workflow, stop and set one up together with the user before iterating — e.g. a warm
REPL held in a tmux session, or a Julia MCP server (see
[antoine-levitt/perso/.claude](https://github.com/antoine-levitt/perso/tree/master/.claude) for
an example setup). Everything below (single-test runs, REPL experiments) assumes that persistent
session.

### A Note on Julia Compilation Times

First-run compilation ("time to first X") is substantial: `using DFTK`, the first SCF, and
`Pkg.test` precompilation can each take **several minutes**. This is normal — don't treat a
silent process under ~5 minutes as a hang. Everything after the first run in a session is fast.

### Iterating on a Feature

In the persistent session, exercise the change directly on a small system (Revise reloads edits,
so just re-run — no restart):
```julia
using AtomsBuilder, PseudoPotentialData
model = model_DFT(bulk(:Si); functionals=LDA(),
                  pseudopotentials=PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth"))
basis = PlaneWaveBasis(model; Ecut=10, kgrid=[2, 2, 2])   # keep small for speed
scfres = self_consistent_field(basis; tol=1e-4)
```

**Don't run the full suite while iterating** — run a single test item via `TestEnv` +
`TestItemRunner`. Test files are not plain scripts (they use `@testitem`/`@testsetup` with
relative-module imports like `using ..TestCases` that only resolve inside TestItemRunner), so
`include("test/silicon_lda.jl")` won't work standalone:
```julia
using TestEnv, Revise; TestEnv.activate()   # pull in test-only deps
using TestItemRunner
TestItemRunner.run_tests("test/"; filter=ti -> ti.name == "Hamiltonian consistency")   # by name
TestItemRunner.run_tests("test/"; filter=ti -> occursin("silicon_lda.jl", ti.filename))  # or file / ti.tags
```
Before a full CI run, sanity-check with `Pkg.test("DFTK"; test_args=["minimal"])`.

### Running Tests

The suite runs via [TestItemRunner](https://github.com/julia-vscode/TestItemRunner.jl).
Each test is an `@testitem` carrying **tags** (symbols). `test/runtests_parser.jl` turns the
arguments into an include/exclude filter:

- `minimal`, `slow`, `example`, `forces`, … are ordinary **tags**. Passing one **restricts**
  the run to items carrying it; passing several unions them.
- Prefix `no` to **exclude** a tag: `noslow` drops `:slow` items.
- `mpi`, `gpu`, `all` are **base modes**: `mpi` runs everything except `:dont_test_mpi`
  (and `:gpu`); `gpu` runs only `:gpu` items; `all` (the default, no args) runs everything.

**Important — two ways to pass args, with different syntax:**

```bash
# Via Pkg.test: each tag is a SEPARATE list element (no dashes).
julia --project -e 'import Pkg; Pkg.test("DFTK"; test_args=["minimal"])'
julia --project -e 'import Pkg; Pkg.test("DFTK"; test_args=["nominimal", "noslow"])'
julia --project -e 'import Pkg; Pkg.test("DFTK"; test_args=["mpi"])'
julia --project -e 'import Pkg; Pkg.test("DFTK"; test_args=["gpu"])'
julia --project -e 'import Pkg; Pkg.test("DFTK"; test_args=["noslow"])'

# Via the DFTK_TEST_ARGS env var: tags are joined with DASHES (this is what CI uses).
DFTK_TEST_ARGS=nominimal-noslow julia --project -e 'import Pkg; Pkg.test("DFTK")'
DFTK_TEST_ARGS=noslow-mpi       julia --project -e 'import Pkg; Pkg.test("DFTK")'
```

Do **not** write `test_args=["nominimal-noslow"]` — as a single element the dash is not
split, so it is misparsed. The dash form is only for the env var.

The MPI test count is controlled by `DFTK_TEST_NPROCS` (default: 2).

### Building Documentation

`julia --project=docs docs/make.jl`. In `docs/make.jl`: `JL_FILES_TO_EXECUTE` limits which
examples run, `draft=true` in `makedocs` skips code execution (fastest, text-only changes), and
`DEBUG = true` gives verbose output.

---

## Coding Conventions

### Style (from `docs/src/developer/style_guide.md`)

- **Line length:** 92 characters max (not a hard constraint, but preferred)
- **NamedTuples:** Use explicit form `(; var=val)` not `(var=val)`
- **Loops:**
  - Range: `for i = 1:10` (use `=`)
  - Collection: `for item in array` (use `in`)
- **`where` clauses:** Always use explicit braces: `where {T <: AbstractFloat}`
- **Keyword arguments:** Always explicit — no implicit positional-to-keyword promotion
- **Internal helpers:** Prefix with `_` (e.g., `_compute_density_helper`)
- **Empty callbacks:** Use `identity` as placeholder
- **Naming:** Prefer readability over brevity — don't shorten variable names just for conciseness

### Unicode Symbols

DFTK uses Unicode extensively (ensure your editor supports Julia Unicode input):

| Symbol | Meaning |
|--------|---------|
| `ψ` | Wavefunction / Bloch state |
| `ρ` | Electron density |
| `ε` | Eigenvalues |
| `εF` | Fermi level |
| `k` | k-point (Brillouin zone vector) |
| `G` | Reciprocal lattice vector |
| `q` | Phonon wavevector |
| `R` | Real-space lattice vector |
| `r`, `x` | Real-space position |
| `Ω` | Unit cell (or its volume) |
| `A` | Real-space lattice matrix (`model.lattice`) |
| `B` | Reciprocal-space lattice matrix (`model.recip_lattice`) |

### Units

**Atomic units throughout** — lengths in Bohr, energies in Hartree.

```julia
using Unitful, UnitfulAtomic
austrip(10u"eV")         # Convert 10 eV → Hartree
auconvert(u"Å", 1.2)    # Convert 1.2 Bohr → Ångström
```

When using AtomsBase objects or external tools (ASE, AtomsIO), unit conversion happens automatically.

### Coordinate Systems

- **Default:** Reduced (fractional) coordinates for all reciprocal-space vectors (k, G, q) and real-space vectors (r, R)
- **Cartesian conversion:**
  - Real space: `x_cart = model.lattice * x_red`
  - Reciprocal space: `k_cart = model.recip_lattice * k_red`
- **Lattice storage:** Column-major — `model.lattice[:, i]` is the i-th lattice vector
- **Reciprocal lattice:** `B = 2π A⁻ᵀ`
- **Real-space unit cell:** `[0, 1)³` in reduced coordinates
- **Brillouin zone:** `[-1/2, 1/2)³` in reduced coordinates

### Normalization Conventions

- **Reciprocal space:** Quantities stored as coefficients in the `e_G` (normalized plane wave) basis — `ℓ²` normalized
- **Real space:** Physical values (not scaled by volume)
- **Wavefunctions:** `norm(ψ) = 1` means properly normalized; on real-space grid: `(|Ω|/N) * sum(|ψ(r)|²) = 1`

---

## Energy Terms (`src/terms/`)

**Terms are two-level** (see `terms.jl`, and `kinetic.jl`/`hartree.jl` as templates):

1. A **builder** struct holding parameters (e.g. `Hartree`, `Kinetic`), made *callable* on a
   basis: `(t::Hartree)(basis) = TermHartree(basis, ...)`. Builders are what live in
   `model.term_types`.
2. The instantiated **`Term`** (e.g. `TermHartree`) holds basis-specific precomputed data and
   lives in `basis.terms`. That field is typed `Vector{Any}` and entries may be `nothing`, so
   always guard iteration with `isnothing(term) && continue`.

Interface each term implements:
- `ene_ops(term, basis, ψ, occupation; kwargs...) -> (; E, ops)` — the one required method.
  `ops` is one `RealFourierOperator` per k-point. The density arrives as the **`ρ` keyword**
  (`ene_ops(...; ρ, kwargs...)`), not positionally. Return `E = T(Inf)` when `ψ`/`occupation`
  are `nothing` (energy not yet defined).
- Subtype `TermLinear` (zero response kernel — `compute_kernel`/`apply_kernel` default to
  `nothing`) or `TermNonlinear`.
- Optional, all defaulting to no-op/`nothing`: `compute_kernel`/`apply_kernel` (response),
  `compute_forces`, `compute_dynmat`/`compute_δHψ_αs` (phonons), and the traits
  `breaks_symmetries` / `breaks_time_reversal_symmetry` (default `false`; set `true` for e.g.
  `Magnetic` and external potentials — otherwise k-point symmetry reduction would be invalid).

| File | Term |
|------|------|
| `kinetic.jl` | Kinetic energy |
| `local.jl` | Local potential |
| `nonlocal.jl` | Nonlocal (pseudopotential projectors) |
| `hartree.jl` | Hartree (electron-electron Coulomb) |
| `xc.jl` | Exchange-correlation (via Libxc) |
| `ewald.jl` | Ewald summation (nucleus-nucleus) |
| `psp_correction.jl` | Pseudopotential correction |
| `entropy.jl` | Entropic/smearing term |
| `hubbard.jl` | DFT+U (Hubbard correction) |
| `exact_exchange.jl` | Exact (Fock) exchange |
| `magnetic.jl` | External magnetic field |
| `anyonic.jl` | Anyonic exchange statistics |

---

## SCF & Postprocessing

- **`src/scf/`** — the main loop is `self_consistent_field.jl`; alternatives are
  `direct_minimization.jl` and `newton.jl`. Supporting pieces: `mixing.jl` /
  `potential_mixing.jl` / `anderson.jl` (acceleration), `nbands_algorithm.jl`,
  `scf_callbacks.jl`. (`ls src/scf/` for the rest.)
- **`src/postprocess/`** — one file per quantity: `band_structure.jl`, `dos.jl`,
  `forces.jl`, `stresses.jl`, `phonon.jl`, `refine.jl`, `current.jl`.

---

## Extensions (`ext/`)

Extensions load automatically when their trigger packages are loaded. **Never `import`/`using`
these packages (CUDA, JLD2, Plots, Wannier, …) from core `src/` files** — the corresponding
code belongs in `ext/`.

| Extension | Trigger | Provides |
|-----------|---------|---------|
| `DFTKCUDAExt` | `CUDA` + `MPI` + `Libxc` | NVIDIA GPU support |
| `DFTKAMDGPUExt` | `AMDGPU` + `MPI` | AMD GPU support |
| `DFTKGenericLinearAlgebraExt` | `GenericLinearAlgebra` | Arbitrary precision |
| `DFTKJLD2Ext` | `JLD2` | HDF5-based checkpoint I/O |
| `DFTKJSON3Ext` | `JSON3` | JSON checkpoint I/O |
| `DFTKPlotsExt` | `Plots` | Band/DOS visualization |
| `DFTKWannierExt` | `Wannier` | Maximally-localized Wannier functions |
| `DFTKWannier90Ext` | `wannier90_jll` | Wannier90 interface |
| `DFTKWriteVTKExt` | `WriteVTK` | VTK file export |
| `DFTKGeometryOptimizationExt` | `GeometryOptimization` | Variable-cell geometry optimization |

---

## Key Public API

### Model Construction
```julia
model_DFT(system; functionals=LDA(), pseudopotentials)   # or PBE(), SCAN(), a hybrid, ...
model_atomic(system; pseudopotentials)   # No e-e interaction
model_HF(system; pseudopotentials)       # Hartree-Fock
```

### Functional Shorthands
Exported constructors: `LDA()`, `PBE()`, `PBEsol()`, `SCAN()`, `r2SCAN()`, and the hybrids
`PBE0()`, `HSE()` (HSE06), plus the generic `HybridFunctional([...])`. Any Libxc functional can
also be passed directly by symbol, e.g. `functionals=[:lda_x, :lda_c_vwn]`.

### Basis and SCF
```julia
PlaneWaveBasis(model; Ecut, kgrid)
self_consistent_field(basis; tol=1e-6, mixing=...)
direct_minimization(basis)
newton(basis, ψ0)
```

### Analysis
```julia
compute_forces_cart(scfres)
compute_stresses_cart(scfres)
compute_bands(scfres, kpath; n_bands=...)
compute_dos(scfres; ...)
compute_ldos(scfres; ...)
compute_χ0(scfres.ham)              # takes a Hamiltonian, not the scfres
phonon_modes(scfres)                # Γ-point phonons; dispersion via the q kwarg
refine_scfres(scfres, basis_ref; tol=1e-10)   # basis_ref is a finer PlaneWaveBasis
```

### I/O
```julia
load_psp("hgh/lda/Si-q4")       # Load pseudopotential by name
list_psp()                        # List available pseudopotentials
save_scfres("result.jld2", scfres)
load_scfres("result.jld2")
```

---

## CI/CD

CI is GitHub Actions (`.github/workflows/ci.yaml`). It builds a matrix that splits the suite
into payloads passed via `DFTK_TEST_ARGS` (`minimal`, `nominimal-noslow`, `example`,
`noslow-mpi`) across Julia 1.10 + latest on ubuntu, plus a macOS-aarch64 and (non-PR only)
a Windows `minimal` run. Coverage goes to Codecov. Check the workflow file for the current
exact matrix rather than trusting a copy here.

---

## Testing Patterns

- **`TestItemRunner`** based — every test is an `@testitem "<name>" tags=[...] setup=[...] begin ... end`
  with tags such as `:minimal`, `:slow`, `:gpu`, `:dont_test_mpi`, `:dont_test_windows`, `:example`
- **Shared setup** lives in `@testsetup`/`@testmodule` blocks (e.g. `TestCases`, `RunSCF`), imported
  into an item via `using ..TestCases: silicon`. The `setup=[...]` list controls availability/order.
- **`testcases.jl`** — defines standard parameterized systems (Silicon, Magnesium) reused across tests
- **`run_scf_and_compare.jl`** — utility for SCF convergence regression tests
- Test items are auto-discovered by TestItemRunner across `test/` (no manual registration); the entry
  point is `test/runtests.jl` → `test/runtests_runner.jl` (`@run_package_tests`).

---

## Idioms Worth Knowing (from reading the source)

These are conventions the code relies on that you would *not* guess from generic Julia:

**Performance / correctness plumbing**
- **Wrap expensive functions in `@timing` / `@timing "label"`** — this writes to the global
  `DFTK.timer`. It is **not thread-safe**: never place it inside a `Threads.@threads` region.
- `PlaneWaveBasis` opts out of broadcasting (`broadcastable(basis) = Ref(basis)`), so it acts as
  a scalar inside `.` expressions — pass it freely into `map`/broadcast.

**Device-agnostic (GPU) code — the same code path runs on CPU and GPU**
- Never hardcode `zeros(...)` or `CuArray(...)` in compute code. Allocate with `zeros_like(X, ...)`
  or `similar(X, ...)` so the result lands on the same device as `X`. Move data only via
  `DFTK.to_device(arch, A)` / `DFTK.to_cpu(A)`, never `CuArray(A)` (not cross-vendor).
- **No scalar indexing into GPU arrays.** To touch one element (e.g. zero the `G=0` Fourier
  component, as in `hartree.jl`) wrap it in `GPUArraysCore.@allowscalar`. Functions that run on
  the device must take `isbits` arguments — avoid closures over non-`isbits` values like `Kpoint`
  (note the `coordinate = kpt.coordinate` hoist in `Gplusk_vectors`).
- The device lives on the basis: `basis.architecture` is `CPU()` or `GPU(CuArray)`/`GPU(ROCArray)`.

**Data layout**
- **Densities and potentials are 4D real arrays `(nx, ny, nz, n_spin)`** — spin is the last axis.
  Use `total_density(ρ)` / `spin_density(ρ)` instead of indexing the spin axis by hand.
- Reciprocal ↔ real space is `fft` / `ifft`; use **`irfft`** when the result must be real (e.g.
  a potential — plain `ifft` leaves high-frequency noise). k-point-restricted transforms take the
  `kpt`: `fft(basis, kpt, f_real)`.
- **Two different G-vector sets — don't confuse them.** `G_vectors(basis)` /
  `G_vectors_cart(basis)` is the **full cube** FFT grid (`basis.fft_size`), where densities and
  potentials live (`fft(basis, f_real)` returns coefficients on it). `G_vectors(basis, kpt)` /
  `Gplusk_vectors(basis, kpt)` / `Gplusk_vectors_cart(basis, kpt)` is the **spherical** set
  `|G+k|² < 2·Ecut` for one k-point, where *wavefunctions* live; `fft(basis, kpt, f_real)`
  returns coefficients on this sphere, i.e. it **truncates** to the orbital cutoff (`kpt.mapping`
  maps sphere→cube). A quantity like a pair density `ψᵢ*ψⱼ` has content beyond the orbital sphere,
  so it belongs on the cube grid, not the k-point sphere. `qpt`/`kpt` arguments are `Kpoint`s (they
  carry both a `.coordinate` and a G-set); a bare reduced vector is a `Vec3`.
- **k-points are MPI-distributed:** `basis.kpoints` is only *this rank's* slice. Any k-summed
  scalar must be reduced — use `weighted_ksum(basis, array)`, or sum locally then
  `mpi_sum(x, basis.comm_kpts)`. For collinear spin the list is all spin-up then all spin-down;
  `krange_spin(basis, σ)` selects one channel.
- Small fixed-size vectors/matrices are `Vec3{T}` / `Mat3{T}` (StaticArrays), not `Vector`/`Matrix`;
  the lattice is a `Mat3` with the lattice vectors in **columns**.

**Reusable helpers — grep `src/common/` before writing a small math/util function.** Frequently
re-invented ones:
- `norm2(v)` = `sum(abs2, v)` = `|v|²`; `norm_cplx(v)` = AD-safe complex-analytic `norm` (`common/norm.jl`).
- `cis2pi(x)` = `e^{2πix}`, `sin2pi`, `cos2pi` (`common/cis2pi.jl`) — with integer-argument fast paths.
- `divided_difference(f, fder, x, y)` = stable `(f(x)-f(y))/(x-y)`, incl. the `x==y` limit `f'(x)`
  (`common/divided_difference.jl`); with `y=0`, `f(0)=0` it is a stable/AD-clean `f(x)/x`.
- `zeros_like(X, ...)` / `similar` for device-agnostic allocation; `to_device`/`to_cpu` to move.
- `estimate_integer_lattice_bounds(M, δ)` — integer coordinates `n` with `‖M·n‖ ≤ δ` (`structure.jl`).
- `sphericalbesselj_fast(l, x)` (`common/spherical_bessels.jl`), `weighted_ksum`, `mpi_sum`.

---

## Key Dependencies

See `Project.toml` for the authoritative list. The ones with non-obvious roles:
`Libxc` + `DftFunctionals` (XC functional evaluation and dispatch), `Spglib` (symmetry),
`Brillouin` (k-paths / BZ), `KrylovKit` (GMRES), `PseudoPotentialData` +
`PseudoPotentialIO` (PSP database and GTH/UPF parsers), `AtomsBase`/`AtomsCalculators`
(structure and calculator interfaces), `StaticArrays` (`Vec3`/`Mat3`), and `ForwardDiff` +
`DifferentiationInterface` (AD). FFTW, MPI, Unitful, Optim, TimerOutputs are used as their
names suggest.

---

## Branch and Git Workflow

- **Main branch:** `master`
- **PR workflow:** fork → branch → PR to `master`
- Standard Julia package conventions — tag releases via `TagBot`
- `CompatHelper` automatically keeps `[compat]` bounds in `Project.toml` up to date
