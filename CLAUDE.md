# CLAUDE.md — AI Assistant Guide for DFTK.jl

## Project Overview

**DFTK** (Density-Functional Toolkit) is a Julia library for plane-wave density-functional theory (DFT) calculations. Its emphasis is on **simplicity and flexibility** to facilitate algorithmic/numerical research and interdisciplinary collaboration in solid-state physics. ~10k lines of code, production-quality, research-focused.

- **Version:** 0.7.24 | **Julia requirement:** ≥ 1.10 | **License:** MIT
- **Authors:** Michael F. Herbst, Antoine Levitt
- **Docs:** https://docs.dftk.org

---

## Repository Layout

```
DFTK.jl/
├── src/                    # Core library (~105 .jl files)
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
├── test/                   # Test suite (~101 .jl files)
│   ├── runtests.jl         # Main test entry point (supports arguments)
│   ├── runtests_runner.jl  # Test runner (used directly for MPI)
│   ├── runtests_parser.jl  # Test argument parsing
│   └── testcases.jl        # Parameterized test systems (Silicon, Magnesium)
├── docs/                   # Documentation source (Literate.jl + Documenter.jl)
│   ├── src/developer/      # Developer guides (conventions, style, data structures)
│   └── src/examples/       # 40+ worked examples as .jl files
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

### Typical Calculation Flow

```julia
using DFTK, AtomsBuilder, PseudoPotentialData

# 1. Define pseudopotentials and construct model
pseudopotentials = PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth")
model = model_DFT(bulk(:Si); functionals=LDA(), pseudopotentials)

# 2. Discretize in plane-wave basis
basis = PlaneWaveBasis(model; Ecut=15, kgrid=[4, 4, 4])

# 3. Run SCF
scfres = self_consistent_field(basis; tol=1e-6)

# 4. Post-process
forces  = compute_forces_cart(scfres)
bands   = compute_bands(scfres, ...)
```

---

## Development Workflows

### Setting Up for Development

```bash
# Clone repo
git clone https://github.com/JuliaMolSim/DFTK.jl /some/path/

# Use it as a dev package in Julia
julia -e 'import Pkg; Pkg.develop("/some/path/")'

# Or start REPL directly against the source
julia --project=/some/path/
```

Use [Revise.jl](https://github.com/timholy/Revise.jl) for auto-reloading changes:
```julia
using Revise
using DFTK
```

To disable precompilation (recommended during development), add `LocalPreferences.toml` in the repo root:
```toml
[DFTK]
precompile_workload = false
```

### Running Tests

Tests support filtering via `test_args`:
```bash
# Minimal (fast tests only)
julia --project -e 'import Pkg; Pkg.test("DFTK"; test_args=["minimal"])'

# Full suite minus slow tests
julia --project -e 'import Pkg; Pkg.test("DFTK"; test_args=["nominimal-noslow"])'

# MPI parallel tests (uses 2 processes by default)
julia --project -e 'import Pkg; Pkg.test("DFTK"; test_args=["mpi"])'

# GPU tests
julia --project -e 'import Pkg; Pkg.test("DFTK"; test_args=["gpu"])'

# Exclude slow tests
julia --project -e 'import Pkg; Pkg.test("DFTK"; test_args=["noslow"])'
```

The MPI test count is controlled by `DFTK_TEST_NPROCS` (default: 2).

### Building Documentation

```bash
julia --project=docs docs/make.jl
```

To build only specific files, edit `JL_FILES_TO_EXECUTE` in `docs/make.jl`. For fastest iteration (text changes only), uncomment `draft=true` in `makedocs`. Set `DEBUG = true` in `docs/make.jl` for verbose output.

---

## Coding Conventions

### Style (from `docs/src/developer/style_guide.md`)

- **Line length:** 92 characters max
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

Add a new term by creating a struct implementing the term interface:

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

## SCF Algorithms (`src/scf/`)

| File | Purpose |
|------|---------|
| `self_consistent_field.jl` | Main SCF loop |
| `scf_solvers.jl` | Solver dispatch (damping, Anderson, etc.) |
| `mixing.jl` | Charge mixing (Kerker, Anderson, χ₀-based) |
| `potential_mixing.jl` | Potential mixing |
| `anderson.jl` | Anderson acceleration |
| `direct_minimization.jl` | Energy minimization without SCF |
| `newton.jl` | Newton's method for SCF |
| `nbands_algorithm.jl` | Adaptive band count |
| `scf_callbacks.jl` | Callbacks for monitoring SCF |

---

## Postprocessing (`src/postprocess/`)

| File | Purpose |
|------|---------|
| `band_structure.jl` | Band structure and k-path |
| `dos.jl` | Density of states (DOS, LDOS, PDOS) |
| `forces.jl` | Hellmann-Feynman forces |
| `stresses.jl` | Stress tensor |
| `phonon.jl` | Phonon modes via finite differences |
| `refine.jl` | Refine SCF result to tighter tolerance |
| `current.jl` | Current density |

---

## Extensions (`ext/`)

Extensions are loaded automatically when the triggering packages are loaded:

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
model_DFT(system; functionals=LDA(), pseudopotentials)
model_DFT(system; functionals=PBE(), pseudopotentials)
model_atomic(system; pseudopotentials)   # No e-e interaction
model_HF(system; pseudopotentials)       # Hartree-Fock
```

### Functional Shorthands
`LDA()`, `PBE()`, `PBEsol()`, `SCAN()`, `PBE0()`, `HSE06()`, `mPBE()`, `PKZB()`, etc.

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
compute_χ0(scfres)
phonon_modes(scfres, supercell_size)
refine_scfres(scfres; tol=1e-10)
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

CI runs on GitHub Actions (`.github/workflows/ci.yaml`):

| Configuration | OS | Julia | Test payload |
|---|---|---|---|
| stable / x64 | ubuntu | 1.10 | `minimal` |
| stable / x64 | ubuntu | 1.10 | `nominimal-noslow` |
| stable / x64 | ubuntu | 1.10 | `example` |
| stable / aarch64 | macOS | 1.10 | `minimal` |
| stable / x64 | ubuntu | 1.10 | `noslow-mpi` |
| latest / x64 | ubuntu | latest | `minimal` |
| stable / x64 | windows | 1.10 | `minimal` (non-PR only) |

Coverage is collected via Codecov. Windows tests only run on non-PR builds.

---

## Testing Patterns

- **`TestItemRunner`** based — test items tagged with symbols (`:minimal`, `:slow`, `:gpu`, `:dont_test_mpi`)
- **`testcases.jl`** — defines standard parameterized systems (Silicon, Magnesium) reused across tests
- **`run_scf_and_compare.jl`** — utility for SCF convergence regression tests
- To add a test file, it must be `@testitem`-annotated and picked up by `runtests_runner.jl`

---

## Common Pitfalls

1. **Units:** Always use Bohr/Hartree internally. Use `austrip()` / `auconvert()` when interfacing with external code or user input in SI/eV/Å.

2. **Coordinate systems:** k-points and G-vectors are in reduced coordinates by default. Functions suffixed `_cart` return Cartesian coordinates.

3. **Column-major lattice:** `model.lattice[:, i]` gives the i-th lattice vector. Don't use row `i` by mistake — this is a common Julia/Python gotcha.

4. **FFT normalization:** DFTK uses physical normalization in real space and ℓ²-normalized coefficients in reciprocal space. Be careful when interfacing with external FFT routines.

5. **MPI awareness:** Many arrays (k-points, wavefunctions) are distributed across MPI ranks. Use `mpi_master()` guards for I/O and `mpi_sum()` for reductions.

6. **Extensions:** GPU, I/O, and Wannier functionality only works when the corresponding Julia packages are loaded (`using CUDA`, `using JLD2`, etc.). Don't `import`/`using` them in core source files — put code in `ext/`.

7. **Precompilation:** During development, disable precompilation via `LocalPreferences.toml` (see Developer Setup above) to avoid waiting on recompilation.

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `FFTW` | Fast Fourier transforms |
| `KrylovKit` | Krylov subspace methods (LOBPCG, GMRES, Lanczos) |
| `Libxc` | XC functionals (LDA, GGA, meta-GGA, hybrids) |
| `DftFunctionals` | Functional dispatch wrapper for Libxc |
| `Spglib` | Crystal symmetry analysis |
| `Brillouin` | Brillouin zone operations and k-paths |
| `AtomsBase` | Standardized atomic structure interface |
| `AtomsCalculators` | Calculator protocol (forces, energies) |
| `PseudoPotentialData` | Pseudopotential database |
| `PseudoPotentialIO` | PSP file parsers (GTH, UPF) |
| `MPI` | Distributed parallelism |
| `ForwardDiff` | Automatic differentiation |
| `DifferentiationInterface` | AD abstraction layer |
| `StaticArrays` | Efficient small fixed-size arrays (`Vec3`, `Mat3`) |
| `TimerOutputs` | Performance profiling |
| `Unitful` + `UnitfulAtomic` | Unit conversions |
| `Optim` | Optimization methods (for direct minimization) |

---

## Branch and Git Workflow

- **Main branch:** `master`
- **PR workflow:** fork → branch → PR to `master`
- Standard Julia package conventions — tag releases via `TagBot`
- `CompatHelper` automatically keeps `[compat]` bounds in `Project.toml` up to date
