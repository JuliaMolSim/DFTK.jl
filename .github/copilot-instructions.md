# Copilot Instructions for DFTK.jl

## Project Overview

DFTK (Density-Functional Toolkit) is a Julia package for plane-wave density-functional theory (DFT) calculations. The codebase is approximately 10k lines emphasizing simplicity, flexibility, and interdisciplinary collaboration.

**Key Features:**
- Plane-wave DFT implementation
- GPU support (CUDA, AMDGPU)
- MPI parallelization
- Written entirely in Julia (requires Julia 1.10+)

## Repository Structure

```
DFTK.jl/
├── src/           # Main source code (~100 .jl files)
│   ├── DFTK.jl   # Main module entry point
│   ├── common/    # Common utilities (types, constants, MPI, threading)
│   ├── eigen/     # Eigenvalue solvers
│   ├── external/  # External integrations (AtomsBase, JLD2, etc.)
│   └── ...        # Other modules (FFT, SCF, terms, postprocess, etc.)
├── ext/           # Package extensions (CUDA, AMDGPU, Plots, etc.)
├── test/          # Test suite using TestItemRunner
├── docs/          # Documentation (built with Documenter.jl)
├── examples/      # Example scripts
├── benchmark/     # Performance benchmarks
└── data/          # Data files (pseudopotentials, etc.)
```

## Development Setup

### Quick Start

1. **Clone and develop the package:**
   ```bash
   git clone https://github.com/JuliaMolSim/DFTK.jl
   cd DFTK.jl
   julia --project=.
   ```

2. **In the Julia REPL:**
   ```julia
   import Pkg
   Pkg.instantiate()  # Install dependencies
   ```

3. **Disable precompilation for faster development iteration:**
   Create `LocalPreferences.toml` in the repository root with:
   ```toml
   [DFTK]
   precompile_workload = false
   ```

### Package Caching Strategy

**Important:** To avoid redownloading and recompiling packages between Copilot sessions:

1. **Julia depot caching:** Julia stores packages in `~/.julia` by default (Linux/macOS) or `%USERPROFILE%/.julia` (Windows). This depot is persistent across sessions.

2. **Project environment:** The `Project.toml` defines dependencies. After running `Pkg.instantiate()`, a `Manifest.toml` is generated that locks exact versions.

3. **Precompilation cache:** Compiled packages are cached in `~/.julia/compiled/`. When using the same `Manifest.toml`, Julia will reuse precompiled code.

4. **For Copilot agents:**
   - The first `Pkg.instantiate()` will download and precompile packages (slow, ~5-10 minutes)
   - Subsequent sessions should reuse the cached depot if `JULIA_DEPOT_PATH` is persistent
   - Use `julia --project=.` to ensure the local Project.toml is used
   - The `LocalPreferences.toml` with `precompile_workload = false` skips DFTK-specific precompilation workload

## Building and Testing

### Run Tests

```bash
# Quick/minimal tests (recommended during development)
julia --project=. -e 'import Pkg; Pkg.test(test_args=["minimal"])'

# All tests except slow ones
julia --project=. -e 'import Pkg; Pkg.test(test_args=["noslow"])'

# Specific test tags
julia --project=. -e 'import Pkg; Pkg.test(test_args=["gpu"])'    # GPU tests
julia --project=. -e 'import Pkg; Pkg.test(test_args=["mpi"])'    # MPI tests

# Run all tests (may take 30+ minutes)
julia --project=. -e 'import Pkg; Pkg.test()'
```

**Note:** Tests use TestItemRunner. Individual test files can be run directly:
```bash
julia --project=. test/silicon_lda.jl
```

### Build Documentation

```bash
julia --project=docs docs/make.jl
# Output in docs/build/
```

### Code Quality

- **No explicit linter:** Julia convention is to follow the [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/)
- **Testing:** Comprehensive test suite in `test/` covering various DFT calculations
- **Continuous Integration:** GitHub Actions runs tests on Linux, macOS, and Windows

## Coding Conventions

### General Guidelines

- **Naming:** 
  - `snake_case` for functions and variables
  - `CamelCase` for types and modules
  - Avoid abbreviations unless standard in the field (e.g., `scf`, `fft`, `kpt`)

- **Type Annotations:**
  - Use type annotations in function signatures for clarity and performance
  - Leverage multiple dispatch where appropriate

- **Documentation:**
  - Use docstrings for all public functions
  - Follow DocStringExtensions conventions (already configured with `@template`)

- **Performance:**
  - Avoid type instabilities
  - Prefer in-place operations where applicable (functions ending in `!`)
  - Use `@views` for array slicing to avoid allocations

### Module Structure

- Main module: `src/DFTK.jl` includes all submodules
- Public API: Functions explicitly exported in `src/DFTK.jl`
- Internal functions: Not exported, use `DFTK.internal_function()` if needed

### Testing

- Each test file is self-contained with `using Test` and `using DFTK`
- Tests are tagged for selective execution (`:minimal`, `:slow`, `:gpu`, `:mpi`, etc.)
- MPI tests require special handling (see `test/runtests.jl`)

## Common Tasks

### Adding a New Feature

1. Implement in appropriate `src/` subdirectory
2. Export public functions in `src/DFTK.jl`
3. Add docstrings with examples
4. Write tests in `test/`
5. Update `docs/src/` if user-facing

### Debugging

- Use `@show`, `@info`, `@warn` for debugging output
- Enable `Revise.jl` for automatic code reloading:
  ```julia
  using Revise
  using DFTK
  ```
- Check TimerOutputs for performance profiling (built-in via `src/common/timer.jl`)

### Working with GPU Code

- GPU extensions are in `ext/DFTKCUDAExt` and `ext/DFTKAMDGPUExt`
- Test with: `julia --project=. -e 'import Pkg; Pkg.test(test_args=["gpu"])'`
- GPU support is optional (weak dependencies)

### Working with MPI

- MPI code is integrated throughout (see `src/common/mpi.jl`)
- MPI tests: `julia --project=. -e 'import Pkg; Pkg.test(test_args=["mpi"])'`
- Default test setup uses 2-4 processes (controlled by `DFTK_TEST_NPROCS`)

## Dependencies

- **Core:** LinearAlgebra, FFTW, ForwardDiff, Interpolations, Optim
- **Physics:** Libxc, PseudoPotentialIO, Spglib, Brillouin
- **Optional:** CUDA, AMDGPU (GPU support), MPI (parallelization), Plots (visualization)
- **Testing:** Test, TestItemRunner, various domain-specific packages

All dependencies are specified in `Project.toml` with version bounds in `[compat]`.

## CI/CD

- **CI:** `.github/workflows/ci.yaml` - runs test matrix on Linux, macOS, Windows
- **Documentation:** `.github/workflows/documentation.yaml` - builds and deploys docs
- **Coverage:** Uses Codecov for code coverage reports

## Resources

- **Documentation:** https://docs.dftk.org (stable) and https://docs.dftk.org/dev (development)
- **Chat:** Zulip (https://juliamolsim.zulipchat.com/#narrow/stream/332493-dftk) or Matrix
- **Examples:** See `examples/` directory for usage patterns
- **Developer docs:** `docs/src/developer/` for internal details

## Notes for Copilot

- **Prefer minimal changes:** This is a scientific code with careful numerical considerations
- **Check existing patterns:** Look at similar code before implementing new features
- **Test thoroughly:** DFT calculations are numerically sensitive; ensure tests pass
- **Understand the physics:** Ask for clarification if DFT concepts are unclear
- **Performance matters:** This code runs large-scale simulations; avoid performance regressions
