# Developer setup

## Source code installation
If you want to start developing DFTK it is highly recommended
that you setup the sources in a way such that Julia can automatically keep
track of your changes to the DFTK code files during your development.
This means you should not `Pkg.add` your package, but use `Pkg.develop` instead.
With this setup also tools such as [Revise.jl](https://github.com/timholy/Revise.jl)
can work properly. Note that using Revise.jl is highly recommended
since this package automatically refreshes changes to the sources
in an active Julia session (see its docs for more details).

To achieve such a setup you have two recommended options:

1. Clone [DFTK](https://dftk.org) into a location of your choice
   ```bash
   $ git clone https://github.com/JuliaMolSim/DFTK.jl /some/path/
   ```
   Whenever you want to use exactly this development version of DFTK
   in a [Julia environment](https://julialang.github.io/Pkg.jl/v1/environments/)
   (e.g. the global one) add it as a `develop` package:
   ```julia
   import Pkg
   Pkg.develop("/some/path/")
   ```
   To run a script or start a Julia REPL using exactly this source tree
   as the DFTK version, use the `--project` flag of Julia,
   see [this documentation](https://julialang.github.io/Pkg.jl/v1/environments/)
   for details. For example to start a Julia REPL with this version of DFTK use
   ```bash
   $ julia --project=/some/path/
   ```
   The advantage of this method is that you can easily have multiple
   clones of DFTK with potentially different modifications made.

2. Add a development version of DFTK to the global Julia environment:
   ```julia
   import Pkg
   Pkg.develop("DFTK")
   ```
   This clones DFTK to the path `~/.julia/dev/DFTK"` (on Linux).
   Note that with this method you cannot install both the stable
   and the development version of DFTK into your global environment.

## Disabling precompilation

For the best experience in using DFTK we employ
[PrecompileTools.jl](https://github.com/JuliaLang/PrecompileTools.jl) to
reduce the time to first SCF. However,
spending the additional time for precompiling DFTK is usually not worth it during development.
We therefore recommend disabling precompilation in a development setup.
See the [PrecompileTools documentation](https://julialang.github.io/PrecompileTools.jl/stable/)
for detailed instructions how to do this.

At the time of writing dropping a file `LocalPreferences.toml` in DFTK's root folder
(next to the `Project.toml`) with the following contents is sufficient:
```toml
[DFTK]
precompile_workload = false
```

A template file `LocalPreferences.toml.example` is provided in the repository
root that you can copy and modify as needed.

## Package depot and caching

Julia stores downloaded packages and precompilation cache in the *depot* directory,
which by default is located at `~/.julia` (Linux/macOS) or `%USERPROFILE%\.julia` (Windows).
This depot is persistent across Julia sessions and projects.

### Understanding Julia's caching behavior

When you run `Pkg.instantiate()` in a project:
1. **Dependency resolution:** Julia reads `Project.toml` and resolves all dependencies,
   creating or updating `Manifest.toml` with exact versions.
2. **Package download:** If packages are not already in the depot, they are downloaded
   to `~/.julia/packages/`.
3. **Precompilation:** Packages are compiled and the cache is stored in
   `~/.julia/compiled/`.

In subsequent sessions using the same `Manifest.toml`:
- Julia reuses downloaded packages (no re-download)
- Julia reuses precompilation cache (no recompilation)
- Only packages that changed or have modified dependencies are recompiled

### For automated tools and CI systems

If you're using DFTK in an automated environment (like GitHub Copilot agents,
CI/CD pipelines, or containers), ensure the Julia depot is persistent across runs:

1. **Persistent depot path:** Set the environment variable `JULIA_DEPOT_PATH` to
   a persistent location:
   ```bash
   export JULIA_DEPOT_PATH="/persistent/path/.julia:$JULIA_DEPOT_PATH"
   ```

2. **Manifest.toml:** While gitignored by default, committing a `Manifest.toml`
   (or using a cached one) ensures reproducible package versions. However, this
   is generally not recommended for libraries.

3. **GitHub Actions:** The workflow in `.github/workflows/ci.yaml` uses
   `julia-actions/cache@v2` which automatically caches the Julia depot between runs.

For development, the default depot at `~/.julia` should work well and persist
packages across sessions automatically.
