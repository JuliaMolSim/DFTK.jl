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
