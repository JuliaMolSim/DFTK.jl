ROOTPATH = abspath(joinpath(@__DIR__, "../.."))
import Pkg
Pkg.activate(@__DIR__)
if !isfile(joinpath(@__DIR__, "Manifest.toml"))
    Pkg.develop(Pkg.PackageSpec(; path=ROOTPATH))
    Pkg.instantiate()
end

import BenchmarkCI
import LibGit2

"""
Launch with
```julia
julia --project=benchmark/humongous -e '
   include("benchmark/humongous/run.jl")
   run_benchmark()'
```
"""
function run_benchmark(; retune=false, baseline="origin/master", target="HEAD",
                       script=nothing)
    mktempdir(mktempdir()) do repo_dir  # TestItemRunner needs access to parent directory as well.
        project = joinpath(ROOTPATH, "benchmark", "humongous")
        # Workaround to be able to benchmark releases before the use of PkgBenchmark.
        # WARN: In this case, we need PkgBenchmark to be installed globally.
        if isnothing(script)
            # We run the default benchmark.
            script = joinpath(project, "benchmarks.jl")
        else
            occursin(ROOTPATH, abspath(script)) &&
                error("Script should be outside the repository.")
        end
        script_copy = joinpath(repo_dir, "benchmarks.jl")
        cp(script, script_copy)

        LibGit2.clone("https://github.com/epolack/DFTK-testproblems",
                      joinpath(repo_dir, "test"))

        BenchmarkCI.judge(; baseline, target, retune, script=script_copy, project)

        BenchmarkCI.displayjudgement()
    end
end
