ROOTPATH  = joinpath(@__DIR__, "..")
import Pkg
Pkg.activate(@__DIR__)
if !isfile(joinpath(@__DIR__, "Manifest.toml"))
    Pkg.develop(Pkg.PackageSpec(; path=ROOTPATH))
    Pkg.instantiate()
end

using BenchmarkCI

# Remove target once merged. Regression tests will only work after this is merged in master.
BenchmarkCI.judge(; baseline="HEAD")

BenchmarkCI.displayjudgement()
