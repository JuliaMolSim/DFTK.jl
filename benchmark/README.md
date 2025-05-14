# Basic DFTK benchmark suite
## Benchmarking locally
```plain
cd benchmarks
./run_benchmarks.jl benchmark <target>
```
Run the benchmarks locally on a target commit.
If the target commit is missing, the benchmark is run on the currently checked out code
(which may be dirty). Most commonly you will just want to run
```plain
./run_benchmarks.jl benchmark
```

## Judging locally
```plain
cd benchmarks
./run_benchmarks.jl judge <baseline> <target>
```
Run locally comparing baseline commit against target commit.

Omitting target compares the current copy of the code against a baseline.
Omitting both baseline and target compares the current code against master.

## Adding new benchmarks
To add a new benchmark suite `xyx`, create a new file `cases/xyx.jl` with the following structure:

```julia
module xyx
include("common.jl")
const SUITE = BenchmarkGroup()

[... Further benchmark code ...]

end
```

Add the following line to the `benchmarks.jl`:
```julia
@include_bm "xyz"
```
This includes this new benchmark into the global suite.


## References
- [BenchmarkTools](https://juliaci.github.io/BenchmarkTools.jl/stable/)
- [PkgBenchmark](https://juliaci.github.io/PkgBenchmark.jl/stable)
- [Gridap benchmarks](https://github.com/gridap/Gridap.jl/blob/master/benchmark/README.md)
