# Basic DFTK benchmark suite
This is a series of benchmarks for DFTK, which should be useful for performance measurement,
particularly on the GPU. These files try to cover various realistic cases with different computational
hot spots. They are meant to be used for performance measurement over time/commit in a consistent manner.
While these calculations are too large for GPU profiling (e.g. with NVIDIA nsys), they can serve as a
starting point before an isolated call to the function of interest.

These benchmarks are automatically executed over night on every commit to master,
producing a report. For technical reasons this report is currently only available
via the private repository [epfl-matmat/DFTK-benchrunner/](https://github.com/epfl-matmat/DFTK-benchrunner/).
If you think you require access please get in touch.

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
# Some comment explaining the rough characteristics of this system and why
# it is included in the benchmark set.

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
- [Earlier attempt of some DFTK benchmarking](https://github.com/mfherbst/DFTK_thread_timings)
