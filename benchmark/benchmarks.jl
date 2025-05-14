using BenchmarkTools
using DFTK
setup_threading()

macro include_case(SUITE,name)
  quote
    include("cases/$($name).jl")
    SUITE["$($name)"] = $(Symbol(name)).SUITE
  end
end

# For inspiration see
# https://github.com/gridap/Gridap.jl/blob/master/benchmark/benchmarks.jl
# https://github.com/gridap/Gridap.jl/blob/master/benchmark/README.md

const SUITE = BenchmarkGroup()

@include_bm "silicon"
@include_bm "aluminium_rattled"
@include_bm "SrVO3"
