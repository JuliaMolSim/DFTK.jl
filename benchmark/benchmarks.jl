using BenchmarkTools
using DFTK
setup_threading()

macro include_case(SUITE, name)
  quote
    include("cases/$($name).jl")
    SUITE["$($name)"] = $(Symbol(name)).SUITE
  end
end

const SUITE = BenchmarkGroup()
@include_case SUITE "silicon"
@include_case SUITE "aluminium_rattled"
@include_case SUITE "SrVO3"
@include_case SUITE "aluminium12"
