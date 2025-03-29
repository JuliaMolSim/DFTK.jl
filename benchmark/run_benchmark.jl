#!/bin/sh
#=
julia "$0"
exit $?
=#

using PkgBenchmark
using DFTK
setup_threading(; n_blas=1)

benchmarkpkg(
    dirname(@__DIR__),
    BenchmarkConfig(; env=Dict("JULIA_NUM_THREADS" => "1"));
    resultfile=joinpath(@__DIR__, "result.json"),
)
