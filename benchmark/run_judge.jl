#!/bin/sh
#=
julia "$0"
exit $?
=#

using PkgBenchmark

baseline::String = length(ARGS) > 1 ? ARGS[1] : "master"
target::String   = length(ARGS) > 2 ? ARGS[2] : "HEAD"

group_target = benchmarkpkg(
    dirname(@__DIR__),
    BenchmarkConfig(; env=Dict("JULIA_NUM_THREADS" => "1"), id=target),
    resultfile = joinpath(@__DIR__, "result-target.json"),
)

group_baseline = benchmarkpkg(
    dirname(@__DIR__),
    BenchmarkConfig(; env=Dict("JULIA_NUM_THREADS" => "1"), id=baseline),
    resultfile = joinpath(@__DIR__, "result-baseline.json"),
)

judgement = judge(group_target, group_baseline)
