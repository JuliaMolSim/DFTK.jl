#!/bin/sh
julia -e '
    include("common.jl")
    baseline::String = length(ARGS) > 1 ? ARGS[1] : "master"
    target::String   = length(ARGS) > 2 ? ARGS[2] : nothing
    run_benchmark(baseline, target)
' "$@"
