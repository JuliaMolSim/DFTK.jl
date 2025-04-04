#!/bin/sh
julia -e '
    include("common.jl")
    id = length(ARGS) > 1 ? ARGS[1] : nothing
    run_benchmark(id)
' "$@"
