#!/bin/bash
THISDIR=$(dirname ${BASH_SOURCE[0]})
julia --project="$THISDIR/.." -e '
    include("common.jl")
    id = length(ARGS) > 1 ? ARGS[1] : nothing
    run_benchmark(id)
' "$@"
