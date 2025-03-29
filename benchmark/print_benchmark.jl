#!/bin/sh
#=
julia "$0"
exit $?
=#

using PkgBenchmark
using Markdown

function displayresult(result)
    md = sprint(export_markdown, result)
    md = replace(md, ":x:" => "❌")
    md = replace(md, ":white_check_mark:" => "✅")
    return display(Markdown.parse(md))
end

result = PkgBenchmark.readresults(joinpath(@__DIR__, "result.json"))
displayresult(result)
