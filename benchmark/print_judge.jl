#!/bin/sh
#=
julia "$0"
exit $?
=#

using PkgBenchmark
using Markdown

group_target   = PkgBenchmark.readresults(joinpath(@__DIR__, "result-target.json"))
group_baseline = PkgBenchmark.readresults(joinpath(@__DIR__, "result-baseline.json"))
judgement = judge(group_target, group_baseline)
displayresult(judgement)

function printnewsection(name)
    println()
    println()
    println()
    printstyled("▃"^displaysize(stdout)[2]; color = :blue)
    println()
    printstyled(name; bold = true)
    println()
    println()
end
function displayresult(result)
    md = sprint(export_markdown, result)
    md = replace(md, ":x:" => "❌")
    md = replace(md, ":white_check_mark:" => "✅")
    return display(Markdown.parse(md))
end

printnewsection("Target result")
displayresult(group_target)

printnewsection("Baseline result")
displayresult(group_baseline)
