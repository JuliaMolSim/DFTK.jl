using PkgBenchmark
using Markdown

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
    display(Markdown.parse(md))
end


function run_benchmark(id; nmpi=1, print_results=true)
    @assert nmpi == 1 # TODO Later also run benchmarks on multiple MPI processors

    juliacmd = `$(joinpath(Sys.BINDIR, Base.julia_exename()))`
    env = Dict("JULIA_NUM_THREADS" => "1", "OMP_NUM_THREADS" => "1")

    fn = "result_current.json"
    if !isnothing(id)
        fn = "result_$(id).json"
    end
    resultfile = joinpath(@__DIR__, "results", fn)

    if isfile(resultfile) && !isnothing(id)
        result = PkgBenchmark.readresults(resultfile)
    else
        config = BenchmarkConfig(; env, id, juliacmd)
        result = benchmarkpkg(dirname(@__DIR__), config; resultfile)
    end

    if print_results
        displayresult(result)
    end
    result
end

function run_judge(baseline, target; print_results=true, kwargs...)
    group_target   = run_benchmark(;id=target, print_results=false, kwargs...)
    group_baseline = run_benchmark(;id=target, print_results=false, kwargs...)
    judgement      = judge(group_target, group_baseline)

    if print_results
        displayresult(judgement)
        printnewsection("Target result")
        displayresult(group_target)
        printnewsection("Baseline result")
        displayresult(group_baseline)
    end
end
