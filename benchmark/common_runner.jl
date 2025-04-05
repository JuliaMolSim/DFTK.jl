using PkgBenchmark
using Markdown
using LibGit2
using DFTK

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

lookup_id_in_dftk_repo(id::Nothing) = "current"
function lookup_id_in_dftk_repo(id::AbstractString)
    try
        repo = LibGit2.GitRepo(dirname(@__DIR__))
        obj  = LibGit2.GitObject(repo, id)
        sha  = string(LibGit2.GitHash(obj))
        return string(sha)
    catch e
        if e isa LibGit2.GitError
            return id
        else
            rethrow()
        end
    end
end

function run_benchmark(id=nothing; n_mpi=1, print_results=true, output_folder="dftk_benchmark")
    @assert n_mpi == 1 # TODO Later also run benchmarks on multiple MPI processors

    juliacmd = `$(joinpath(Sys.BINDIR, Base.julia_exename()))`
    env = Dict("JULIA_NUM_THREADS" => "1", "OMP_NUM_THREADS" => "1")

    id = lookup_id_in_dftk_repo(id)
    resultfile = joinpath(output_folder, "result_$(id).json")
    mdfile     = joinpath(output_folder, "result_$(id).json")

    if isfile(resultfile) && !isnothing(id)
        result = PkgBenchmark.readresults(resultfile)
    else
        mkpath(output_folder)
        config = BenchmarkConfig(; env, id, juliacmd)
        result = benchmarkpkg(dirname(@__DIR__), config; resultfile)
    end
    export_markdown(mdfile, result)

    if print_results
        displayresult(result)
    end
    result
end

function run_judge(baseline="master", target=nothing; print_results=true, kwargs...)
    group_target   = run_benchmark(;id=target,   print_results=false, kwargs...)
    group_baseline = run_benchmark(;id=baseline, print_results=false, kwargs...)
    judgement      = judge(group_target, group_baseline)

    if print_results
        displayresult(judgement)
        printnewsection("Target result")
        displayresult(group_target)
        printnewsection("Baseline result")
        displayresult(group_baseline)
    end

    judgement
end
