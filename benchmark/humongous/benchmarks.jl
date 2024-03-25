using BenchmarkTools
using TestItemRunner

function run_scenario(scenario, complexity)
    scenario_filter(i) = occursin(string(scenario), i.filename) && complexity ∈ i.tags
    @run_package_tests filter=scenario_filter
end

all_scenarios() = [:AlSiO2H, :Cr19, :Fe2MnAl, :Mn2RuGa, :WFe]
function make_suite(; scenarios=all_scenarios(), complexity=:debug)
    @assert complexity ∈ [:debug, :small, :full]
    @assert all(scenarios .∈ Ref(all_scenarios()))

    suite = BenchmarkGroup()
    for scenario in scenarios
        suite[scenario] = @benchmarkable run_scenario($scenario, $complexity)
    end
    suite
end

const SUITE = make_suite(; scenarios=[:AlSiO2H])
