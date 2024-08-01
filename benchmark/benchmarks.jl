using BenchmarkTools
using TestItemRunner

const SUITE = BenchmarkGroup()

@run_package_tests filter=ti->(:regression ∈ ti.tags)
