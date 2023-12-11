using BenchmarkTools

const SUITE = BenchmarkGroup()

julia_cmd = unsafe_string(Base.JLOptions().julia_bin)
SUITE["load"] = @benchmarkable run(`$julia_cmd \
                                        --startup-file=no \
                                        --project=$(Base.active_project()) \
                                        -e 'using DFTK'`)
SUITE["pecompilation"] =
    @benchmarkable run(`$julia_cmd \
                           --startup-file=no \
                           --project=$(Base.active_project()) \
                           -e 'Base.compilecache(Base.identify_package("DFTK"))'`)
