# Run the actual example line by line, skipping Plots
ENV["GKSwstype"] = "100"

using DFTK
using LinearAlgebra
using LinearMaps

# Read the example
code = read("examples/divAgrad_solver.jl", String)

# Remove the "using Plots" line and the final plot calls
code = replace(code, r"using Plots.*\n" => "")
code = replace(code, r"# Create plots.*" => "# Plots removed\nprintln(\"Example completed successfully!\")")

# Write to temp file and include it
write("/tmp/divAgrad_noplot.jl", code)
include("/tmp/divAgrad_noplot.jl")
