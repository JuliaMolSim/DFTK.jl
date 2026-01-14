# Script to run divAgrad_solver example and export plot to PDF

# First, try to add Plots if not available
using Pkg
if !haskey(Pkg.project().dependencies, "Plots")
    println("Adding Plots package...")
    Pkg.add("Plots")
end

# Set backend for non-interactive plotting
ENV["GKSwstype"] = "100"
ENV["SAVE_PLOT"] = "true"
ENV["PLOT_FILE"] = "divAgrad_result.pdf"

println("Running divAgrad_solver example...")
include("examples/divAgrad_solver.jl")
println("\n✓ Example completed successfully!")
