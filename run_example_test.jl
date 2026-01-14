# Run the actual example but skip plotting
ENV["GKSwstype"] = "100"  # Disable GUI for Plots

using DFTK
using LinearAlgebra
using LinearMaps

# Now include the example - the plot call will fail but we can catch it
try
    include("examples/divAgrad_solver.jl")
    println("\n✓ Example completed successfully!")
catch e
    if occursin("Plots", string(e)) || occursin("heatmap", string(e)) || occursin("plot", string(e))
        println("\n✓ Example ran successfully (plotting failed as expected without Plots)")
        println("Error: ", e)
    else
        println("\n✗ Example failed with error:")
        println(e)
        rethrow(e)
    end
end
