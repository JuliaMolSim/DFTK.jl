"""
Plot the trace of an SCF, i.e. the absolute error of the total energy at
each iteration versus the converged energy in a semilog plot. By default
a new plot canvas is generated, but an existing one can be passed and reused
along with `kwargs` for the call to `plot!`.
"""
function ScfPlotTrace(plt=plot(yaxis=:log); kwargs...)
    energies = Float64[]
    function callback(info)
        if info.stage == :finalize
            minenergy = minimum(energies[max(1, end-5):end])
            error = abs.(energies .- minenergy)
            error[error .== 0] .= NaN
            extra = ifelse(:mark in keys(kwargs), (), (mark=:x, ))
            plot!(plt, error; extra..., kwargs...)
            display(plt)
        else
            push!(energies, info.energies.total)
        end
    end
end

"""
Default callback function for `self_consistent_field`, which prints a convergence table
"""
function ScfDefaultCallback()
    prev_energies = nothing
    function callback(info)
        info.stage == :finalize && return
        if info.n_iter == 1
            E_label = haskey(info.energies, "Entropy") ? "Free energy" : "Energy"
            @printf "n     %-12s      Eₙ-Eₙ₋₁     ρout-ρin   Diag\n" E_label
            @printf "---   ---------------   ---------   --------   ----\n"
        end
        E = isnothing(info.energies) ? Inf : info.energies.total
        Estr  = (@sprintf "%+15.12f" round(E, sigdigits=13))[1:15]
        prev_E = prev_energies === nothing ? Inf : prev_energies.total
        Δρ = norm(info.ρout.fourier - info.ρin.fourier)
        ΔE = prev_E == Inf ? "      NaN" : @sprintf "% 3.2e" E - prev_E
        diagiter = sum(info.diagonalization.iterations) / length(info.diagonalization.iterations)
        @printf "% 3d   %s   %s   %2.2e   % 3.1f \n" info.n_iter Estr ΔE Δρ diagiter
        prev_energies = info.energies
    end
    callback
end

"""
Flag convergence as soon as total energy change drops below tolerance
"""
function ScfConvergenceEnergy(tolerance)
    energy_total = NaN

    function is_converged(info)
        info.energies === nothing && return false # first iteration

        # The ρ change should also be small, otherwise we converge if the SCF is just stuck
        norm(info.ρout.fourier - info.ρin.fourier) > 10sqrt(tolerance) && return false

        etot_old = energy_total
        energy_total = info.energies.total
        abs(energy_total - etot_old) < tolerance
    end
    return is_converged
end

"""
Flag convergence by using the L2Norm of the change between
input density and unpreconditioned output density (ρout)
"""
function ScfConvergenceDensity(tolerance)
    info -> norm(info.ρout.fourier - info.ρin.fourier) < tolerance
end

"""
Determine the tolerance used for the next diagonalization. This function takes
``|ρnext - ρin|`` and multiplies it with `ratio_ρdiff` to get the next `diagtol`,
ensuring additionally that the returned value is between `diagtol_min` and `diagtol_max`
and never increases.
"""
function ScfDiagtol(;ratio_ρdiff=0.2, diagtol_min=nothing, diagtol_max=0.03)
    function determine_diagtol(info)
        isnothing(diagtol_min) && (diagtol_min = 100eps(real(eltype(info.ρin))))
        info.n_iter ≤ 0 && return diagtol_max
        info.n_iter == 1 && (diagtol_max /= 5)  # Enforce more accurate Bloch wave

        diagtol = norm(info.ρnext.fourier - info.ρin.fourier) * ratio_ρdiff
        diagtol = min(diagtol_max, diagtol)  # Don't overshoot
        diagtol = max(diagtol_min, diagtol)  # Don't undershoot
        @assert isfinite(diagtol)


        # Adjust maximum to ensure diagtol may only shrink during an SCF
        diagtol_max = min(diagtol, diagtol_max)
        diagtol
    end
end


