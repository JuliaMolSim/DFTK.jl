# For ScfPlotTrace() see DFTK.jl/src/plotting.jl, which is conditionally loaded upon
# Plots.jl is included.

"""
Default callback function for `self_consistent_field`, which prints a convergence table
"""
function ScfDefaultCallback()
    prev_energies = nothing
    function callback(info)
        if info.stage == :finalize
            info.converged || @warn "SCF not converged."
            return
        end
        collinear = info.basis.model.spin_polarization == :collinear
        dVol      = info.basis.model.unit_cell_volume / prod(info.basis.fft_size)

        if info.n_iter == 1
            E_label = haskey(info.energies, "Entropy") ? "Free energy" : "Energy"
            magn    = collinear ? ("   Magnet", "   ------") : ("", "")
            @printf "n     %-12s      Eₙ-Eₙ₋₁     ρout-ρin%s   Diag\n" E_label magn[1]
            @printf "---   ---------------   ---------   --------%s   ----\n" magn[2]
        end
        E    = isnothing(info.energies) ? Inf : info.energies.total
        Δρ   = norm(info.ρout.fourier - info.ρin.fourier)
        magn = isnothing(info.ρ_spin_out) ? NaN : sum(info.ρ_spin_out.real) * dVol

        Estr   = (@sprintf "%+15.12f" round(E, sigdigits=13))[1:15]
        prev_E = prev_energies === nothing ? Inf : prev_energies.total
        ΔE     = prev_E == Inf ? "      NaN" : @sprintf "% 3.2e" E - prev_E
        Mstr = collinear ? "   $((@sprintf "%6.3f" round(magn, sigdigits=4))[1:6])" : ""
        diagiter = sum(info.diagonalization.iterations) / length(info.diagonalization.iterations)
        @printf "% 3d   %s   %s   %2.2e%s   % 3.1f \n" info.n_iter Estr ΔE Δρ Mstr diagiter
        prev_energies = info.energies
        info
    end
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


