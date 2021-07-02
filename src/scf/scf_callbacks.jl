"""
Adds simplistic checkpointing to a DFTK self-consistent field calculation.
Requires JLD2 to be loaded.
"""
function ScfSaveCheckpoints end  # implementation in src/jld2io.jl

"""
Plot the trace of an SCF, i.e. the absolute error of the total energy at
each iteration versus the converged energy in a semilog plot. By default
a new plot canvas is generated, but an existing one can be passed and reused
along with `kwargs` for the call to `plot!`. Requires Plots to be loaded.
"""
function ScfPlotTrace end  # implementation in src/plotting.jl


"""
Default callback function for `self_consistent_field` and `newton`, which prints a convergence table.
If used for newton algorithm, set algo_newton to true.
"""
function ScfDefaultCallback(; algo_newton=false)
    prev_energies = nothing
    function callback(info)
        # Gather MPI-distributed information
        # Average number of diagonalisations per k-Point needed for this SCF step
        # Note: If two Hamiltonian diagonalisations have been used (e.g. adaptive damping),
        # the per k-Point values are summed.
        diagiter = mpi_mean(sum(mean(diag.iterations) for diag in info.diagonalization),
                            info.basis.comm_kpts)

        # Rest is printing => only do on master
        !mpi_master() && return info
        if info.stage == :finalize
            info.converged || @warn algo_newton ? "Newton not converged." : "SCF not converged."
            return info
        end
        collinear = info.basis.model.spin_polarization == :collinear

        if info.n_iter == 1
            E_label = haskey(info.energies, "Entropy") ? "Free energy" : "Energy"
            magn    = collinear ? ("   Magnet", "   ------") : ("", "")
            diag    = !algo_newton ? ("Diag", "----") : ("", "")
            @printf "n     %-12s      Eₙ-Eₙ₋₁     ρout-ρin%s   %s\n" E_label magn[1] diag[1]
            @printf "---   ---------------   ---------   --------%s   %s\n" magn[2] diag[2]
        end
        E    = isnothing(info.energies) ? Inf : info.energies.total
        Δρ   = norm(info.ρout - info.ρin) * sqrt(info.basis.dvol)
        if size(info.ρout, 4) == 1
            magn = NaN
        else
            magn = sum(spin_density(info.ρout)) * info.basis.dvol
        end

        Estr   = (@sprintf "%+15.12f" round(E, sigdigits=13))[1:15]
        ΔE     = isnan(prev_energy) ? "      NaN" : @sprintf "% 3.2e" E - prev_energy
        αstr   = isnan(info.α) ? "  NaN" : @sprintf "% 4.2f" info.α
        Mstr = collinear ? "   $((@sprintf "%6.3f" round(magn, sigdigits=4))[1:6])" : ""
        if algo_newton
            @printf "% 3d   %s   %s   %2.2e%s \n" info.n_iter Estr ΔE Δρ Mstr
        else
            diagiter = sum(info.diagonalization.iterations) / length(info.diagonalization.iterations)
            @printf "% 3d   %s   %s   %2.2e%s   % 3.1f \n" info.n_iter Estr ΔE Δρ Mstr diagiter
        end
        prev_energies = info.energies

        flush(stdout)
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
        if norm(info.ρout - info.ρin) * sqrt(info.basis.dvol) > 10sqrt(tolerance)
            return false
        end

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
    info -> (norm(info.ρout - info.ρin) * sqrt(info.basis.dvol) < tolerance)
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
        info.n_iter ≤ 1 && return diagtol_max
        info.n_iter == 2 && (diagtol_max /= 5)  # Enforce more accurate Bloch wave

        ρnext = hasproperty(info, :ρnext) ? info.ρnext : info.ρout
        diagtol = (norm(ρnext - info.ρin)
                   * sqrt(info.basis.dvol)
                   * ratio_ρdiff)

        # TODO Quantum espresso divides diagtol by the number of electrons
        diagtol = clamp(diagtol, diagtol_min, diagtol_max)
        @assert isfinite(diagtol)

        # Adjust maximum to ensure diagtol may only shrink during an SCF
        diagtol_max = min(diagtol, diagtol_max)
        diagtol
    end
end


