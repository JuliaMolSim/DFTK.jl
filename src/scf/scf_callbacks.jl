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
Default callback function for `self_consistent_field`, which prints a convergence table
"""
function ScfDefaultCallback()
    prev_total = NaN
    function callback(info)
        # Gather MPI-distributed information
        diagiter = mpi_mean(sum(mean(diag.iterations) for diag in info.diagonalization),
                            info.basis.comm_kpts)

        !mpi_master() && return info  # Printing only on master
        if info.stage == :finalize
            info.converged || @warn "SCF not converged."
            return info
        end
        collinear = info.basis.model.spin_polarization == :collinear

        if info.n_iter == 1
            E_label = haskey(info.energies, "Entropy") ? "Free energy" : "Energy"
            magn    = collinear ? ("   Magnet", "   ------") : ("", "")
            @printf "n     %-12s      Eₙ-Eₙ₋₁     ρout-ρin%s  α      Diag\n" E_label magn[1]
            @printf "---   ---------------   ---------   --------%s  -----  ----\n" magn[2]
        end
        E    = isnothing(info.energies) ? Inf : info.energies.total
        Δρ   = norm(info.ρout - info.ρin) * sqrt(info.basis.dvol)
        if size(info.ρout, 4) == 1
            magn = NaN
        else
            magn = sum(spin_density(info.ρout)) * info.basis.dvol
        end

        Estr   = (@sprintf "%+15.12f" round(E, sigdigits=13))[1:15]
        ΔE     = isnan(prev_total) ? "      NaN" : @sprintf "% 3.2e" E - prev_total
        αstr   = isnan(info.α) ? "  NaN" : @sprintf "% 4.2f" info.α
        Mstr = collinear ? "   $((@sprintf "%6.3f" round(magn, sigdigits=4))[1:6])" : ""

        @printf "% 3d   %s   %s   %2.2e%s  %s %5.1f \n" info.n_iter Estr ΔE Δρ Mstr αstr diagiter
        prev_total = info.energies.total

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

        # TODO Does it make sense to always use ρout here, since it measures better how much
        #      the SCF is converged?
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


