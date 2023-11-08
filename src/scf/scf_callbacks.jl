"""
Adds simplistic checkpointing to a DFTK self-consistent field calculation.
Requires JLD2 to be loaded.
"""
function ScfSaveCheckpoints(filename="dftk_scf_checkpoint.jld2"; keep=false, overwrite=false)
    # TODO Save only every 30 minutes or so
    function callback(info)
        if info.n_iter == 1
            isfile(filename) && !overwrite && error(
                "Checkpoint $filename exists. Use 'overwrite=true' to force overwriting."
            )
        end
        if info.stage == :finalize
            if mpi_master() && !keep
                isfile(filename) && rm(filename)  # Cleanup checkpoint
            end
        else
            scfres = (; (k => v for (k, v) in pairs(info) if !startswith(string(k), "ρ"))...)
            scfres = merge(scfres, (; ρ=info.ρout))
            save_scfres(filename, scfres)
        end
        info
    end
end



"""
Plot the trace of an SCF, i.e. the absolute error of the total energy at
each iteration versus the converged energy in a semilog plot. By default
a new plot canvas is generated, but an existing one can be passed and reused
along with `kwargs` for the call to `plot!`. Requires Plots to be loaded.
"""
function ScfPlotTrace end  # implementation in src/plotting.jl


"""
Default callback function for `self_consistent_field` and `newton`, which prints a convergence table.
"""
function ScfDefaultCallback(; show_damping=true, show_time=true)
    prev_time   = nothing
    prev_energy = NaN
    function callback(info)
        show_magn = info.basis.model.spin_polarization == :collinear
        show_diag = hasproperty(info, :diagonalization)
        show_damp = hasproperty(info, :α) && show_damping

        if show_diag
            # Gather MPI-distributed information
            # Average number of diagonalizations per k-point needed for this SCF step
            # Note: If two Hamiltonian diagonalizations have been used (e.g. adaptive damping),
            # the per k-point values are summed.
            diagiter = mpi_mean(sum(mean(diag.n_iter) for diag in info.diagonalization),
                                info.basis.comm_kpts)
        end

        !mpi_master() && return info  # Rest is printing => only do on master
        if info.stage == :finalize
            info.converged || @warn "$(info.algorithm) not converged."
            return info
        end

        # TODO We should really do this properly ... this is really messy
        if info.n_iter == 1
            label_magn = show_magn ? ("   Magnet", "   ------") : ("", "")
            label_damp = show_damp ? ("   α   ", "   ----") : ("", "")
            label_diag = show_diag ? ("   Diag", "   ----") : ("", "")
            label_time = show_time ? ("   Δtime", "   ------") : ("", "")
            @printf "n     Energy            log10(ΔE)   log10(Δρ)"
            println(label_magn[1], label_damp[1], label_diag[1], label_time[1])
            @printf "---   ---------------   ---------   ---------"
            println(label_magn[2], label_damp[2], label_diag[2], label_time[2])
        end
        E    = isnothing(info.energies) ? Inf : info.energies.total
        Δρ   = norm(info.ρout - info.ρin) * sqrt(info.basis.dvol)
        magn = sum(spin_density(info.ρout)) * info.basis.dvol

        tstr = " "^9
        if show_time && !isnothing(prev_time)
            tstr = @sprintf "   % 6s" TimerOutputs.prettytime(time_ns() - prev_time)
        end

        format_log8(e) = @sprintf "%8.2f" log10(abs(e))

        Estr    = (@sprintf "%+15.12f" round(E, sigdigits=13))[1:15]
        if isnan(prev_energy)
            ΔE = " "^9
        else
            sign = E < prev_energy ? " " : "+"
            ΔE = sign * format_log8(E - prev_energy)
        end
        Δρstr   = " " * format_log8(Δρ)
        Mstr    = show_magn ? "   $((@sprintf "%6.3f" round(magn, sigdigits=4))[1:6])" : ""
        diagstr = show_diag ? "  $(@sprintf "% 5.1f" diagiter)" : ""

        αstr = ""
        show_damp && (αstr = isnan(info.α) ? "       " : @sprintf "  % 4.2f" info.α)

        @printf "% 3d   %s   %s   %s" info.n_iter Estr ΔE Δρstr
        println(Mstr, αstr, diagstr, tstr)
        prev_energy = info.energies.total
        prev_time = time_ns()

        flush(stdout)
        info
    end
end

# TODO Convergence ideas:
#      - Flag convergence only after two subsequent steps converged

"""
Flag convergence as soon as total energy change drops below tolerance
"""
function ScfConvergenceEnergy(tolerance)
    previous_energy = NaN
    function is_converged(info)
        info.energies === nothing && return false # first iteration

        # The ρ change should also be small, otherwise we converge if the SCF is just stuck
        if norm(info.ρout - info.ρin) * sqrt(info.basis.dvol) > 10sqrt(tolerance)
            return false
        end

        error = abs(info.energies.total - previous_energy)
        previous_energy = info.energies.total
        error < tolerance
    end
end

"""
Flag convergence by using the L2Norm of the change between
input density and unpreconditioned output density (ρout)
"""
function ScfConvergenceDensity(tolerance)
    info -> (norm(info.ρout - info.ρin) * sqrt(info.basis.dvol) < tolerance)
end

"""
Flag convergence on the change in cartesian force between two iterations.
"""
function ScfConvergenceForce(tolerance)
    previous_force = nothing
    function is_converged(info)
        force = compute_forces_cart(info.basis, info.ψ, info.occupation; ρ=info.ρout)
        error = isnothing(previous_force) ? NaN : norm(previous_force - force)
        previous_force = force
        error < tolerance
    end
end


"""
Determine the tolerance used for the next diagonalization. This function takes
``|ρnext - ρin|`` and multiplies it with `ratio_ρdiff` to get the next `diagtol`,
ensuring additionally that the returned value is between `diagtol_min` and `diagtol_max`
and never increases.
"""
function ScfDiagtol(; ratio_ρdiff=0.2, diagtol_min=nothing, diagtol_max=0.03)
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

function default_diagtol(basis; tol, kwargs...)
    if any(t -> t isa TermNonlinear, basis.term)
        ScfDiagtol(; diagtol_max=0.03)
    else
        ScfDiagtol(; diagtol_max=tol / 10)
    end
end
