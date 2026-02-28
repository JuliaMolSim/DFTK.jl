#
# Callbacks
#

"""
Adds checkpointing to a DFTK self-consistent field calculation.
The checkpointing file is silently overwritten. Requires the package for
writing the output file (usually JLD2) to be loaded.

- `filename`: Name of the checkpointing file.
- `compress`: Should compression be used on writing (rarely useful)
- `save_ψ`:   Should the bands also be saved (noteworthy additional cost ... use carefully)
"""
@kwdef struct ScfSaveCheckpoints
    filename::String = "dftk_scf_checkpoint.jld2"
    compress::Bool   = false
    save_ψ::Bool     = false
end
function (cb::ScfSaveCheckpoints)(info)
    if info.stage == :iterate
        scfres = (; (k => v for (k, v) in pairs(info) if !startswith(string(k), "ρ"))...)
        scfres = merge(scfres, (; ρ=info.ρout))
        save_scfres(cb.filename, scfres; cb.save_ψ, cb.compress)
    end
    info
end

const SCF_CALLBACK_SHOW_MEMORY = convert(Bool, @load_preference("SCF_CALLBACK_SHOW_MEMORY", false))

"""
Default callback function for `self_consistent_field` methods,
which prints a convergence table.
"""
@kwdef struct ScfDefaultCallback
    show_damping::Bool     = true
    show_time::Bool        = true
    show_memory::Bool      = SCF_CALLBACK_SHOW_MEMORY
    prev_time::Ref{UInt64} = Ref(zero(UInt64))
end
function (cb::ScfDefaultCallback)(info)
    # If first iteration clear a potentially cached previous time
    info.n_iter ≤ 1 && (cb.prev_time[] = 0)

    show_magn = info.basis.model.spin_polarization in (:collinear, :full)
    show_diag = hasproperty(info, :diagonalization)
    show_damp = hasproperty(info, :α) && cb.show_damping
    show_time = hasproperty(info, :runtime_ns) && cb.show_time
    show_memory = cb.show_memory

    if show_diag
        # Gather MPI-distributed information
        # Average number of diagonalizations per k-point needed for this SCF step
        # Note: If two Hamiltonian diagonalizations have been used (e.g. adaptive damping),
        # the per k-point values are summed.
        diagiter = mpi_mean(sum(mean(diag.n_iter) for diag in info.diagonalization),
                            info.basis.comm_kpts)
    end

    show_gpumem = false
    if show_memory
        # Gather memory information and maximise over MPI processes
        mem_usage = mpi_max(memory_usage(info.basis.architecture), info.basis.comm_kpts)
        show_gpumem = hasproperty(mem_usage, :gpu)
    end

    !mpi_master() && return info  # Rest is printing => only do on master
    if info.stage == :finalize
        info.converged || @warn "$(info.algorithm) not converged."
        return info
    end

    # TODO We should really do this properly ... this is really messy
    if info.n_iter == 1
        if info.basis.model.spin_polarization == :full
            label_magn = ("   Magnet (x, y, z)           |Magn|", "   ----------------           ------")
        elseif show_magn
            label_magn = ("   Magnet   |Magn|", "   ------   ------")
        else
            label_magn = ("", "")
        end
        label_damp = show_damp   ? ("   α   ",   "   ----")   : ("", "")
        label_diag = show_diag   ? ("   Diag",   "   ----")   : ("", "")
        label_time = show_time   ? ("   Δtime ",  "   ------") : ("", "")
        label_memo = show_memory ? ("   Memory",  "   ------") : ("", "")
        label_dmem = show_gpumem ? ("   GPUmem",  "   ------") : ("", "")
        @printf "n     Energy            log10(ΔE)   log10(Δρ)"
        println(label_magn[1], label_damp[1], label_diag[1], label_time[1], label_memo[1], label_dmem[1])
        @printf "---   ---------------   ---------   ---------"
        println(label_magn[2], label_damp[2], label_diag[2], label_time[2], label_memo[2], label_dmem[2])
    end
    E    = isnothing(info.energies) ? Inf : info.energies.total
    
    if info.basis.model.spin_polarization == :full
        # Integrate the vector magnetization over the unit cell
        magn_x = sum(@view info.ρout[:, :, :, 2]) * info.basis.dvol
        magn_y = sum(@view info.ρout[:, :, :, 3]) * info.basis.dvol
        magn_z = sum(@view info.ρout[:, :, :, 4]) * info.basis.dvol
        magn = [magn_x, magn_y, magn_z]
        abs_magn = norm(magn)
    elseif show_magn
        magn = sum(spin_density(info.ρout)) * info.basis.dvol
        abs_magn = sum(abs, spin_density(info.ρout)) * info.basis.dvol
    else
        magn = 0.0
        abs_magn = 0.0
    end

    tstr = cb.show_time ? " "^9 : ""
    if show_time
        tstr = @sprintf "   % 6s" TimerOutputs.prettytime(info.runtime_ns - cb.prev_time[])
    end
    cb.prev_time[] = info.runtime_ns

    memstr = ""
    if show_memory
        memstr = @sprintf "  % 6s" TimerOutputs.prettymemory(mem_usage.gc_bytes)
        if show_gpumem
            memstr *= @sprintf "  % 6s" TimerOutputs.prettymemory(mem_usage.gpu)
        end
    end

    Estr    = (@sprintf "%+15.12f" round(E, sigdigits=13))[1:15]
    if info.n_iter < 2
        ΔE = " "^9
    else
        prev_energy = info.history_Etot[end-1]
        sign = E < prev_energy ? " " : "+"
        ΔE = sign * format_log8(E - prev_energy)
    end
    Δρstr   = " " * format_log8(last(info.history_Δρ))
    
    if info.basis.model.spin_polarization == :full
        Mstr = @sprintf "  [%5.2f, %5.2f, %5.2f]" magn[1] magn[2] magn[3]
        absMstr = @sprintf "  %6.3f" abs_magn
    elseif show_magn
        Mstr    = "  $((@sprintf "%6.3f" round(magn, sigdigits=4))[1:6])"
        absMstr = "  $((@sprintf "%6.3f" round(abs_magn, sigdigits=4))[1:6])"
    else
        Mstr = ""
        absMstr = ""
    end
    diagstr = show_diag ? "  $(@sprintf "% 5.1f" diagiter)" : ""

    αstr = ""
    show_damp && (αstr = isnan(info.α) ? "       " : @sprintf "  % 4.2f" info.α)

    @printf "% 3d   %s   %s   %s" info.n_iter Estr ΔE Δρstr
    println(Mstr, absMstr, αstr, diagstr, tstr, memstr)

    flush(stdout)
    info
end

format_log8(e) = @sprintf "%8.2f" log10(abs(e))

#
# Convergence checks
#

# TODO Convergence ideas:
#      - Flag convergence only after two subsequent steps converged

"""
Flag convergence as soon as total energy change drops below a tolerance.
"""
struct ScfConvergenceEnergy
    tolerance::Float64
end
function (conv::ScfConvergenceEnergy)(info)
    if last(info.history_Δρ) > 10sqrt(conv.tolerance)
        return false  # The ρ change should also be small to avoid the SCF being just stuck
    end
    length(info.history_Etot) < 2 && return false
    ΔE = (info.history_Etot[end-1] - info.history_Etot[end])
    ΔE < conv.tolerance
end

"""
Flag convergence by using the L2Norm of the density change in one SCF step.
"""
struct ScfConvergenceDensity
    tolerance::Float64
end
(conv::ScfConvergenceDensity)(info) = last(info.history_Δρ) < conv.tolerance

"""
Flag convergence on the change in Cartesian force between two iterations.
"""
mutable struct ScfConvergenceForce
    tolerance
    previous_force
end
ScfConvergenceForce(tolerance) = ScfConvergenceForce(tolerance, nothing)
function (conv::ScfConvergenceForce)(info)
    # If first iteration clear a potentially cached previous force
    info.n_iter ≤ 1 && (conv.previous_force = nothing)
    force = compute_forces_cart(info.basis, info.ψ, info.occupation; ρ=info.ρout)
    error = isnothing(conv.previous_force) ? NaN : norm(conv.previous_force - force)
    conv.previous_force = force
    error < conv.tolerance
end


@doc raw"""
Algorithm for the tolerance used for the next diagonalization. This function takes
``|ρ_{\rm next} - ρ_{\rm in}|`` and multiplies it with `ratio_ρdiff` to get the next `diagtol`,
ensuring additionally that the returned value is between `diagtol_min` and `diagtol_max`
and never increases.

# Examples
For difficult cases with bad SCF convergence it can be helpful to *reduce*
`ratio_ρdiff` to a slightly smaller value to enforce the bands to be converged more
tightly in an SCF step. For example:
```julia
diagtolalg = AdaptiveDiagtol(; ratio_ρdiff=0.05)
self_consistent_field(basis; diagtolalg, kwargs...)
```
"""
@kwdef struct AdaptiveDiagtol
    ratio_ρdiff   = 0.2
    diagtol_min   = nothing  # Minimal tolerance (autodetermined from number type if unset)
    diagtol_max   = 0.005    # Maximal tolerance
    diagtol_first = 6diagtol_max  # Relaxed tolerance used on first iteration
end
function determine_diagtol(alg::AdaptiveDiagtol, info)
    info.n_iter ≤ 1 && return min(alg.diagtol_first, 5alg.diagtol_max)

    # TODO if n_iter is small and the eigenvector residuals are all rather small,
    #      then maybe we should clamp the tolerance more aggressively
    #      (this likely indicates a restart or an extremely good initial guess)

    # This ensures diagtol can only shrink during an SCF
    diagtol = minimum(info.history_Δρ .* alg.ratio_ρdiff)
    @assert isfinite(diagtol)

    diagtol_min = something(alg.diagtol_min, 100eps(eltype(info.history_Δρ)))
    clamp(diagtol, diagtol_min, alg.diagtol_max)
end
# Note: In the past we experimented with more involved criteria for adaptively
# selecting the diagonalization tolerance, e.g. versions that take the system size
# into account (e.g. ratio_ρdiff = 10/Nocc or a criterion similar to the adaptive
# Sternheimer tolerance, i.e. ratio_ρdiff = 2Ω / (sqrt(prod(fft_size)) * Nocc)).
# The differences were very minor and we decided for the simpler heuristic above
# as opposed to any of these more sophisticated criteria.

function default_diagtolalg(basis; tol, kwargs...)
    if any(t -> t isa TermNonlinear, basis.terms)
        AdaptiveDiagtol()
    else
        AdaptiveDiagtol(; diagtol_first=tol/5)
    end
end
