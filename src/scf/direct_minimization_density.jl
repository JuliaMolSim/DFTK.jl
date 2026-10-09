using Optim
using LineSearches

"""
Minimize an orbital-free model over its density using L-BFGS.
Currently restricted to spin-unpolarised, zero-temperature Float64 CPU calculations.
"""
function direct_minimization_density(basis::PlaneWaveBasis{T};
                                     ρ=nothing, tol=1e-6, maxiter=300,
                                     show_trace=true) where {T}
    model = basis.model
    @assert T == Float64 && basis.architecture isa CPU
    @assert model.n_spin_components == 1
    @assert iszero(model.temperature) && isnothing(model.εF)
    mpi_nprocs(basis.comm_kpts) == 1 ||
        error("Density direct minimization with MPI is not supported yet")

    n_electrons = model.n_electrons
    # ρ = (n_electrons / dvol) x² with ‖x‖ = 1 preserves positivity and charge.
    density(x) = reshape((n_electrons / basis.dvol) .* x .^ 2, basis.fft_size..., 1)
    ρ = @something ρ guess_density(basis)
    x_init = vec(sqrt.(max.(total_density(ρ), zero(T))))
    @assert norm(x_init) > 0
    normalize!(x_init)

    function fg!(F, G, x)
        (; energies, ham) = energy_hamiltonian(basis, nothing, nothing; ρ=density(x))
        if !isnothing(G)
            # The integration weight dvol cancels the 1/dvol in dρ/dx.
            G .= 2n_electrons .* x .* vec(total_local_potential(ham))
        end
        isnothing(F) ? nothing : energies.total
    end

    options = Optim.Options(; iterations=maxiter, g_abstol=tol,
                            x_abstol=-1, x_reltol=-1, f_abstol=-1, f_reltol=-1,
                            show_trace)
    optim_res = Optim.optimize(Optim.only_fg!(fg!), x_init,
                               Optim.LBFGS(; manifold=Optim.Sphere(),
                                          linesearch=LineSearches.BackTracking()),
                               options)
    ρ = density(Optim.minimizer(optim_res))
    (; energies) = energy_hamiltonian(basis, nothing, nothing; ρ)
    (; basis, ρ, energies, converged=Optim.g_converged(optim_res), optim_res)
end
