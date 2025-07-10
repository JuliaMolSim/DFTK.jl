using AtomsIO
using BenchmarkTools
using DFTK
using ForwardDiff
using LinearAlgebra
using PseudoPotentialData
using Random

function setup_dummy_scfres(basis; kwargs...)
    Random.seed!(42)
    self_consistent_field(basis; maxiter=1, callback=identity, kwargs...)
end

function setup_atomic_perturbation(scfres; n_displaced=4)
    # Displace up to n_displaced atoms into a random direction.
    #
    basis = scfres.basis
    model = scfres.basis.model
    @assert length(model.symmetries) == 1  # Symmetries disabled in model

    Random.seed!(1234)
    n_displaced = min(n_displaced, length(model.positions))
    displaced_atoms = shuffle(1:length(model.positions))[1:n_displaced]
    R = map(1:length(model.positions)) do i
        if i in displaced_atoms
            normalize!(-ones(3) + 2 * rand(3))
        else
            zeros(3)
        end
    end

    function displacement(ε::T) where {T}
        newpositions = basis.model.positions + ε * R
        modelV = Model(Matrix{T}(model.lattice), model.atoms, newpositions;
                       terms=[DFTK.AtomicLocal(), DFTK.AtomicNonlocal()], symmetries=false)
        basisV = PlaneWaveBasis(modelV; basis.Ecut, basis.kgrid)
        DFTK.total_local_potential(Hamiltonian(basisV))
    end
    δV  = ForwardDiff.derivative(displacement, 0.0)
    δHψ = -DFTK.multiply_ψ_by_blochwave(scfres.basis, scfres.ψ, δV, zeros(3))
end


#
# Benchmarks
#
function bm_scf(basis; tol=1e-6, kwargs...)
    # TODO Problem here is that benchmarktools runs a warmup
    # TODO Use setup / teardown integration of BenchmarkTools to collect and store timings using TimerOutputs
    Random.seed!(101)
    self_consistent_field(basis; tol, callback=identity, kwargs...)
end

bm_compute_forces(scfres) = compute_forces_cart(scfres)

function bm_response(scfres, δHψ; tol=1e-6, kwargs...)
    Random.seed!(101)
    DFTK.solve_ΩplusK_split(scfres, δHψ; tol, callback=identity, kwargs...)
end

# TODO other benchmarks to include:
#  - Hamiltonian application
#  - Diagonalisation
#  - Elastic constant computation

#
# Benchmark collections
#
function add_default_benchmarks!(SUITE, basis, scfres)
    SUITE["scf_3steps"]     = @benchmarkable bm_scf($basis; maxiter=3)  evals=1 samples=1
    SUITE["scf_full"]       = @benchmarkable bm_scf($basis)             evals=1 samples=1
    SUITE["compute_forces"] = @benchmarkable bm_compute_forces($scfres) evals=1 samples=3
end
