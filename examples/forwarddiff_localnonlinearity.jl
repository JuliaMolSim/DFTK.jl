using DFTK
using ForwardDiff
using LinearAlgebra
using Random
using Optim
using Test

# In this example we want to learn a Hamiltonian term from observations using implicit differentiation
# and differentiable programming. The setting is an inverse problem:
# We assume a dataset of observations
# ```
#   forces = compute_forces(term(θ), positions) + noise
# ```
# for several different positions and fixed uknown true parameters θ.
# We now want to infer θ - for which we will use gradient-based optimization.
# More generally, such a setting allows *approximate* any unknown term by 
# choosing a sufficiently flexible parameterization, e.g. a neural network.
# To keep it simple in this example, we start with a single parameter.

function compute_force(θ::T, positions=[[0.2, 0, 0], [0.8, 0, 0]]) where {T}
    # solve the 1D Gross-Pitaevskii equation with ElementGaussian potential
    lattice = 10.0 .* [[1 0 0.]; [0 0 0]; [0 0 0]]
    gauss = ElementGaussian(1.0, 0.5)
    atoms = [gauss, gauss]
    n_electrons = 1
    terms = [Kinetic(), AtomicLocal(), LocalNonlinearity(ρ -> θ * ρ^2)]
    model = Model(Matrix{T}(lattice), atoms, positions;
                  n_electrons, terms, spin_polarization=:spinless)
    basis = PlaneWaveBasis(model; Ecut=500, kgrid=(1, 1, 1))
    ρ = zeros(Float64, basis.fft_size..., 1)
    is_converged = DFTK.ScfConvergenceDensity(1e-10)
    scfres = self_consistent_field(basis; ρ, is_converged,
                                    response=ResponseOptions(verbose=true))
    compute_forces_cart(scfres)
end

@testset "force derivatives w.r.t. LocalNonlinearity" begin
    θ = 1.0
    derivative_ε = let ε = 1e-5
       (compute_force(θ + ε) - compute_force(θ - ε)) / 2ε
    end
    derivative_fd = ForwardDiff.derivative(compute_force, θ)
    @test norm(derivative_ε - derivative_fd) < 1e-4
end

function sample_positions(seed)
    disp = 0.5 * rand(MersenneTwister(seed))
    positions = [[0.2, 0, 0], [0.8 + disp, 0, 0]]
    positions
end


θ_true = 1.0
n_data = 10
x = [sample_positions(seed) for seed in 1:n_data]
y = [compute_force(θ_true, positions) for positions in x]

sqnorm(x) = sum(abs2, x)
function loss(θ, x, y)
    sum(zip(x, y)) do (positions, forces_true)
        forces = compute_force(θ, positions)
        sum(sqnorm, forces - forces_true)
    end / length(x)
end

loss(1.0, x, y)
ForwardDiff.derivative(θ -> loss(θ, x, y), 1.0)
ForwardDiff.derivative(θ -> loss(θ, x, y), 2.0)

opt = optimize(θ -> loss(θ[1], x, y), [2.0], LBFGS(), autodiff= :forward)


# * Status: success

# * Candidate solution
#    Final objective value:     3.713626e-22

# * Found with
#    Algorithm:     L-BFGS

# * Convergence measures
#    |x - x'|               = 2.89e-06 ≰ 0.0e+00
#    |x - x'|/|x'|          = 2.89e-06 ≰ 0.0e+00
#    |f(x) - f(x')|         = 2.68e-14 ≰ 0.0e+00
#    |f(x) - f(x')|/|f(x')| = 7.21e+07 ≰ 0.0e+00
#    |g(x)|                 = 3.75e-13 ≤ 1.0e-08

# * Work counters
#    Seconds run:   9  (vs limit Inf)
#    Iterations:    5
#    f(x) calls:    16
#    ∇f(x) calls:   16

