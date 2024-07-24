# Helpers functions for tests.
@testmodule Phonon begin
using Test
using DFTK
using DFTK: normalize_kpoint_coordinate, _phonon_modes, dynmat_red_to_cart
using LinearAlgebra
using ForwardDiff
using FiniteDifferences

function generate_random_supercell(; max_length=6)
    n_max = min(max_length, 5)
    supercell_size = nothing
    while true
        supercell_size = rand(1:n_max, 3)
        prod(supercell_size) < max_length && break
    end
    supercell_size
end

function generate_supercell_qpoints(; supercell_size=generate_random_supercell())
    qpoints_list = Iterators.product([1:n_sc for n_sc in supercell_size]...)
    qpoints = map(qpoints_list) do n_sc
        normalize_kpoint_coordinate.([n_sc[i] / supercell_size[i] for i = 1:3])
    end |> vec

    (; supercell_size, qpoints)
end

function test_approx_frequencies(ω_uc, ω_ref; tol=1e-10)
    # Because three eigenvalues should be close to zero and the square root near
    # zero decrease machine accuracy, we expect at least ``3×2×2 - 3 = 9``
    # eigenvalues to have norm related to the accuracy of the SCF convergence
    # parameter and the rest to be larger.
    n_dim = 3
    n_atoms = length(ω_uc) ÷ 3

    @test count(abs.(ω_uc - ω_ref) .< sqrt(tol)) ≥ n_dim*n_atoms - n_dim
    @test count(sqrt(tol) .< abs.(ω_uc - ω_ref) .< tol) ≤ n_dim
end

function test_frequencies(model_tested, testcase; ω_ref=nothing, Ecut=7, kgrid=[2, 1, 3],
                          tol=1e-12, randomize=false, compute_ref=nothing)
    supercell_size = randomize ? generate_random_supercell() : kgrid
    qpoints = generate_supercell_qpoints(; supercell_size).qpoints
    scf_tol = tol
    χ0_tol  = scf_tol/10
    scf_kwargs = (; is_converged=ScfConvergenceDensity(scf_tol),
                  diagtolalg=AdaptiveDiagtol(; diagtol_max=scf_tol))

    model = model_tested(testcase.lattice, testcase.atoms, testcase.positions;
                         symmetries=false, testcase.temperature)
    nbandsalg = AdaptiveBands(model; occupation_threshold=1e-10)
    scf_kwargs = merge(scf_kwargs, (; nbandsalg))
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    scfres = self_consistent_field(basis; scf_kwargs...)

    ω_uc = sort!(reduce(vcat, map(qpoints) do q
        phonon_modes(scfres; q, tol=χ0_tol).frequencies
    end))

    !isnothing(ω_ref) && return test_approx_frequencies(ω_uc, ω_ref; tol=10scf_tol)

    supercell = create_supercell(testcase.lattice, testcase.atoms, testcase.positions,
                                 supercell_size)
    model_supercell = model_tested(supercell.lattice, supercell.atoms, supercell.positions;
                               symmetries=false, testcase.temperature)
    nbandsalg = AdaptiveBands(model_supercell; occupation_threshold=1e-10)
    scf_kwargs = merge(scf_kwargs, (; nbandsalg))
    basis_supercell = PlaneWaveBasis(model_supercell; Ecut, kgrid=[1, 1, 1])
    scfres_supercell = self_consistent_field(basis_supercell; scf_kwargs...)

    ω_sc = sort(phonon_modes(scfres_supercell; tol=χ0_tol).frequencies)
    test_approx_frequencies(ω_uc, ω_sc; tol=10scf_tol)

    isnothing(compute_ref) && return

    dynamical_matrix_ref = compute_dynmat_ref(scfres_supercell.basis, model_tested; Ecut,
                                              kgrid=[1, 1, 1], scf_tol, method=compute_ref)
    ω_ref = sort(_phonon_modes(basis_supercell, dynamical_matrix_ref).frequencies)

    test_approx_frequencies(ω_uc, ω_ref; tol=10scf_tol)
end

# Reference results using finite differences or automatic differentiation.
# This should be run by hand to obtain the reference values of the quick computations of the
# tests, as they are too slow for CI runs.
function compute_dynmat_ref(basis, model_tested; Ecut=5, kgrid=[1,1,1], scf_tol, method=:ad)
    # TODO: Cannot use symmetries: https://github.com/JuliaMolSim/DFTK.jl/issues/817
    @assert isone(only(basis.model.symmetries))
    @assert method ∈ [:ad, :fd]

    model = basis.model
    n_atoms = length(model.positions)
    n_dim = model.n_dim
    T = eltype(model.lattice)
    dynmat = zeros(T, 3, n_atoms, 3, n_atoms)
    scf_kwargs = (; is_converged=ScfConvergenceDensity(scf_tol),
                  diagtolalg=AdaptiveDiagtol(; diagtol_max=scf_tol))

    diff_fn = method == :ad ? ForwardDiff.derivative : FiniteDifferences.central_fdm(5, 1)
    for s = 1:n_atoms, α = 1:n_dim
        displacement = zero.(model.positions)
        displacement[s] = DFTK.setindex(displacement[s], one(T), α)
        dynmat[:, :, α, s] = -diff_fn(zero(T)) do ε
            lattice = convert(Matrix{eltype(ε)}, model.lattice)
            positions = ε*displacement .+ model.positions
            model_disp = model_tested(lattice, model.atoms, positions; symmetries=false,
                                      model.temperature)
            # TODO: Would be cleaner with PR #675.
            basis_disp = PlaneWaveBasis(model_disp; Ecut, kgrid)
            nbandsalg = AdaptiveBands(model_disp; occupation_threshold=1e-10)
            scf_kwargs = merge(scf_kwargs, (; nbandsalg))
            scfres_disp = self_consistent_field(basis_disp; scf_kwargs...)
            forces = compute_forces(scfres_disp)
            stack(forces)
        end
    end
    dynmat_red_to_cart(model, dynmat)
end
end
