# Helpers functions for tests.
@testsetup module Phonon
using Test
using DFTK
using DFTK: compute_dynmat_cart, setindex, dynmat_red_to_cart, normalize_kpoint_coordinate
using LinearAlgebra
using ForwardDiff

# We do not take the square root to compare eigenvalues with machine precision.
function compute_squared_frequencies(matrix)
    n, m = size(matrix, 1), size(matrix, 2)
    Ω = eigvals(reshape(matrix, n*m, n*m))
    real(Ω)
end

# Reference against automatic differentiation.
function ph_compute_reference(basis_supercell)
    model_supercell = basis_supercell.model
    n_atoms = length(model_supercell.positions)
    n_dim = model_supercell.n_dim
    T = eltype(model_supercell.lattice)
    dynmat_ad = zeros(T, 3, n_atoms, 3, n_atoms)
    for τ = 1:n_atoms, γ = 1:n_dim
        displacement = zero.(model_supercell.positions)
        displacement[τ] = setindex(displacement[τ], one(T), γ)
        dynmat_ad[:, :, γ, τ] = -ForwardDiff.derivative(zero(T)) do ε
            lattice = convert(Matrix{eltype(ε)}, model_supercell.lattice)
            positions = ε*displacement .+ model_supercell.positions
            model_disp = Model(convert(Model{eltype(ε)}, model_supercell); lattice, positions)
            # TODO: Would be cleaner with PR #675.
            basis_disp_bs = PlaneWaveBasis(model_disp; Ecut=5)
            forces = compute_forces(basis_disp_bs, nothing, nothing)
            reduce(hcat, forces)
        end
    end
    hessian_ad = dynmat_red_to_cart(model_supercell, dynmat_ad)
    sort(compute_squared_frequencies(hessian_ad))
end

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

# Test against a reference array.
function test_frequencies(testcase, terms, ω_ref; tol=1e-9)
    model = Model(testcase.lattice, testcase.atoms, testcase.positions; terms)
    basis_bs = PlaneWaveBasis(model; Ecut=5)

    supercell_size = [2, 1, 3]
    phonon = (; supercell_size, generate_supercell_qpoints(; supercell_size).qpoints)

    ω_uc = []
    for q in phonon.qpoints
        hessian = compute_dynmat_cart(basis_bs, nothing, nothing; q)
        push!(ω_uc, compute_squared_frequencies(hessian))
    end
    ω_uc = sort!(collect(Iterators.flatten(ω_uc)))

    @test norm(ω_uc - ω_ref) < tol
end

# Random test. Slow but more robust than against some reference.
# TODO: Will need rework for local term in future PR.
function test_rand_frequencies(testcase, terms; tol=1e-9)
    model = Model(testcase.lattice, testcase.atoms, testcase.positions; terms)
    basis_bs = PlaneWaveBasis(model; Ecut=5)

    supercell_size = supercell_size=generate_random_supercell()
    phonon = (; supercell_size, generate_supercell_qpoints(; supercell_size).qpoints)

    ω_uc = []
    for q in phonon.qpoints
        hessian = compute_dynmat_cart(basis_bs, nothing, nothing; q)
        push!(ω_uc, compute_squared_frequencies(hessian))
    end
    ω_uc = sort!(collect(Iterators.flatten(ω_uc)))

    supercell = create_supercell(testcase.lattice, testcase.atoms, testcase.positions,
                                 phonon.supercell_size)
    model_supercell = Model(supercell.lattice, supercell.atoms, supercell.positions; terms)
    basis_supercell_bs = PlaneWaveBasis(model_supercell; Ecut=5)
    hessian_supercell = compute_dynmat_cart(basis_supercell_bs, nothing, nothing)
    ω_supercell = sort(compute_squared_frequencies(hessian_supercell))
    @test norm(ω_uc - ω_supercell) < tol

    ω_ad = ph_compute_reference(basis_supercell_bs)

    @test norm(ω_ad - ω_supercell) < tol
end
end
