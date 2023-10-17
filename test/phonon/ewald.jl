using Test
using DFTK
using LinearAlgebra
using ForwardDiff

include("../testcases.jl")
include("helpers.jl")

function ph_compute_reference_ewald(model_supercell)
    n_atoms = length(model_supercell.positions)
    n_dim = model_supercell.n_dim
    T = eltype(model_supercell.lattice)
    dynmat_ad = zeros(T, 3, n_atoms, 3, n_atoms)
    for τ in 1:n_atoms
        for γ in 1:n_dim
            displacement = zero.(model_supercell.positions)
            displacement[τ] = setindex(displacement[τ], one(T), γ)
            dynmat_ad[:, :, γ, τ] = -ForwardDiff.derivative(zero(T)) do ε
                cell_disp = (; model_supercell.lattice, model_supercell.atoms,
                             positions=ε*displacement .+ model_supercell.positions)
                model_disp = Model(convert(Model{eltype(ε)}, model_supercell); cell_disp...)
                forces = DFTK.energy_forces_ewald(model_disp).forces
                hcat(Array.(forces)...)
            end
        end
    end
    hessian_ad = DFTK.dynmat_to_cart(model_supercell, dynmat_ad)
    sort(compute_ω²(hessian_ad))
end

@testset "Ewald: comparison to ref testcase" begin
    tol = 1e-9
    terms = [Ewald()]
    model = Model(silicon.lattice, silicon.atoms, silicon.positions; terms)
    basis_bs = PlaneWaveBasis(model; Ecut=5)

    supercell_size = [2, 1, 3]
    phonon = (; supercell_size, generate_supercell_qpoints(; supercell_size).qpoints)

    ω_uc = []
    for q in phonon.qpoints
        hessian = DFTK.compute_dynmat_cart(basis_bs, nothing, nothing; q)
        push!(ω_uc, compute_ω²(hessian))
    end
    ω_uc = sort!(collect(Iterators.flatten(ω_uc)))

    ω_ref = [ -0.2442311083805831
              -0.24423110838058237
              -0.23442208238107232
              -0.23442208238107184
              -0.1322944535508822
              -0.13229445355088176
              -0.10658869539441493
              -0.10658869539441468
              -0.10658869539441346
              -0.10658869539441335
              -4.891274318712944e-16
              -3.773447798738169e-17
              1.659776058962626e-15
              0.09553958285993536
              0.18062696253387409
              0.18062696253387464
              0.4959725605665635
              0.4959725605665648
              0.49597256056656597
              0.5498259359834827
              0.5498259359834833
              0.6536453595829087
              0.6536453595829091
              0.6536453595829103
              0.6536453595829105
              0.6961890494198791
              0.6961890494198807
              0.7251587593311752
              0.7251587593311782
              0.9261195383192374
              0.9261195383192381
              1.2533843205271504
              1.2533843205271538
              1.7010950724885228
              1.7010950724885254
              1.752506588801463]
    @test norm(ω_uc - ω_ref) < tol
end

if !isdefined(Main, :FAST_TESTS) || !FAST_TESTS
@testset "Ewald: comparison to automatic differentiation" begin
    tol = 1e-9
    terms = [Ewald()]
    model = Model(silicon.lattice, silicon.atoms, silicon.positions; terms)
    basis_bs = PlaneWaveBasis(model; Ecut=5)

    supercell_size = supercell_size=generate_random_supercell()
    phonon = (; supercell_size, generate_supercell_qpoints(; supercell_size).qpoints)

    ω_uc = []
    for q in phonon.qpoints
        hessian = DFTK.compute_dynmat_cart(basis_bs, nothing, nothing; q)
        push!(ω_uc, compute_ω²(hessian))
    end
    ω_uc = sort!(collect(Iterators.flatten(ω_uc)))

    supercell = create_supercell(silicon.lattice, silicon.atoms, silicon.positions,
                                    phonon.supercell_size)
    model_supercell = Model(supercell.lattice, supercell.atoms, supercell.positions;
                            terms)
    basis_supercell_bs = PlaneWaveBasis(model_supercell; Ecut=5)
    hessian_supercell = DFTK.compute_dynmat_cart(basis_supercell_bs, nothing, nothing)
    ω_supercell = sort(compute_ω²(hessian_supercell))
    @test norm(ω_uc - ω_supercell) < tol

    ω_ad = ph_compute_reference_ewald(model_supercell)
    @test norm(ω_ad - ω_supercell) < tol
end
end
