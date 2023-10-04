using Test
using DFTK
using LinearAlgebra
using ForwardDiff

include("../testcases.jl")

function ph_compute_reference_ewald(model_supercell)
    n_atoms = length(model_supercell.positions)
    n_dim = model_supercell.n_dim
    T = eltype(silicon.lattice)
    dynmat_ad = zeros(T, 3, n_atoms, 3, n_atoms)
    for τ in 1:n_atoms
        for γ in 1:n_dim
            displacement = zero.(model_supercell.positions)
            displacement[τ] = setindex(displacement[τ], one(T), γ)
            dynmat_ad[:, :, γ, τ] = -ForwardDiff.derivative(zero(T)) do ε
                cell_disp = (; model_supercell.lattice, model_supercell.atoms,
                             positions=ε*displacement .+ model_supercell.positions)
                model_disp = Model(convert(Model{eltype(ε)}, model_supercell); cell_disp...)
                forces = DFTK.energy_forces_ewald(model_disp; compute_forces=true).forces
                hcat(Array.(forces)...)
            end
        end
    end
    hessian_ad = dynmat_to_cart((; model=model_supercell), dynmat_ad)
    sort(compute_ω²(hessian_ad))
end

@testset "Dynamical matrix for Ewald" begin
    tol = 1e-9
    model = Model(silicon.lattice, silicon.atoms, silicon.positions)

    ω_uc = []
    for q in phonon.qpoints
        hessian = dynmat_to_cart((; model), DFTK.dynmat_ewald(model; q))
        push!(ω_uc, compute_ω²(hessian))
    end
    ω_uc = sort!(collect(Iterators.flatten(ω_uc)))

    if REFERENCE_PH_CALC
        supercell = create_supercell(silicon.lattice, silicon.atoms, silicon.positions,
                                     phonon.supercell_size)
        model_supercell = Model(supercell.lattice, supercell.atoms, supercell.positions)
        hessian_supercell = dynmat_to_cart((; model=model_supercell), DFTK.dynmat_ewald(model_supercell))
        ω_supercell = sort(compute_ω²(hessian_supercell))
        @test norm(ω_uc - ω_supercell) < tol

        ω_ad = ph_compute_reference_ewald(model_supercell)
        @test norm(ω_ad - ω_supercell) < tol
    else
        ω_ref = [-0.2442311083805831
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
end
