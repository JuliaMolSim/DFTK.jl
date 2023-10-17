using Test
using DFTK
using DFTK: setindex
using LinearAlgebra
using ForwardDiff
using Random

include("helpers.jl")

function ph_compute_reference_pairwise(model_supercell)
    n_atoms = length(model_supercell.positions)
    n_dim = model_supercell.n_dim
    T = eltype(model_supercell.lattice)
    dynmat_ad = zeros(T, 3, n_atoms, 3, n_atoms)
    term = only(model_supercell.term_types)
    for τ in 1:n_atoms
        for γ in 1:n_dim
            displacement = zero.(model_supercell.positions)
            displacement[τ] = setindex(displacement[τ], one(T), γ)
            dynmat_ad[:, :, γ, τ] = -ForwardDiff.derivative(zero(T)) do ε
                cell_disp = (; model_supercell.lattice, model_supercell.atoms,
                             positions=ε*displacement .+ model_supercell.positions)
                model_disp = Model(convert(Model{eltype(ε)}, model_supercell); cell_disp...)
                forces = DFTK.energy_forces_pairwise(model_disp, term.V, term.params;
                                                     term.max_radius).forces
                hcat(Array.(forces)...)
            end
        end
    end
    hessian_ad = DFTK.dynmat_to_cart(model_supercell, dynmat_ad)
    sort(compute_ω²(hessian_ad))
end

@testset "Pairwise: comparison to ref testcase" begin
    tol = 1e-4  # low because of the small radius that we use to speed-up computations
    a = 5.131570667152971
    lattice = a .* [0 1 1; 1 0 1; 1 1 0]
    # perturb positions away from equilibrium to get nonzero force
    atoms     = [ElementCoulomb(:Li), ElementCoulomb(:H)]
    positions = [ones(3)/8, -ones(3)/8]
    symbols   = [:Li, :H]
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)
    params = Dict((:Li, :H ) => (; ε=1, σ=2),
                  ( :H, :H ) => (; ε=1, σ=2),
                  (:Li, :Li) => (; ε=1, σ=2))
    terms = [PairwisePotential(V, params; max_radius=10)]
    model = Model(lattice, atoms, positions; terms)
    basis_bs = PlaneWaveBasis(model; Ecut=5)

    supercell_size = [2, 1, 3]
    phonon = (; supercell_size, generate_supercell_qpoints(; supercell_size).qpoints)

    ω_uc = []
    for q in phonon.qpoints
        hessian = DFTK.compute_dynmat_cart(basis_bs, nothing, nothing; q)
        push!(ω_uc, compute_ω²(hessian))
    end
    ω_uc = sort!(collect(Iterators.flatten(ω_uc)))

    ω_ref = [ -0.17524941818128295
              -0.17524941818128292
              -0.16832870864925772
              -0.1683287086492577
              -0.15195539471448133
              -0.15193439163371011
              -0.15024477964426225
              -0.1474510503442476
              -0.1474510503442476
              -0.1474293810073767
              -0.1474293810073767
              -0.14475429352434355
              -0.1447542935243435
              -0.13232858616165102
              -0.13230326232686676
              -0.13230326232686657
              -0.09377914380407501
              -0.09377914380407495
              -0.05435449538374675
              -0.05435449538374669
              -0.003915427003557917
              -0.003915427003557904
              -0.0033812394777427185
              -1.7065097811479572e-17
              -3.9611338270885374e-18
               1.7013223995880296e-17
               0.013331409088687666
               0.013331409088687688
               0.013352636670909857
               0.013352636670909858
               0.017234067155574892
               0.017254437310419694
               0.030332222284351517
               0.03033222228435154
               0.03334700396064381
               0.03334700396064386 ]
    @show norm(ω_ref)
    @test norm(ω_uc - ω_ref) < tol
end

if !isdefined(Main, :FAST_TESTS) || !FAST_TESTS
@testset "Pairwise: comparison to automatic differentiation" begin
    Random.seed!()
    tol = 1e-4  # low because of the small radius that we use to speed-up computations
    a = 5.131570667152971
    lattice = a .* [0 1 1; 1 0 1; 1 1 0]
    # perturb positions away from equilibrium to get nonzero force
    atoms     = [ElementCoulomb(:Li), ElementCoulomb(:H)]
    positions = [ones(3)/8+rand(3)/20, -ones(3)/8]
    symbols   = [:Li, :H]
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)
    params = Dict((:Li, :H ) => (; ε=1, σ=2),
                  ( :H, :H ) => (; ε=1, σ=2),
                  (:Li, :Li) => (; ε=1, σ=2))
    terms = [PairwisePotential(V, params; max_radius=10)]
    model = Model(lattice, atoms, positions; terms)
    basis_bs = PlaneWaveBasis(model; Ecut=5)

    supercell_size = supercell_size=generate_random_supercell()
    phonon = (; supercell_size, generate_supercell_qpoints(; supercell_size).qpoints)

    ω_uc = []
    for q in phonon.qpoints
        hessian = DFTK.compute_dynmat_cart(basis_bs, nothing, nothing; q)
        push!(ω_uc, compute_ω²(hessian))
    end
    ω_uc = sort!(collect(Iterators.flatten(ω_uc)))

    supercell = create_supercell(lattice, atoms, positions, phonon.supercell_size)
    model_supercell = Model(supercell.lattice, supercell.atoms, supercell.positions;
                            terms)
    basis_supercell_bs = PlaneWaveBasis(model_supercell; Ecut=5)
    hessian_supercell = DFTK.compute_dynmat_cart(basis_supercell_bs, nothing, nothing)
    ω_supercell = sort(compute_ω²(hessian_supercell))
    @test norm(ω_uc - ω_supercell) < tol

    ω_ad = ph_compute_reference_pairwise(model_supercell)
    @test norm(ω_ad - ω_supercell) < tol
end
end
