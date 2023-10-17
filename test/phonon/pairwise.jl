using Test
using DFTK
using LinearAlgebra
using ForwardDiff

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
    positions = [ones(3)/8+rand(3)/20, -ones(3)/8]
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

    ω_ref = [ -0.2751958160657958
              -0.26041797812075534
              -0.2604179781207553
              -0.2547442698188065
              -0.24775236697358288
              -0.2477523669735825
              -0.1993297637363033
              -0.1993297637363032
              -0.18423374442602247
              -0.17929668011545924
              -0.17929668011545907
              -0.1665695828143445
              -0.08461624208414233
              -0.0846162420841423
              -0.07641521316620953
              -0.05869616863756916
              -0.058696168637569116
              -0.017708985408624065
              -0.017708985408624003
              -7.90344156095121e-18
               5.575140690944854e-18
               1.9385101894996632e-17
               0.0015658259492413475
               0.005179213798543904
               0.005179213798543916
               0.015882923455615604
               0.019165691337711125
               0.019232726297107625
               0.019232726297107646
               0.01962761876861809
               0.024023175651671887
               0.024023175651671887
               0.034234911940017605
               0.03423491194001761
               0.03929535653539282
               0.03929535653539288 ]
    @test norm(ω_uc - ω_ref) < tol
end

if !isdefined(Main, :FAST_TESTS) || !FAST_TESTS
@testset "Pairwise: comparison to automatic differentiation" begin
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
