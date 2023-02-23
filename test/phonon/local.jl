using Test
using DFTK
using LinearAlgebra
using ForwardDiff
using Random

@testset "Phonons: Local term" begin

include("../testcases.jl")

function model_pw(lattice::AbstractMatrix,
                      atoms::Vector{<:DFTK.Element},
                      positions::Vector{<:AbstractVector};
                      extra_terms=[], kinetic_blowup=BlowupIdentity(), kwargs...)
    @assert !(:terms in keys(kwargs))
    terms = [Kinetic(; blowup=kinetic_blowup),
             AtomicLocal(),
             extra_terms...]
    Model(lattice, atoms, positions; model_name="atomic", terms, kwargs...)
end

function compute_ω_1d(; n_scell=1, q=0.0, Ecut=20, kgrid=[1, 1, 1], εF=nothing,
                      temperature=0, ad=false, kwargs...)
    a= 5.0
    scf_kwargs = (; callback=identity, tol=1e-9)

    # We create a create a supercell with two atoms to catch edge cases.
    lattice = a * 2 .* [1.0 0 0; 0 0 0; 0 0 0]
    hpot = 3.0
    spread = 0.5
    unit_cell = (; lattice, positions=[zeros(3), [0.5, 0, 0]],
                 atoms=[ElementGaussian(hpot, spread; symbol=:X),
                        ElementGaussian(hpot, spread; symbol=:X)])

    q = Vec3([q, 0, 0])
    supercell_size = [n_scell, 1, 1]
    cell = create_supercell(unit_cell.lattice, unit_cell.atoms, unit_cell.positions,
                            supercell_size)
    # We use a simple Lennard-Jones potential.
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)
    params = Dict((:X, :X) => (; ε=1, σ=a / 2^(1/6)))
    extra_terms = [PairwisePotential(V, params, max_radius=1e3)]
    function create_local_model(ucell)
        n_electrons = isnothing(εF) ? length(ucell.positions) : nothing
        model_pw(ucell.lattice, ucell.atoms, ucell.positions; symmetries=false,
                 temperature, εF, disable_electrostatics_check=true, n_electrons,
                 spin_polarization=:spinless, extra_terms)
    end
    model = create_local_model(cell)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    nbandsalg = isnothing(εF) ? AdaptiveBands(model) : FixedBands(; n_bands_converge=16)
    scf_kwargs = merge(scf_kwargs, (; nbandsalg, eigensolver=diag_full))
    if ad
        dynamical_matrix = compute_dynmat_ad(basis; scf_kwargs...)
        ω = phonon_eigenvalues(basis, dynamical_matrix)
    else
        scfres = self_consistent_field(basis; scf_kwargs...)
        #verbose_phonons(scfres)
        δρs, δψs, δoccupations = DFTK.compute_δρ(scfres; q)

        dynamical_matrix = compute_dynmat(scfres; δρs, δψs, δoccupations, q)
        ω = phonon_eigenvalues(basis, dynamical_matrix)
    end
end

@testset "ElementGaussian" begin
    Random.seed!()
    # Random number of k-points for unit cell computations.
    k_rand = rand(1:3)
    # Random number of supercells.
    n_sc = rand(2:4)

    # Corresponding parameters.
    n_uc = 1
    kgrid_uc = [n_sc*k_rand, 1, 1]
    kgrid_sc = [k_rand, 1, 1]

    qpoints_big = [i / n_sc for i in 1:n_sc]
    qpoints = map(qpoints_big) do q
                     q = mod(q, 1)               # coordinate in [0, 1)³
                     q ≥ 0.5 - 1e-3 ? q - 1 : q  # coordinate in [-½, ½)³
               end
    temperature = 0.1
    ω_uc = []
    for q in qpoints
        ω_q = compute_ω_1d(; n_scell=n_uc, q, kgrid=kgrid_uc, temperature)
        ω_uc = hcat(ω_uc..., ω_q)
    end
    ωr = compute_ω_1d(; n_scell=n_sc, q=0.0, kgrid=kgrid_sc, temperature, ad=true)
    ωs = compute_ω_1d(; n_scell=n_sc, q=0.0, kgrid=kgrid_sc, temperature)

    @test count(abs.(ωr - ωs) .< 5e-3) ≤ 2n_sc
    @test count(abs.(ωr - ωs) .< 1e-6) ≥ 2n_sc-1
    @test count(abs.(sort(ω_uc; dims=2)-sort(ωs; dims=2)) .< 5e-3) ≤ 2n_sc
    @test count(abs.(sort(ω_uc; dims=2)-sort(ωs; dims=2)) .< 1e-6) ≥ 2n_sc-1
end

# Model with no exchange-correlation and only a local potential.
function model_loc(lattice::AbstractMatrix,
                   atoms::Vector{<:DFTK.Element},
                   positions::Vector{<:AbstractVector};
                   extra_terms=[], kinetic_blowup=BlowupIdentity(), kwargs...)
    @assert !(:terms in keys(kwargs))
    terms = [Kinetic(; blowup=kinetic_blowup),
             AtomicLocal(),
             Ewald(),
             PspCorrection(),
             Hartree(),
             extra_terms...]
    if :temperature in keys(kwargs) && kwargs[:temperature] != 0
        terms = [terms..., Entropy()]
    end
    Model(lattice, atoms, positions; model_name="atomic", terms, kwargs...)
end

@testset "LDA supercell consistency" begin
    for (fast, cell, temperature) in ((false, aluminium_primitive, nothing),
                                      (true, aluminium_primitive, 0.1),
                                      (true, magnesium, nothing),
                                      (true, magnesium, 0.1),
                                      )
        @testset "1" begin
            Random.seed!()
            Ecut  = 7
            kgrid = fast ? [2, 1, 1] : [2, 3, 2]
            if !isnothing(temperature)
                cell = merge(cell, (; temperature))
            end
            scf_tol = 1e-12
            is_converged = DFTK.ScfConvergenceDensity(scf_tol)
            determine_diagtol = DFTK.ScfDiagtol(diagtol_max=scf_tol)

            scf_kwargs = (; is_converged, determine_diagtol)

            if fast
                supercell_size = [2, 1, 1]
                kgrid_supercell = [1, 1, 1]
                qpoints = [zeros(3), [1/2, 0, 0]]
            else
                rand_dir = rand(1:3)
                supercell_size = ones(3)
                supercell_size[rand_dir] = kgrid[rand_dir]
                kgrid_supercell = copy(kgrid)
                kgrid_supercell[rand_dir] = 1
                qpoints_big = [i*ones(3) ./ supercell_size for i in 1:supercell_size[rand_dir]]
                qpoints = [map(qpt) do q
                                 q = mod(q, 1)               # coordinate in [0, 1)³
                                 q ≥ 0.5 - 1e-3 ? q - 1 : q  # coordinate in [-½, ½)³
                           end
                           for qpt in qpoints_big]
            end

            ω_uc = []
            model = model_loc(cell.lattice, cell.atoms, cell.positions; symmetries=false,
                              cell.temperature)
            nbandsalg = AdaptiveBands(model; occupation_threshold=1e-10)
            scf_kwargs = merge(scf_kwargs, (; nbandsalg))
            basis = PlaneWaveBasis(model; Ecut, kgrid)
            scfres = self_consistent_field(basis; scf_kwargs...)
            for q in qpoints
                δρs, δψs, δoccupations = DFTK.compute_δρ(scfres; q)
                dynamical_matrix = compute_dynmat(scfres; δρs, δψs, δoccupations, q)
                ω_q = phonon_eigenvalues(basis, dynamical_matrix)
                ω_uc = hcat(ω_uc..., ω_q)
            end

            supercell = create_supercell(cell.lattice, cell.atoms, cell.positions,
                                         supercell_size)
            model_supercell = model_loc(supercell.lattice, supercell.atoms,
                                        supercell.positions; symmetries=false,
                                        cell.temperature)
            nbandsalg = AdaptiveBands(model_supercell; occupation_threshold=1e-10)
            scf_kwargs = merge(scf_kwargs, (; nbandsalg))
            basis_supercell = PlaneWaveBasis(model_supercell; Ecut, kgrid=kgrid_supercell)

            scfres = self_consistent_field(basis_supercell; scf_kwargs...)
            δρs, δψs, δoccupations = DFTK.compute_δρ(scfres)
            dynamical_matrix = compute_dynmat(scfres; δρs, δψs, δoccupations)
            ω_sc = sort(phonon_eigenvalues(basis_supercell, dynamical_matrix); dims=2)

            dynamical_matrix_ad = compute_dynmat_ad(basis_supercell; scf_kwargs...)
            ω_ad = sort(phonon_eigenvalues(basis_supercell, dynamical_matrix_ad); dims=2)

            # Because three eigenvalues should be close to zero and the square root near
            # zero decrease machine accuracy, we expect at least ``3×2×2 - 3 = 9``
            # eigenvalues to have norm related to the accuracy of the SCF convergence
            # parameter and the rest to be larger.
            n_atoms = length(basis_supercell.model.positions)
            n_dim = basis_supercell.model.n_dim
            for ω_ref in (ω_sc, ω_ad)
                @test count(abs.(sort(ω_uc; dims=2) - ω_ref) .< sqrt(scf_tol)) ≥ n_dim*n_atoms - n_dim
                @test count(sqrt(scf_tol) .< abs.(sort(ω_uc; dims=2) - ω_ref) .< scf_tol^(1/4)) ≤ n_dim
            end
        end
    end
end

end
