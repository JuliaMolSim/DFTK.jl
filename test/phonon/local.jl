using Test
using DFTK
using LinearAlgebra
using StaticArrays
using ForwardDiff
using FiniteDifferences

@testset "Phonons: Local term" begin

include("../testcases.jl")

# TODO: To speed-up computations on the CI, be smarter later.
fast = true

# Useful to find a suitable Fermi level.
function verbose_phonons(scfres)
    @debug scfres.basis.fft_size
    @debug "eigs" [eig_ik[scfres.occupation[ik] .> scfres.occupation_threshold]
                   for (ik, eig_ik) in enumerate(scfres.eigenvalues)]
    @debug "occupations" [sum(occ_ik .> scfres.occupation_threshold)
                          for occ_ik in scfres.occupation]
end


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
    Ecut  = 5
    #kgrid = fast ? [4, 3, 2] : [2, 3, 2]
    kgrid = fast ? [2, 1, 1] : [2, 3, 2]
    cell = aluminium_primitive
    #cell = magnesium
    #cell = aluminium
    cell = merge(cell, (; temperature=0.01))
    scf_tol = 1e-12
    is_converged = DFTK.ScfConvergenceDensity(scf_tol)
    determine_diagtol = DFTK.ScfDiagtol(diagtol_max=scf_tol)

    scf_kwargs = (; is_converged, determine_diagtol)
    #scf_kwargs = (; callback=identity, is_converged, determine_diagtol)

    if fast
        supercell_size = [2, 1, 1]
        #kgrid_supercell = [2, 3, 2]
        kgrid_supercell = [1, 1, 1]
        qpoints = [zeros(3), 1 ./ supercell_size]
    else
        rand_dir = rand(1:3)
        supercell_size = ones(3)
        supercell_size[rand_dir] = kgrid[rand_dir]
        kgrid_supercell = copy(kgrid)
        kgrid_supercell[rand_dir] = 1
        qpoints = [zeros(3), 1 ./ supercell_size]
    end
    supercell = create_supercell(cell.lattice, cell.atoms, cell.positions, supercell_size)
    model_supercell = model_loc(supercell.lattice, supercell.atoms, supercell.positions; symmetries=false, cell.temperature)
    nbandsalg = AdaptiveBands(model_supercell; occupation_threshold=1e-10)
    scf_kwargs = merge(scf_kwargs, (; nbandsalg))
    basis_supercell = PlaneWaveBasis(model_supercell; Ecut, kgrid=kgrid_supercell)

    scfres = self_consistent_field(basis_supercell; scf_kwargs...)
    δρs, δψs, δoccupations = DFTK.compute_δρ(scfres)
    dynamical_matrix = compute_dynmat(scfres; δρs, δψs, δoccupations)
    ω_sc = sort(phonon_eigenvalues(basis_supercell, dynamical_matrix); dims=2)

    dynamical_matrix_ad = compute_dynmat_ad(basis_supercell; scf_kwargs...)
    ω_ad = sort(phonon_eigenvalues(basis_supercell, dynamical_matrix_ad); dims=2)

    ω_uc = []
    model = model_loc(cell.lattice, cell.atoms, cell.positions; symmetries=false, cell.temperature)
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

    # Because three eigenvalues should be close to zero and the square root near zero
    # decrease machine accuracy, we expect at least ``3×2×2 - 3 = 9`` eigenvalues to have
    # norm related to the accuracy of the SCF convergence parameter and the rest to be
    # larger.
    n_atoms = length(basis.model.positions)
    n_dim = basis.model.n_dim
    for ω_ref in (ω_sc, ω_ad)
        @test count(abs.(sort(ω_uc; dims=2) - ω_ref) .< sqrt(scf_tol)) ≥ n_dim*n_atoms - n_dim
        @test count(sqrt(scf_tol) .< abs.(sort(ω_uc; dims=2) - ω_ref) .< scf_tol^(1/4)) ≤ n_dim
    end
end

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

function ωo_dbg(; n_scell=1, q=0.0, Ecut=20, kgrid=[1, 1, 1], k=1, dir=1, εF=nothing,
                temperature=0, nonline=false, n_at=2,
                fft_size=nothing, ad=false, kwargs...)
    #scf_callback = (; callback=identity, tol=1e-9)
    scf_callback = (; tol=1e-9)
    a= 5.0
    lattice = a * n_at .* [1.0 0 0; 0 0 0; 0 0 0]
    hpot = 3.0
    spread = 0.5
    if n_at == 1
        unit_cell = (; lattice, positions=[zeros(3)], atoms=[ElementGaussian(hpot, spread; symbol=:X)])
    else
        unit_cell = (; lattice, positions=[zeros(3), [0.5, 0, 0]],
                     atoms=[ElementGaussian(hpot, spread; symbol=:X),
                            ElementGaussian(hpot, spread; symbol=:X)])
    end

    supercell_size = [1, 1, 1]
    q1 = q
    q = [0.0, 0, 0]
    kgrid[dir] = kgrid[1] * k
    supercell_size[dir] = n_scell
    q[dir] = q1
    cell = create_supercell(unit_cell.lattice, unit_cell.atoms, unit_cell.positions, supercell_size)
    q = Vec3(q)
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)
    params = Dict((:X, :X) => (; ε=1, σ=a / 2^(1/6)))
    PW = PairwisePotential(V, params, max_radius=1e3)
    if nonline
        extra_terms = [LocalNonlinearity(ρ -> 10 * ρ^2), PW]
    else
       extra_terms = [PW]
    end
    function create_local_model(ucell)
        n_electrons = isnothing(εF) ? length(ucell.positions) : nothing
        model_pw(ucell.lattice, ucell.atoms, ucell.positions; symmetries=false,
                 temperature, εF, disable_electrostatics_check=true, n_electrons,
                 spin_polarization=:spinless,
                 extra_terms)
    end
    model = create_local_model(cell)
    basis = PlaneWaveBasis(model; Ecut, kgrid, fft_size)
    nbandsalg = isnothing(εF) ? AdaptiveBands(model) : FixedBands(; n_bands_converge=8)
    if ad
        T = eltype(basis)
        n_atoms = length(model.positions)
        n_dim = model.n_dim
        d2_term = zeros(eltype(basis), (n_dim, n_atoms, n_dim, n_atoms))
        for τ in 1:n_atoms
            for γ in 1:n_dim
                displacement = zero.(model.positions)
                displacement[τ] = StaticArrays.setindex(displacement[τ], one(T), γ)
                d2 = -ForwardDiff.derivative(zero(T)) do ε
                    cell_disp = (; lattice=eltype(ε).(cell.lattice),
                                 cell.atoms,
                                 positions=ε*displacement .+ cell.positions)
                    model_disp = create_local_model(cell_disp)
                    basis_disp = PlaneWaveBasis(model_disp; Ecut, kgrid, fft_size)
                    scfres = self_consistent_field(basis_disp; eigensolver=diag_full, scf_callback..., nbandsalg)
                    forces = compute_forces(scfres)
                    forces
                end
                d2_term[:, :, γ, τ] = hcat(d2...)[1:n_dim, :]
            end
        end
        dynamical_matrix = reshape(d2_term, n_dim*n_atoms, n_dim*n_atoms)
        ω = phonon_eigenvalues(basis, dynamical_matrix)
        (; ω, dynamical_matrix)
    else
        scfres = self_consistent_field(basis; eigensolver=diag_full, scf_callback..., nbandsalg)
        verbose_phonons(scfres)
        δρs, δψs, δoccupations = DFTK.compute_δρ(scfres; q=q)

        dynamical_matrix = compute_dynmat(scfres; δρs, δψs, δoccupations, q)
        ω = phonon_eigenvalues(basis, dynamical_matrix)
        (; ω, dynamical_matrix)
    end
end

@testset "ElementGaussian n_at = 1" begin
    display("εF no temp")
    ω0 = ωo_dbg(; n_scell=1, q=0, kgrid=[2,1,1], n_at=1, Ecut=40, εF=-1.22, temperature=0.0).ω
    ω½ = ωo_dbg(; n_scell=1, q=0.5, kgrid=[2,1,1], n_at=1, Ecut=40, εF=-1.22, temperature=0.0).ω
    ωr = ωo_dbg(; n_scell=2, q=0, kgrid=[1,1,1], n_at=1, Ecut=40, εF=-1.22, temperature=0.0, ad=true).ω
    ωs = ωo_dbg(; n_scell=2, q=0, kgrid=[1,1,1], n_at=1, Ecut=40, εF=-1.22, temperature=0.0).ω
    ωu = hcat(ω0, ω½)
    @test count(abs.(ωr - ωs) .< 5e-3) == 2
    @test count(abs.(ωr - ωs) .< 1e-6) ≥ 1
    @test count(abs.(sort(ωu; dims=2))-sort(ωs; dims=2) .< 5e-3) == 2
    @test count(abs.(sort(ωu; dims=2))-sort(ωs; dims=2) .< 1e-6) ≥ 1

    display("no εF no temp")
    ω0 = ωo_dbg(; n_scell=1, q=0.5, kgrid=[2,1,1], n_at=1, Ecut=40, temperature=0.0).ω
    ω½ = ωo_dbg(; n_scell=1, q=0, kgrid=[2,1,1], n_at=1, Ecut=40, temperature=0.0).ω
    ωr = ωo_dbg(; n_scell=2, q=0, kgrid=[1,1,1], n_at=1, Ecut=40, temperature=0.0, ad=true).ω
    ωs = ωo_dbg(; n_scell=2, q=0, kgrid=[1,1,1], n_at=1, Ecut=40, temperature=0.0).ω
    ωu = hcat(ω0, ω½)
    @test count(abs.(ωr - ωs) .< 5e-3) == 2
    @test count(abs.(ωr - ωs) .< 1e-6) ≥ 1
    @test count(abs.(sort(ωu; dims=2)-sort(ωs; dims=2)) .< 5e-3) == 2
    @test count(abs.(sort(ωu; dims=2)-sort(ωs; dims=2)) .< 1e-6) ≥ 1

    display("no εF temp")
    ω0 = ωo_dbg(; n_scell=1, q=0.5, kgrid=[2,1,1], n_at=1, Ecut=40, temperature=0.1).ω
    ω½ = ωo_dbg(; n_scell=1, q=0, kgrid=[2,1,1], n_at=1, Ecut=40, temperature=0.1).ω
    ωr = ωo_dbg(; n_scell=2, q=0, kgrid=[1,1,1], n_at=1, Ecut=40, temperature=0.1, ad=true).ω
    ωs = ωo_dbg(; n_scell=2, q=0, kgrid=[1,1,1], n_at=1, Ecut=40, temperature=0.1).ω
    ωu = hcat(ω0, ω½)
    @test count(abs.(ωr - ωs) .< 5e-3) == 2
    @test count(abs.(ωr - ωs) .< 1e-6) ≥ 1
    @test count(abs.(sort(ωu; dims=2)-sort(ωs; dims=2)) .< 5e-3) == 2
    @test count(abs.(sort(ωu; dims=2)-sort(ωs; dims=2)) .< 1e-6) ≥ 1
end

@testset "ElementGaussian" begin
    display("εF no temp")
    ω0 = ωo_dbg(; n_scell=1, q=0, kgrid=[2,1,1], Ecut=20, εF=-1.22, temperature=0.0).ω
    ω½ = ωo_dbg(; n_scell=1, q=0.5, kgrid=[2,1,1], Ecut=20, εF=-1.22, temperature=0.0).ω
    ωr = ωo_dbg(; n_scell=2, q=0, kgrid=[1,1,1], Ecut=20, εF=-1.22, temperature=0.0, ad=true).ω
    ωs = ωo_dbg(; n_scell=2, q=0, kgrid=[1,1,1], Ecut=20, εF=-1.22, temperature=0.0).ω
    ωu = hcat(ω0, ω½)
    @test count(abs.(ωr - ωs) .< 5e-3) == 4
    @test count(abs.(ωr - ωs) .< 1e-6) ≥ 3
    @test count(abs.(sort(ωu; dims=2)-sort(ωs; dims=2)) .< 5e-3) == 4
    @test count(abs.(sort(ωu; dims=2)-sort(ωs; dims=2)) .< 1e-6) ≥ 3

    display("no εF no temp")
    ω0 = ωo_dbg(; n_scell=1, q=0.5, kgrid=[2,1,1], Ecut=20, temperature=0.0).ω
    ω½ = ωo_dbg(; n_scell=1, q=0, kgrid=[2,1,1], Ecut=20, temperature=0.0).ω
    ωr = ωo_dbg(; n_scell=2, q=0, kgrid=[1,1,1], Ecut=20, temperature=0.0, ad=true).ω
    ωs = ωo_dbg(; n_scell=2, q=0, kgrid=[1,1,1], Ecut=20, temperature=0.0).ω
    ωu = hcat(ω0, ω½)
    @test count(abs.(ωr - ωs) .< 5e-3) == 4
    @test count(abs.(ωr - ωs) .< 1e-6) ≥ 3
    @test count(abs.(sort(ωu; dims=2)-sort(ωs; dims=2)) .< 5e-3) == 4
    @test count(abs.(sort(ωu; dims=2)-sort(ωs; dims=2)) .< 1e-6) ≥ 3

    display("no εF temp")
    ω0 = ωo_dbg(; n_scell=1, q=0.5, kgrid=[2,1,1], Ecut=20, temperature=0.1).ω
    ω½ = ωo_dbg(; n_scell=1, q=0, kgrid=[2,1,1], Ecut=20, temperature=0.1).ω
    ωr = ωo_dbg(; n_scell=2, q=0, kgrid=[1,1,1], Ecut=20, temperature=0.1, ad=true).ω
    ωs = ωo_dbg(; n_scell=2, q=0, kgrid=[1,1,1], Ecut=20, temperature=0.1).ω
    ωu = hcat(ω0, ω½)
    @test count(abs.(ωr - ωs) .< 5e-3) == 4
    @test count(abs.(ωr - ωs) .< 1e-6) ≥ 3
    @test count(abs.(sort(ωu; dims=2)-sort(ωs; dims=2)) .< 5e-3) == 4
    @test count(abs.(sort(ωu; dims=2)-sort(ωs; dims=2)) .< 1e-6) ≥ 3
end

# Humongous test
#=
@testset "LDA supercell consistency" begin
    # We change a bit the default lattice from the silicon example to be able not to use
    # temperature with the convergence parameters we use.
    lattice = 10.61 .* [0.0  0.5 0.5; 0.5 0.0 0.5; 0.5 0.5  0.0]
    test_cells = (merge(silicon, (; temperature=0.03)), merge(silicon, (; lattice)))
    for cell in test_cells
    #for temperature in (0.0, 0.1)
        Ecut  = 5
        kgrid = fast ? [2, 1, 1] : [2, 3, 2]
        scf_tol = 1e-9
        is_converged = DFTK.ScfConvergenceDensity(scf_tol)
        determine_diagtol = DFTK.ScfDiagtol(diagtol_max=scf_tol)

        scf_kwargs = (; is_converged, determine_diagtol)
        #scf_kwargs = (; callback=identity, is_converged, determine_diagtol)

        if fast
            supercell_size = [2, 1, 1]
            kgrid_supercell = [1, 1, 1]
            qpoints = [zeros(3), 1 ./ supercell_size]
        else
            rand_dir = rand(1:3)
            supercell_size = ones(3)
            supercell_size[rand_dir] = kgrid[rand_dir]
            kgrid_supercell = copy(kgrid)
            kgrid_supercell[rand_dir] = 1
            qpoints = [zeros(3), 1 ./ supercell_size]
        end
        supercell = create_supercell(cell.lattice, cell.atoms, cell.positions, supercell_size)
        model_supercell = model_loc(supercell.lattice, supercell.atoms, supercell.positions; symmetries=false, cell.temperature)
        basis_supercell = PlaneWaveBasis(model_supercell; Ecut, kgrid=kgrid_supercell)
        dynamical_matrix_ad = compute_dynmat_ad(basis_supercell; scf_kwargs...)
        ωs_sc = sort(phonon_eigenvalues(basis_supercell, dynamical_matrix_ad); dims=2)

        ωs_uc = []
        for q in qpoints
            model = model_loc(cell.lattice, cell.atoms, cell.positions; symmetries=false, cell.temperature)
            basis = PlaneWaveBasis(model; Ecut, kgrid)
            scfres = self_consistent_field(basis; scf_kwargs...)
            δρs, δψs, δoccupations = DFTK.compute_δρ(scfres; q)
            dynamical_matrix = compute_dynmat(scfres; δρs, δψs, δoccupations, q)
            ωs_q = phonon_eigenvalues(basis, dynamical_matrix)
            ωs_uc = hcat(ωs_uc..., ωs_q)
        end
        display(ωs_sc)
        display(sort(ωs_uc; dims=2))
        display(ωs_sc - sort(ωs_uc; dims=2))

        # Because three eigenvalues should be close to zero and the square root near zero
        # decrease machine accuracy, we expect at least ``3×2×2 - 3 = 9`` eigenvalues to have
        # norm related to the accuracy of the SCF convergence parameter and the rest to be
        # larger.
        @test count(abs.(sort(ωs_uc; dims=2) - ωs_sc) .< sqrt(scf_tol)) ≥ 9
        @test count(sqrt(scf_tol) .< abs.(sort(ωs_uc; dims=2) - ωs_sc) .< scf_tol^(1/4)) ≤ 3
    end
end
=#

end
