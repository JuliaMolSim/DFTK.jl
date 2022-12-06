using Test
using DFTK
include("testcases.jl")

if mpi_nprocs() == 1  # can't be bothered to convert the tests

# Quick test to make sure temperature, smearing and Fermi level are correctly propagated
@testset "Supercell copy" begin
    Ecut    = 4
    kgrid   = [2, 1, 1]
    # Parameters
    model = model_LDA(magnesium.lattice, magnesium.atoms, magnesium.positions;
                      magnesium.temperature, εF=0.5, spin_polarization=:spinless,
                      disable_electrostatics_check=true)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    # SCF
    scfres = self_consistent_field(basis; nbandsalg=FixedBands(; n_bands_converge=20))
    scfres_supercell = cell_to_supercell(scfres)

    # Compare energies
    @test scfres.energies.total * prod(kgrid) ≈ scfres_supercell.energies.total
end

@testset "Compare scf results in unit cell and supercell" begin
    Ecut    = 4
    kgrid   = [3, 3, 3]
    kshift  = zeros(3)
    tol     = 1e-12
    scf_tol = (; is_converged=DFTK.ScfConvergenceDensity(tol))
    # Parameters
    model = model_LDA(silicon.lattice, silicon.atoms, silicon.positions)
    basis = PlaneWaveBasis(model; Ecut, kgrid, kshift)
    basis_supercell = cell_to_supercell(basis)
    # SCF
    scfres = self_consistent_field(basis; scf_tol...)
    scfres_supercell_manual = self_consistent_field(basis_supercell; scf_tol...)
    scfres_supercell = cell_to_supercell(scfres)

    # Compare energies
    @test norm(scfres.energies.total * prod(kgrid) -
               scfres_supercell_manual.energies.total) < 1e-8
    @test scfres.energies.total * prod(kgrid) ≈ scfres_supercell.energies.total

    # Compare densities
    ρ_ref = DFTK.interpolate_density(dropdims(scfres.ρ, dims=4), basis, basis_supercell)
    @test norm(ρ_ref .- scfres_supercell.ρ) < 10*tol
    @test norm(ρ_ref .- scfres_supercell_manual.ρ) < 10*tol
end

@testset "Supercell response" begin
    Ecut    = 5.0
    kgrid   = [2, 1, 1]
    scf_kw  = (; is_converged=DFTK.ScfConvergenceDensity(1e-9),
               determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-9),
               eigensolver=diag_full,
               callback=identity)

    for system in (silicon, magnesium), extra_terms in ([], [Hartree()])
        @testset "$(system.psp) with $extra_terms" begin
            model = model_atomic(system.lattice, system.atoms, system.positions;
                                 symmetries=false, system.temperature, extra_terms)
            basis = PlaneWaveBasis(model; Ecut, kgrid)
            scfres = self_consistent_field(basis; scf_kw...)

            n_spin = model.n_spin_components
            δV = ifft(basis, fft(basis, randn(eltype(basis), basis.fft_size..., n_spin)))
            δV_supercell = vcat(δV, δV)

            # Unit cell computations.
            δρ = apply_χ0(scfres, δV)

            # Supercell with manually unpacking scfres.
            scfres_supercell₁ = cell_to_supercell(scfres)
            δρ_supercell₁ = apply_χ0(scfres_supercell₁, δV_supercell)

            @test norm(δρ - δρ_supercell₁[1:size(δρ, 1), :, :]) < 1e-5

            # Supercell with manually empacking only basis.
            basis_supercell = cell_to_supercell(basis)
            scfres_supercell₂ = self_consistent_field(basis_supercell; scf_kw...)
            δρ_supercell₂ = apply_χ0(scfres_supercell₂, δV_supercell)

            @test norm(δρ - δρ_supercell₂[1:size(δρ, 1), :, :]) < 1e-5
        end
    end
end

end
