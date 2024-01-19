# Quick test to make sure temperature, smearing and Fermi level are correctly propagated
@testitem "Supercell copy" tags=[:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    magnesium = TestCases.magnesium

    Ecut    = 4
    kgrid   = [2, 3, 1]

    model = model_LDA(magnesium.lattice, magnesium.atoms, magnesium.positions;
                      magnesium.temperature, εF=0.5, spin_polarization=:spinless,
                      disable_electrostatics_check=true)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    scfres = self_consistent_field(basis; nbandsalg=FixedBands(; n_bands_converge=20))
    scfres_supercell = cell_to_supercell(scfres)

    @test scfres.energies.total * prod(kgrid) ≈ scfres_supercell.energies.total
end

@testitem "Compare scf results in unit cell and supercell" #=
    =#    tags=[:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using LinearAlgebra
    silicon = TestCases.silicon

    Ecut    = 4
    kgrid   = [3, 3, 3]
    kshift  = zeros(3)
    tol     = 1e-10

    model = model_LDA(silicon.lattice, silicon.atoms, silicon.positions)
    basis = PlaneWaveBasis(model; Ecut, kgrid, kshift)
    basis_supercell = cell_to_supercell(basis)

    scfres = self_consistent_field(basis; tol)
    scfres_supercell_manual = self_consistent_field(basis_supercell; tol)
    scfres_supercell = cell_to_supercell(scfres)

    # Compare energies
    @test norm(scfres.energies.total * prod(kgrid) -
               scfres_supercell_manual.energies.total) < 1e-8
    @test scfres.energies.total * prod(kgrid) ≈ scfres_supercell.energies.total

    # Compare densities
    ρ_ref = DFTK.interpolate_density(scfres.ρ, basis, basis_supercell)
    @test norm(ρ_ref .- scfres_supercell.ρ) < 10*tol
    @test norm(ρ_ref .- scfres_supercell_manual.ρ) < 10*tol
end

@testitem "Supercell response" tags=[:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using LinearAlgebra
    (; silicon, magnesium) = TestCases.all_testcases

    Ecut    = 5.0
    kgrid   = [2, 1, 1]
    tol     = 1e-6

    for system in (silicon, magnesium), extra_terms in ([], [Hartree()])
        @testset "$(DFTK.periodic_table[system.atnum].symbol) with $extra_terms" begin
            model = model_atomic(system.lattice, system.atoms, system.positions;
                                 system.temperature, extra_terms)
            basis = PlaneWaveBasis(model; Ecut, kgrid)
            scfres = self_consistent_field(basis; tol)

            n_spin = model.n_spin_components
            δV = guess_density(basis)
            δV_supercell = vcat(δV, δV)

            # Unit cell computations.
            δρ = apply_χ0(scfres, δV)

            # Supercell with manually unpacking scfres.
            scfres_supercell_1 = cell_to_supercell(scfres)
            δρ_supercell_1 = apply_χ0(scfres_supercell_1, δV_supercell)

            @test norm(δρ - δρ_supercell_1[1:size(δρ, 1), :, :]) < 10*tol

            # Supercell with manually unpacking only basis.
            basis_supercell = cell_to_supercell(basis)
            scfres_supercell_2 = self_consistent_field(basis_supercell; tol)
            δρ_supercell_2 = apply_χ0(scfres_supercell_2, δV_supercell)

            @test norm(δρ - δρ_supercell_2[1:size(δρ, 1), :, :]) < 10*tol
        end
    end
end
