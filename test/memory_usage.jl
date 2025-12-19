@testitem "memory usage" setup=[TestCases] begin
    using DFTK
    silicon = TestCases.silicon

    @testset "silicon no spin" begin
        model = model_DFT(silicon.lattice, silicon.atoms, silicon.positions;
                          functionals=LDA())
        basis = PlaneWaveBasis(model; kgrid=[2, 2, 2], Ecut=10)
        memstats = DFTK.estimate_memory_usage(model; kgrid=[2, 2, 2], Ecut=10)
        N_IRREDUCIBLE_KPOINTS = 3 # for silicon with 2x2x2 grid

        @test memstats.n_kpoints == cld(N_IRREDUCIBLE_KPOINTS, mpi_nprocs())
        ρ = guess_density(basis)
        @test memstats.ρ_bytes == sizeof(ρ)
    end
    @testset "silicon with spin" begin
        model = model_DFT(silicon.lattice, silicon.atoms, silicon.positions;
                          functionals=LDA(), spin_polarization=:collinear,
                          magnetic_moments=[2, 2])
        basis = PlaneWaveBasis(model; kgrid=[2, 2, 2], Ecut=10)
        memstats = DFTK.estimate_memory_usage(model; kgrid=[2, 2, 2], Ecut=10)
        N_IRREDUCIBLE_KPOINTS = 3 # for silicon with 2x2x2 grid

        @test memstats.n_kpoints == 2 * cld(N_IRREDUCIBLE_KPOINTS, mpi_nprocs())
        ρ = guess_density(basis, [2, 2])
        @test memstats.ρ_bytes == sizeof(ρ)
    end
end