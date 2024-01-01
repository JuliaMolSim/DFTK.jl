@testitem "Coulomb vertex generation" tags=[:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    testcase = TestCases.silicon
    model  = model_LDA(testcase.lattice, testcase.atoms, testcase.positions)
    basis  = PlaneWaveBasis(model; Ecut=7, kgrid=(2, 2, 2))
    scfres = self_consistent_field(basis; tol=1e-4)

    n_bands = 10
    bands = compute_bands(scfres, basis.kgrid; n_bands)

    ΓmnG = DFTK.compute_coulomb_vertex(bands)

    # Test the Vertex reproduces the Coulomb energy
    energy_ref = scfres.energies["Hartree"]
    energy_from_vertex = let
        for (ik, occk) in enumerate(scfres.occupation)
            Γ_nnG = zero(ComplexF64, prod(basis.fft_size))
            Γ_nnG += sum(1:n_bands) do n
                ΓmnG[ik, n, ik, n, :] * occk[ik][n]
            end
        end
        (1//2) * real(dot(Γ_nnG, Γ_nnG))
    end
    @test energy_ref ≈ energy_from_vertex atol=1e-12

    # Test the symmetry of the vertex
    for ik in 1:length(basis.kpoints), n in 1:n_bands
        for jk in 1:length(basis.kpoints), m in 1:n_bands
            @test ΓmnG[ik, n, jk, m, :] ≈ conj(ΓmnG[jk, m, ik, n, :]) atol=1e-12
        end
    end
end
