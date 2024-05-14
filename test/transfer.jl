@testitem "Transfer of blochwave" setup=[TestCases] begin
    using DFTK
    using DFTK: transfer_blochwave, compute_transfer_matrix
    using LinearAlgebra
    silicon = TestCases.silicon

    tol = 1e-7
    Ecut = 5

    model  = model_LDA(silicon.lattice, silicon.atoms, silicon.positions)
    kgrid  = [2, 2, 2]
    kshift = [1, 1, 1] / 2
    basis  = PlaneWaveBasis(model; Ecut, kgrid, kshift)

    ψ = self_consistent_field(basis; tol, callback=identity).ψ

    ## Testing transfers from basis to a bigger_basis and backwards

    # Transfer to bigger basis then same basis (both interpolations are
    # tested then)
    bigger_basis = PlaneWaveBasis(model; Ecut=(Ecut + 5), kgrid, kshift)
    ψ_b  = transfer_blochwave(ψ, basis, bigger_basis)
    ψ_bb = transfer_blochwave(ψ_b, bigger_basis, basis)
    @test norm(ψ - ψ_bb) < eps(eltype(basis))

    T    = compute_transfer_matrix(basis, bigger_basis)  # transfer
    Tᵇ   = compute_transfer_matrix(bigger_basis, basis)  # back-transfer
    Tψ   = [Tk * ψk   for (Tk, ψk)   in zip(T, ψ)]
    TᵇTψ = [Tᵇk * Tψk for (Tᵇk, Tψk) in zip(Tᵇ, Tψ)]
    @test norm(ψ - TᵇTψ) < eps(eltype(basis))

    # TᵇT should be the identity and TTᵇ should be a projection
    TᵇT = [Tᵇk * Tk  for (Tk, Tᵇk) in zip(T, Tᵇ)]
    @test all(M -> maximum(abs, M-I) < eps(eltype(basis)), TᵇT)

    TTᵇ = [Tk  * Tᵇk for (Tk, Tᵇk) in zip(T, Tᵇ)]
    @test all(M -> M ≈ M*M, TTᵇ)

    # Transfer between same basis (not very useful, but is worth testing)
    bigger_basis = PlaneWaveBasis(model; Ecut, kgrid, kshift)
    ψ_b = transfer_blochwave(ψ, basis, bigger_basis)
    ψ_bb = transfer_blochwave(ψ_b, bigger_basis, basis)
    @test norm(ψ-ψ_bb) < eps(eltype(basis))
end

@testitem "Transfer of density" begin
    using DFTK
    using DFTK: transfer_density
    using LinearAlgebra

    model = Model(diagm(ones(3)))
    kgrid = [1, 1, 1]
    Ecut  = 10

    # Test grids that have both even and odd sizes.
    @testset "Small -> big -> small is identity" begin
        basis      = PlaneWaveBasis(model; Ecut, kgrid, fft_size=(15, 30,  1))
        basis_big  = PlaneWaveBasis(model; Ecut, kgrid, fft_size=(20, 33, 11))

        # A random density on an even-sized grid has a Fourier component G, where its
        # counterpart -G is *not* part of the FFT grid, therefore ifft(fft(ρ)) would
        # not be an identity. To prevent this we use enforce_real! to explicitly set
        # the non-matched Fourier component to zero.
        ρ = random_density(basis, 1)
        ρ_fourier_purified = DFTK.enforce_real!(fft(basis, ρ), basis)
        ρ = irfft(basis, ρ_fourier_purified)

        ρ_b  = transfer_density(ρ,   basis,     basis_big)
        ρ_bb = transfer_density(ρ_b, basis_big, basis    )
        @test ρ_bb ≈ ρ rtol=10eps(eltype(basis))
    end

    # Transfer from big to small basis and back, check that this acts
    # like identity on the Fourier components of the small basis,
    # where G has a matching -G (for the others a small error might occur)
    @testset "Big -> small -> big is identity on small basis" begin
        basis_big  = PlaneWaveBasis(model; Ecut, kgrid, fft_size=(16, 24, 1))
        basis      = PlaneWaveBasis(model; Ecut, kgrid, fft_size=( 9, 10, 1))

        ρ    = random_density(basis_big, 1)
        ρ_s  = transfer_density(ρ,   basis_big, basis    )
        ρ_ss = transfer_density(ρ_s, basis,     basis_big)

        Δρ_fourier = fft(basis_big, ρ - ρ_ss)
        for (iG, G) in enumerate(G_vectors(basis_big))
            idx          = DFTK.index_G_vectors(basis,  G)
            idx_matching = DFTK.index_G_vectors(basis, -G)
            if !isnothing(idx) && !isnothing(idx_matching)
                @test abs(Δρ_fourier[iG]) < 10eps(eltype(basis))
            end
        end
    end
end

# Don't test MPI for now, as the processor that has k-point k may not have k+p.
@testitem "Construct k-point from equivalent" tags=[:dont_test_mpi] begin
    using DFTK
    using DFTK: get_kpoint
    using LinearAlgebra

    model = Model(diagm(ones(3)))
    k = 10

    basis = PlaneWaveBasis(model; Ecut=100, kgrid=[k for _ in 1:3])
    coordinate = [rand(1:k) for _ in 1:3] ./ [k for _ in 1:3]
    for kpt in basis.kpoints
        kpt_new = get_kpoint(basis, kpt.coordinate + coordinate, kpt.spin).kpt
        kpt_ref = Kpoint(basis, kpt.coordinate + coordinate, kpt.spin)
        @test kpt_new.spin == kpt_ref.spin
        @test kpt_new.coordinate ≈ kpt_ref.coordinate
        @test sort(kpt_new.mapping) == kpt_ref.mapping
        @test sort(kpt_new.G_vectors) == sort(kpt_ref.G_vectors)
    end
end
