@testitem "Density symmetrization via stars" tags=[:minimal] setup=[TestCases] begin
    using DFTK
    using DFTK: _accumulate_over_symmetries_stars!, _accumulate_over_symmetries_direct!,
                index_G_vectors, cis2pi, Mat3, G_vectors
    using LinearAlgebra
    silicon = TestCases.silicon

    model = model_DFT(silicon.lattice, silicon.atoms, silicon.positions; functionals=LDA())
    basis = PlaneWaveBasis(model; Ecut=15, kgrid=[2, 2, 2])
    symmetries = basis.symmetries
    @test length(symmetries) > 1  # otherwise the test is vacuous

    # Reference: ρ_sym(G) = Σ_S e^{-2πi G·τ} ρ(S⁻¹G), evaluated directly at every G.
    fft_size = basis.fft_size
    Gs  = reshape(G_vectors(fft_size), fft_size)
    lin = LinearIndices(fft_size)
    ρ   = rand(ComplexF64, fft_size)
    ref = zero(ρ)
    for I in CartesianIndices(Gs), symop in symmetries
        idx = index_G_vectors(fft_size, symop.invS * Gs[I])
        isnothing(idx) || (ref[lin[I]] += cis2pi(-dot(Gs[I], symop.τ)) * ρ[lin[idx]])
    end

    out_stars = zero(ρ)
    _accumulate_over_symmetries_stars!(out_stars, ρ, basis.symmetry_stars)
    @test out_stars ≈ ref

    # The direct map! path (used on the GPU and for symmetry subsets) must agree.
    out_direct = zero(ρ)
    _accumulate_over_symmetries_direct!(out_direct, ρ, basis, symmetries)
    @test out_direct ≈ ref
end
