@testitem "Dielectric tensor of silicon" tags=[:slow] setup=[TestCases] begin
    using DFTK
    using LinearAlgebra
    silicon = TestCases.silicon

    # The field response reuses the k-point symmetry reduction only on a symmetry-free basis
    # (a reduced basis would over-symmetrize the response); build the model accordingly.
    model  = model_DFT(silicon.lattice, silicon.atoms, silicon.positions;
                       functionals=LDA(), symmetries=false)
    basis  = PlaneWaveBasis(model; Ecut=12, kgrid=(4, 4, 4))
    scfres = self_consistent_field(basis; tol=1e-10)

    res = compute_dielectric(scfres; tol=1e-8, verbose=false)
    ε∞  = res.ε∞

    # Silicon is cubic, so ε∞ must be isotropic (≈ ε·I). Nothing enforces this on the
    # symmetry-free grid, so it is a genuine correctness check. The off-diagonal is zero only
    # up to the response-solver tolerance, hence a relative threshold.
    @test norm(ε∞ - Diagonal(diag(ε∞))) < 1e-4 * norm(diag(ε∞))
    @test all(d -> d ≈ diag(ε∞)[1], diag(ε∞))

    # ε∞ is symmetric (self-adjointness of (Ω+K)⁻¹) and larger than 1 for a stable insulator.
    @test norm(ε∞ - ε∞') < 1e-8
    @test all(diag(ε∞) .> 1)

    # Physical range for Si LDA (this coarse grid over-estimates; the converged value is ≈ 13).
    @test 10 < diag(ε∞)[1] < 40

    # Internal consistency of the polarizability: ε∞ = 1 + 4π (Ω·χ)/Ω on the diagonal.
    Ω = model.unit_cell_volume
    @test diag(ε∞) ≈ 1 .+ 4π .* diag(res.polarizability) ./ Ω

    # Metals (fractional occupations from smearing) are out of scope and must error.
    model_metal  = model_DFT(silicon.lattice, silicon.atoms, silicon.positions;
                             functionals=LDA(), temperature=0.02, symmetries=false)
    basis_metal  = PlaneWaveBasis(model_metal; Ecut=12, kgrid=(2, 2, 2))
    scfres_metal = self_consistent_field(basis_metal; tol=1e-8)
    @test_throws ErrorException compute_dielectric(scfres_metal; verbose=false)
end
