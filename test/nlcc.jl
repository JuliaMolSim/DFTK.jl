@testitem "Core charge density is positive" begin
    using DFTK
    using DFTK: CoreDensity, atomic_total_density
    using LinearAlgebra
    using PseudoPotentialData

    elements = (
        :Si, :Fe, :Ir,  # With NLCC
        :Li, :Mg,       # without NLCC
    )

    lattice = 5 * I(3)
    positions = [zeros(3)]
    pd_lda_family = PseudoFamily("dojo.nc.sr.lda.v0_4_1.oncvpsp3.standard.upf")
    for element in elements
        atoms = [ElementPsp(element, pd_lda_family)]
        model = model_DFT(lattice, atoms, positions; functionals=LDA())
        basis = PlaneWaveBasis(model; Ecut=24, kgrid=[2, 2, 2])
        ρ_core = @inferred atomic_total_density(basis, CoreDensity())
        ρ_core_neg = abs(sum(ρ_core[ρ_core .< 0]))
        @test ρ_core_neg * model.unit_cell_volume / prod(basis.fft_size) < 1e-6
    end
end
