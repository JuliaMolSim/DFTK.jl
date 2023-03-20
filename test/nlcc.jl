using Test
using DFTK: CoreDensity, atomic_total_density
using LinearAlgebra
using PseudoPotentialIO

pseudos = Dict(
    # With NLCC
    :Si => PseudoPotentialIO.load_psp("pd_nc_sr_lda_standard_0.4.1_upf", "Si.upf"),
    :Fe => PseudoPotentialIO.load_psp("pd_nc_sr_lda_standard_0.4.1_upf", "Fe.upf"),
    :Ir => PseudoPotentialIO.load_psp("pd_nc_sr_lda_standard_0.4.1_upf", "Ir.upf"),
    # Without NLCC
    :Li => PseudoPotentialIO.load_psp("pd_nc_sr_lda_standard_0.4.1_upf", "Li.upf"),
    :Mg => PseudoPotentialIO.load_psp("pd_nc_sr_lda_standard_0.4.1_upf", "Mg.upf")
)

@testset "Core charge density is positive" begin
    lattice = 5 * I(3)
    positions = [zeros(3)]
    for (element, psp) in pseudos
        atoms = [ElementPsp(element, psp=psp)]
        model = model_LDA(lattice, atoms, positions)
        basis = PlaneWaveBasis(model; Ecut=24, kgrid=[2, 2, 2])
        ρ_core = atomic_total_density(basis, CoreDensity())
        ρ_core_neg = abs(sum(ρ_core[ρ_core .< 0]))
        @test ρ_core_neg * model.unit_cell_volume / prod(basis.fft_size) < 1e-6
    end
end
