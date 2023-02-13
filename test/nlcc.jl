using Test
using DFTK: core_density_superposition
using LinearAlgebra
using LazyArtifacts

pseudo_urls = Dict(
    # With NLCC
    :Si => joinpath(artifact"pd_nc_sr_lda_standard_0.4.1_upf", "Si.upf"),
    :Fe => joinpath(artifact"pd_nc_sr_lda_standard_0.4.1_upf", "Fe.upf"),
    :Ir => joinpath(artifact"pd_nc_sr_lda_standard_0.4.1_upf", "Ir.upf"),
    # Without NLCC
    :Li => joinpath(artifact"pd_nc_sr_lda_standard_0.4.1_upf", "Li.upf"),
    :Mg => joinpath(artifact"pd_nc_sr_lda_standard_0.4.1_upf", "Mg.upf")
)
pseudos = Dict(
    key => load_psp(value)
    for (key, value) in pseudo_urls
)

@testset "Core charge density is positive" begin
    lattice = 5 * I(3)
    positions = [zeros(3)]
    for (element, psp) in pseudos
        atoms = [ElementPsp(element, psp=psp)]
        model = model_LDA(lattice, atoms, positions)
        basis = PlaneWaveBasis(model; Ecut=24, kgrid=[2, 2, 2])
        ρ_core = core_density_superposition(basis)
        ρ_core_neg = abs(sum(ρ_core[ρ_core .< 0]))
        @test ρ_core_neg * model.unit_cell_volume / prod(basis.fft_size) < 1e-6
    end
end
