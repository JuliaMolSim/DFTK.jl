using DFTK
using Test
include("testcases.jl")

@testset "Symmetrization and not symmetrization yields the same density and energy" begin
    args = ((kgrid=[2, 2, 2], kshift=[1/2, 0, 0]),
            (kgrid=[2, 2, 2], kshift=[1/2, 1/2, 0]),
            (kgrid=[2, 2, 2], kshift=[0, 0, 0]),
            (kgrid=[3, 2, 3], kshift=[0, 0, 0]),
            (kgrid=[3, 2, 3], kshift=[0, 1/2, 1/2]),
            )
    for a in args
        Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
        atoms = [Si => [(ones(3)) / 8, -ones(3) / 8]]

        model_nosym = model_DFT(silicon.lattice, atoms, :lda_xc_teter93, symmetries=false)
        basis = PlaneWaveBasis(model_nosym; Ecut=5, kgrid=(1, 1, 1), a...)
        DFTK.check_group(basis.symmetries)

        scfres = self_consistent_field(basis; is_converged=DFTK.ScfConvergenceDensity(1e-10))
        ρ1 = scfres.ρ
        E1 = scfres.energies.total

        model_sym = model_DFT(silicon.lattice, atoms, :lda_xc_teter93)
        basis = PlaneWaveBasis(model_sym; Ecut=5, kgrid=(1, 1, 1), a...)
        scfres = self_consistent_field(basis; is_converged=DFTK.ScfConvergenceDensity(1e-10))
        ρ2 = scfres.ρ
        E2 = scfres.energies.total

        @test abs(E1 - E2) < 1e-10
        @test norm(ρ1 - ρ2) .* sqrt(basis.dvol) < 1e-8
    end
end
