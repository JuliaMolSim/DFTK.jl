using Test
using DFTK

include("testcases.jl")

Si = ElementPsp(silicon.atnum, psp=load_psp("hgh/lda/si-q4"))
atoms = [Si, Si]

@testset "Energy cutoff smearing on silicon LDA" begin
    # LDA term with modified kinetic term
    blowup=BlowupCHV()
    terms = [Kinetic(;blowup), AtomicLocal(), AtomicNonlocal(),
             Ewald(), PspCorrection(), Hartree(), Xc([:lda_x, :lda_c_pw])]
    # Launch scf for a low Ecut = 5 Ha
    model = Model(silicon.lattice, atoms, silicon.positions; terms)
    basis = PlaneWaveBasis(model, 5, silicon.kcoords, silicon.kweights)
    scfres = self_consistent_field(basis; n_bands=8)
    @test scfres.converged
end
