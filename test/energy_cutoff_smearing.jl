"""
For Ecut=5, the first silicon band displays a discontinuity between the
X and U points. This code tests chekcs the presence of the discontinuity for
the standard kinetic term and checks that the same band computed with a modified
kinetic terms displays C^2 regularity.
"""

using Test
using DFTK

include("testcases.jl")

Si = ElementPsp(silicon.atnum, psp=load_psp("hgh/lda/si-q4"))
atoms = [Si, Si]

@testset "Energy cutoff smearing on silicon LDA" begin
    # Compute reference ground state density and fft_grid
    model = model_LDA(silicon.lattice, atoms, silicon.positions)
    basis = PlaneWaveBasis(model, 20, silicon.kcoords, silicon.kweights)
    scfres = self_consistent_field(basis; n_bands=8)

    # Kpath around one discontinuity of the first band of silicon (between X and U points)
    k_start = [0.5274, 0.0548, 0.5274]
    k_end = [0.5287, 0.0573, 0.5287]
    num_k = 100 # small number of kpoints to avoid long computations
    kcoords = map(x->(1-x)*k_start .+ x*k_end, LinRange(0, 1, num_k))

    # Assemble basis for standard and modified kinetic terms
    basis_std = PlaneWaveBasis(model, 5, silicon.kcoords, silicon.kweights; fft_size=basis.fft_size)
    terms_mod = [Kinetic(;blowup=BlowupCHV()), AtomicLocal(), AtomicNonlocal(),
                        Ewald(), PspCorrection(), Hartree(), Xc([:lda_x, :lda_c_pw])]
    model_mod = Model(silicon.lattice, atoms, silicon.positions; terms=terms_mod)
    basis_mod = PlaneWaveBasis(model_mod, 5, silicon.kcoords, silicon.kweights; fft_size=basis.fft_size)

    # Bands and (very rough) finite difference 2nd order derivative
    λ_std = vcat(compute_bands(basis_std, kcoords, n_bands=1, ρ=ρ_ref).λ...)
    λ_mod = vcat(compute_bands(basis_mod, kcoords, n_bands=1, ρ=ρ_ref).λ...)

    δk = norm(kcoords[2] .- kcoords[1], 1)
    ∂2λ_std = [(λ_std[i+1] - 2*λ_std[i] + λ_std[i-1])/δk^2 for i in 2:num_k-1]
    ∂2λ_mod = [(λ_mod[i+1] - 2*λ_mod[i] + λ_mod[i-1])/δk^2 for i in 2:num_k-1]

    @test norm(∂2λ_std) > 1e5 # Band with a discontinuity
    @test norm(∂2λ_mod) < 1   # Smooth band
end
