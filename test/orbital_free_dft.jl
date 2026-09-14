@testitem "Orbital-free aluminium against DFTpy" tags=[:minimal, :dont_test_mpi] begin
    using DFTK

    # DFTpy 2.2.0: TF+VW, LDA(PZ), the same UPF pseudopotential and 24³ grid.
    # CG-HS optimization, econv=1e-12, starting from a uniform density.
    ref_etot = -2.90385354842395

    lattice = 7.652 / 2 * [0 1 1; 1 0 1; 1 1 0]
    Al = ElementPsp(:Al, load_psp(joinpath(@__DIR__, "pseudos", "Al_m.upf")))
    model = model_OFDFT(lattice, [Al], [zeros(3)];
                        kinetic_functionals=[:lda_k_tf, :gga_k_vw],
                        functionals=Xc([:lda_x, :lda_c_pz]; use_nlcc=false))
    basis = PlaneWaveBasis(model; Ecut=15, fft_size=(24, 24, 24), kgrid=(1, 1, 1))
    ρ = fill(3 / model.unit_cell_volume, basis.fft_size..., 1)
    result = direct_minimization_density(basis; ρ, tol=1e-7, maxiter=500,
                                         show_trace=false)

    @test result.converged
    @test sum(result.ρ) * basis.dvol ≈ 3 atol=1e-12
    @test result.energies.total ≈ ref_etot atol=1e-7
end
