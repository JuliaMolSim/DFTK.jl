#=
Tests on GPU
=#

include("hamiltonian_consistency_common.jl")
using CUDA
using Random
Random.seed!(0)

@testset "GPU Hamiltonian consistency" begin

    test_consistency_term(Kinetic(), architecture=DFTK.GPU(CuArray))
    test_consistency_term(AtomicLocal(), architecture=DFTK.GPU(CuArray))
    test_consistency_term(AtomicNonlocal(), architecture=DFTK.GPU(CuArray))
    test_consistency_term(ExternalFromReal(X -> cos(X[1])), architecture=DFTK.GPU(CuArray))
    test_consistency_term(ExternalFromFourier(X -> abs(norm(X))), architecture=DFTK.GPU(CuArray))
    test_consistency_term(LocalNonlinearity(ρ -> ρ^2), architecture=DFTK.GPU(CuArray))
    test_consistency_term(Hartree(), architecture=DFTK.GPU(CuArray))
    test_consistency_term(Ewald(), architecture=DFTK.GPU(CuArray))
    test_consistency_term(PspCorrection(), architecture=DFTK.GPU(CuArray))
    test_consistency_term(Xc(:lda_xc_teter93), architecture=DFTK.GPU(CuArray))
    test_consistency_term(Xc(:lda_xc_teter93), spin_polarization=:collinear, architecture=DFTK.GPU(CuArray))
    test_consistency_term(Xc(:gga_x_pbe), spin_polarization=:collinear, architecture=DFTK.GPU(CuArray))

    # test_consistency_term(Xc(:mgga_x_tpss), architecture=DFTK.GPU(CuArray))
    # test_consistency_term(Xc(:mgga_x_scan), architecture=DFTK.GPU(CuArray))
    # test_consistency_term(Xc(:mgga_c_scan), spin_polarization=:collinear, architecture=DFTK.GPU(CuArray))
    # test_consistency_term(Xc(:mgga_x_b00), architecture=DFTK.GPU(CuArray))
    # test_consistency_term(Xc(:mgga_c_b94), spin_polarization=:collinear, architecture=DFTK.GPU(CuArray))

    let
        a = 6
        pot(x, y, z) = (x - a/2)^2 + (y - a/2)^2
        Apot(x, y, z) = .2 * [y - a/2, -(x - a/2), 0]
        Apot(X) = Apot(X...)
        test_consistency_term(Magnetic(Apot); kgrid=[1, 1, 1], kshift=[0, 0, 0],
                              lattice=[a 0 0; 0 a 0; 0 0 0], Ecut=20, architecture=DFTK.GPU(CuArray))
        # Anyonic term not implemented
        #test_consistency_term(DFTK.Anyonic(2, 3.2); kgrid=[1, 1, 1], kshift=[0, 0, 0],
        #                      lattice=[a 0 0; 0 a 0; 0 0 0], Ecut=20, architecture=DFTK.GPU(CuArray))
    end
end