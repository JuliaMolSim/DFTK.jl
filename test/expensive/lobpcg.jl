using Test
using DFTK: PlaneWaveBasis, load_psp, build_local_potential, Hamiltonian
using DFTK: lobpcg, eval_psp_local_fourier

include("../silicon_testcases.jl")

@testset "Diagonalisation of kinetic + local psp" begin
    Ecut = 25
    grid_size = [33, 33, 33]
    pw = PlaneWaveBasis(lattice, grid_size, Ecut, kpoints, kweights)
    hgh = load_psp("si-pade-q4.hgh")

    pot_local = build_local_potential(pw, positions,
                                      G -> eval_psp_local_fourier(hgh, pw.recip_lattice * G))
    ham = Hamiltonian(pw, pot_local=pot_local)
    res = lobpcg(ham, 6, tol=1e-8)

    ref = [
        [-4.087198659513310, -4.085326314828677, -0.506869382308294,
         -0.506869382280876, -0.506869381798614],
        [-4.085824585443292, -4.085418874576503, -0.509716820984169,
         -0.509716820267449, -0.508545832298541],
        [-4.086645155119840, -4.085209948598607, -0.514320642233337,
         -0.514320641863231, -0.499373272772206],
        [-4.085991608422304, -4.085039856878318, -0.517299903754010,
         -0.513805498246478, -0.497036479690380]
    ]
    for ik in 1:length(kpoints)
        @test res.λ[ik][1:5] ≈ ref[ik] atol=5e-7
    end
end
