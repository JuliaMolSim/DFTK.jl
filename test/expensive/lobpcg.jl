using Test
using DFTK: PlaneWaveBasis, Model, Hamiltonian, term_external
using DFTK: lobpcg_hyper

include("../testcases.jl")

@testset "Diagonalisation of kinetic + local PSP" begin
    Ecut = 25
    fft_size = [33, 33, 33]

    Si = Element(silicon.atnum, psp=load_psp("hgh/lda/si-q4"))
    model = Model(silicon.lattice, silicon.n_electrons,
                  external=term_external(Si => silicon.positions))
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)
    ham = Hamiltonian(basis)

    res = diagonalise_all_kblocks(lobpcg_hyper, ham, 6, tol=1e-8)

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
    for ik in 1:length(silicon.kcoords)
        @test res.λ[ik][1:5] ≈ ref[ik] atol=5e-7
    end
end
