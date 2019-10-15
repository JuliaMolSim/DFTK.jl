using Test
using DFTK: PlaneWaveModel, Model, Hamiltonian, lobpcg, PreconditionerKinetic, term_external

include("../testcases.jl")

@testset "Diagonalisation of kinetic + local PSP" begin
    Ecut = 25
    fft_size = [33, 33, 33]

    Si = Species(silicon.atnum, psp=load_psp("si-pade-q4.hgh"))
    model = Model(silicon.lattice, silicon.n_electrons,
                  external=term_external(Si => silicon.positions))
    basis = PlaneWaveModel(model, fft_size, Ecut, silicon.kcoords, silicon.kweights,
                           silicon.ksymops)
    ham = Hamiltonian(basis)
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
    for ik in 1:length(silicon.kcoords)
        @test res.λ[ik][1:5] ≈ ref[ik] atol=5e-7
    end
end
