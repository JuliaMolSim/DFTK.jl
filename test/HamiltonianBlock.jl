using Test
using DFTK: PlaneWaveBasis, Model, model_hcore, load_psp, ElementPsp
using DFTK: Hamiltonian, kblock

include("testcases.jl")

@testset "kblock of a core Hamiltonian" begin
    testcase = silicon
    Ecut = 10
    fft_size = [21, 21, 21]

    spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
    model = model_hcore(testcase.lattice, [spec => testcase.positions])
    basis = PlaneWaveBasis(model, Ecut, testcase.kcoords, testcase.ksymops; fft_size=fft_size)
    ham = Hamiltonian(basis)

    for (ik, kpt) in enumerate(basis.kpoints)
        hamk = kblock(ham, kpt)
        mat = Matrix(kblock(ham, kpt))

        v = randn(ComplexF64, length(G_vectors(kpt)))
        @test mat * v ≈ hamk * v
    end
end
