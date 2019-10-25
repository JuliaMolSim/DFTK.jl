using Test
using DFTK: PlaneWaveModel, Model, model_hcore, load_psp, Species
using DFTK: Hamiltonian, kblock

include("testcases.jl")

@testset "kblock of a core Hamiltonian" begin
    testcase = silicon
    Ecut = 10
    fft_size = [21, 21, 21]

    spec = Species(testcase.atnum, psp=load_psp(testcase.psp))
    model = model_hcore(testcase.lattice, spec => testcase.positions)
    basis = PlaneWaveModel(model, fft_size, Ecut, testcase.kcoords, testcase.ksymops)
    ham = Hamiltonian(basis)

    for (ik, kpt) in enumerate(basis.kpoints)
        hamk = kblock(ham, kpt)
        mat = Matrix(kblock(ham, kpt))

        v = randn(ComplexF64, length(kpt.basis))
        @test mat * v ≈ hamk * v
    end
end
