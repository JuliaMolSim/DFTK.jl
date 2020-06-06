using DFTK
using Test

include("testcases.jl")

@testset "Kernels" begin
    Ecut=2
    kgrid = [2, 2, 2]
    testcase = silicon
    ε = 1e-8

    spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
    model = Model(testcase.lattice; atoms=[spec => testcase.positions],
                  terms=[PowerNonlinearity(1.2, 2.0),
                         Xc(:lda_xc_teter93),
                         Hartree()])
    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

    ρ0 = guess_density(basis)
    dρ = randn(size(ρ0.real))

    for term in basis.terms
        ρ_minus = from_real(basis, ρ0.real - ε * dρ)
        pot_minus = ene_ops(term, nothing, nothing; ρ=ρ_minus).ops[1].potential
        ρ_plus = from_real(basis, ρ0.real .+ ε * dρ)
        pot_plus = ene_ops(term, nothing, nothing; ρ=ρ_plus).ops[1].potential
        dV = (pot_plus - pot_minus) / (2ε)

        dV_apply = DFTK.apply_kernel(term, dρ; ρ=ρ0)
        ker = DFTK.compute_kernel(term; ρ=ρ0)
        dV_compute = reshape(ker * vec(dρ), size(dρ))
        @test norm(dV - dV_apply) < 1e-6
        @test norm(dV - dV_compute) < 1e-6
    end
end
