using DFTK
using Test

include("testcases.jl")

@testset "Apply LDA XC kernel" begin
    Ecut=3
    fft_size = [10, 1, 10]
    tol=1e-10
    ε = 1e-8
    kgrid = [3, 1, 1]
    testcase = silicon

    spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
    model = model_LDA(testcase.lattice, [spec => testcase.positions])
    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, fft_size=fft_size)
    ρ = guess_density(basis)

    xc_terms = [t for t in basis.terms if t isa DFTK.TermXc]
    @assert length(xc_terms) == 1
    xc = xc_terms[1]

    # In LDA, the potential depends on each component individually, so
    # we just compute each with finite differences
    pot0 = ene_ops(xc, nothing, nothing; ρ=ρ).ops[1].potential
    ρ_pert = from_real(basis, ρ.real .+ ε .* ones(Float64, size(ρ.real)))
    pot1 = ene_ops(xc, nothing, nothing; ρ=ρ_pert).ops[1].potential
    kernel = Diagonal((vec(pot1) .- vec(pot0)) ./ ε)

    dρ = 0.01rand(size(ρ.real)...)
    ref = reshape(kernel * vec(dρ), size(dρ))

    res = DFTK.apply_kernel(xc, from_real(basis, dρ), ρ=ρ).real
    @test ref ≈ res atol=1e-7
end


# TODO Test Hartree kernel
