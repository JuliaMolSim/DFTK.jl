using DFTK
using Test

include("testcases.jl")

# TODO Test Hartree kernel

@testset "LDA XC kernel" begin
    Ecut=3
    ε = 1e-8
    kgrid = [3, 3, 3]
    testcase = silicon

    spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
    model = model_LDA(testcase.lattice, [spec => testcase.positions])
    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
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

    res = DFTK.apply_kernel(xc, from_real(basis, dρ), ρ=ρ)
    @test ref ≈ res atol=1e-6

    fxc = DFTK.compute_kernel(xc, ρ=ρ)
    @test fxc ≈ kernel atol=1e-4
end


@testset "GGA XC kernel" begin
    Ecut=4
    kgrid = [3, 3, 3]
    testcase = silicon
    tol=1e-10
    ε = 1e-6

    spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
    model = model_DFT(testcase.lattice, [spec => testcase.positions],
                      [:gga_x_pbe, :gga_c_pbe])
    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

    # Do 1 SCF step
    ρ0 = guess_density(basis)
    energies, ham0 = energy_hamiltonian(basis, nothing, nothing; ρ=ρ0)
    ρ1 = DFTK.next_density(ham0, tol=tol, eigensolver=diag_full, n_bands=6).ρ

    xc_terms = [t for t in basis.terms if t isa DFTK.TermXc]
    @assert length(xc_terms) == 1
    xc = xc_terms[1]

    pot0 = ene_ops(xc, nothing, nothing; ρ=ρ1).ops[1].potential
    ρ1_pert = from_real(basis, ρ1.real .+ ε * ρ0.real)
    pot1 = ene_ops(xc, nothing, nothing; ρ=ρ1_pert).ops[1].potential

    response = DFTK.apply_kernel(xc, from_real(basis, ρ0.real), ρ=ρ1)

    @test ((pot1 - pot0) ./ ε) ≈ response atol=5e-6
end
