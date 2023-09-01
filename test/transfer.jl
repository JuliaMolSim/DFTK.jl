using Test
using DFTK
using DFTK: transfer_blochwave, compute_transfer_matrix, transfer_density
using LinearAlgebra

include("testcases.jl")

@testset "Transfer of blochwave" begin
    tol = 1e-7
    Ecut = 5

    model  = model_LDA(silicon.lattice, silicon.atoms, silicon.positions)
    kgrid  = [2, 2, 2]
    kshift = [1, 1, 1] / 2
    basis  = PlaneWaveBasis(model; Ecut, kgrid, kshift)

    ψ = self_consistent_field(basis; tol, callback=identity).ψ

    ## Testing transfers from basis to a bigger_basis and backwards

    # Transfer to bigger basis then same basis (both interpolations are
    # tested then)
    bigger_basis = PlaneWaveBasis(model; Ecut=(Ecut + 5), kgrid, kshift)
    ψ_b = transfer_blochwave(ψ, basis, bigger_basis)
    ψ_bb = transfer_blochwave(ψ_b, bigger_basis, basis)
    @test norm(ψ - ψ_bb) < eps(eltype(basis))

    T    = compute_transfer_matrix(basis, bigger_basis)  # transfer
    Tᵇ   = compute_transfer_matrix(bigger_basis, basis)  # back-transfer
    Tψ   = [Tk * ψk   for (Tk, ψk)   in zip(T, ψ)]
    TᵇTψ = [Tᵇk * Tψk for (Tᵇk, Tψk) in zip(Tᵇ, Tψ)]
    @test norm(ψ - TᵇTψ) < eps(eltype(basis))

    # TᵇT should be the identity and TTᵇ should be a projection
    TᵇT = [Tᵇk * Tk  for (Tk, Tᵇk) in zip(T, Tᵇ)]
    @test all(M -> maximum(abs, M-I) < eps(eltype(basis)), TᵇT)

    TTᵇ = [Tk  * Tᵇk for (Tk, Tᵇk) in zip(T, Tᵇ)]
    @test all(M -> M ≈ M*M, TTᵇ)

    # Transfer between same basis (not very useful, but is worth testing)
    bigger_basis = PlaneWaveBasis(model; Ecut, kgrid, kshift)
    ψ_b = transfer_blochwave(ψ, basis, bigger_basis)
    ψ_bb = transfer_blochwave(ψ_b, bigger_basis, basis)
    @test norm(ψ-ψ_bb) < eps(eltype(basis))
end

@testset "Transfer of density" begin
    model = Model(diagm(ones(3)))
    kgrid = [1,1,1]
    Ecut = 10

    # Test grids that have both even and odd sizes.

    # Transfer from small to big basis and back, check that this acts
    # like identity.
    basis      = PlaneWaveBasis(model; Ecut, kgrid, fft_size=(15,30,1))
    basis_big  = PlaneWaveBasis(model; Ecut, kgrid, fft_size=(20,33,11))
    ρ = random_density(basis,1)
    ρ_b  = transfer_density(ρ, basis, basis_big)
    ρ_bb = transfer_density(ρ_b, basis_big, basis)
    @test norm(ρ_bb - ρ)/norm(ρ) < 10eps(eltype(basis))

    # Transfer from big to small basis and back, check that this acts
    # like identity on the Fourier components of the small basis.
    basis      = PlaneWaveBasis(model; Ecut, kgrid, fft_size=(9,10,1))
    basis_big  = PlaneWaveBasis(model; Ecut, kgrid, fft_size=(16,24,1))
    ρ = random_density(basis_big,1)
    ρ_s  = transfer_density(ρ, basis_big, basis)
    ρ_ss = transfer_density(ρ_s, basis, basis_big)
    err = norm([x-y for (x,y) in zip(fft(basis_big, ρ_ss)[:,1,1,1],
                                     fft(basis_big,ρ)[:,1,1,1])
                if abs(x)>10eps(eltype(basis_big))])
    @test err/norm(ρ_ss) < 10eps(eltype(basis_big))
end
