using Test
using DFTK: interpolate_density, transfer_blochwave
using LinearAlgebra

include("testcases.jl")

@testset "Interpolation of density" begin
    lattice = Array{Float64}(I, 3, 3)
    N = 20
    f(n) = cos(2π*(n-1)/N)
    grid_in = (N, N, N)
    grid_out = (N+20, N-5, N+2)
    ρin = [f(i)*f(j)+f(k) for i = 1:N, j = 1:N, k = 1:N]

    # Test interpolation
    ρout = interpolate_density(ρin, grid_in, grid_out, lattice, lattice)
    ρin2 = interpolate_density(ρout, grid_out, grid_in, lattice, lattice)
    @test maximum(abs.(ρin2-ρin)) < .01

    # Test supercell
    ρout = interpolate_density(ρin, grid_in, grid_in, lattice, 2lattice)
    ρout1 = interpolate_density(ρout, grid_in, grid_in, 2lattice, 4lattice)
    ρout2 = interpolate_density(ρin, grid_in, grid_in, lattice, 4lattice)
    @test maximum(abs.(ρout2 - ρout1)) < sqrt(eps())

    ρout = interpolate_density(ρin, grid_in, grid_in, lattice, 2lattice)
    ρout1 = interpolate_density(ρout, grid_in, grid_out, 2lattice, 4lattice)
    ρout2 = interpolate_density(ρin, grid_in, grid_out, lattice, 4lattice)
    @test maximum(abs.(ρout2 - ρout1)) < .01
end

@testset "Transfer of blochwave" begin
    tol = 1e-7

    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
    model = model_LDA(silicon.lattice, [Si => silicon.positions])
    kgrid = [2,2,2]
    Ecut = 5
    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

    ψ = self_consistent_field(basis; tol=tol, callback=info->nothing).ψ

    ## Testing transfers from basis to a bigger_basis and backwards

    # Transfer to bigger basis then same basis (both interpolations are
    # tested then)
    bigger_basis = PlaneWaveBasis(model, Ecut+5; kgrid=kgrid)
    ψ_b = transfer_blochwave(ψ, basis, bigger_basis)
    ψ_bb = transfer_blochwave(ψ_b, bigger_basis, basis)
    @test norm(ψ-ψ_bb) < eps(eltype(basis))

    # Transfer between same basis (not very useful, but is worth testing)
    bigger_basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
    ψ_b = transfer_blochwave(ψ, basis, bigger_basis)
    ψ_bb = transfer_blochwave(ψ_b, bigger_basis, basis)
    @test norm(ψ-ψ_bb) < eps(eltype(basis))

end
