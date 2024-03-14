@testitem "Interpolation of density" begin
    using DFTK
    using DFTK: interpolate_density
    using LinearAlgebra

    lattice = Array{Float64}(I, 3, 3)
    N = 20
    f(n) = cos(2π*(n-1)/N)
    grid_in = (N, N, N)
    grid_out = (N+20, N-5, N+2)
    ρin = [f(i)*f(j)+f(k) for i = 1:N, j = 1:N, k = 1:N, σ = 1:1]

    # Test interpolation
    ρout = interpolate_density(ρin,  grid_in,  grid_out, lattice, lattice)
    ρin2 = interpolate_density(ρout, grid_out, grid_in,  lattice, lattice)
    @test maximum(abs, ρin2 - ρin) < .01

    # Test supercell
    ρout  = interpolate_density(ρin,  grid_in, grid_in,  lattice, 2lattice)
    ρout1 = interpolate_density(ρout, grid_in, grid_in, 2lattice, 4lattice)
    ρout2 = interpolate_density(ρin,  grid_in, grid_in,  lattice, 4lattice)
    @test maximum(abs, ρout2 - ρout1) < sqrt(eps())

    ρout  = interpolate_density(ρin,  grid_in, grid_in,   lattice, 2lattice)
    ρout1 = interpolate_density(ρout, grid_in, grid_out, 2lattice, 4lattice)
    ρout2 = interpolate_density(ρin,  grid_in, grid_out,  lattice, 4lattice)
    @test maximum(abs, ρout2 - ρout1) < .01
end
