using DFTK
using LinearAlgebra
using Test

@testset "Anyons" begin
    # Test that the magnetic field satisfies ∇∧A = 2π ρref, ∇⋅A = 0
    x = 1.23
    y = -1.8
    ε = 1e-8
    M = 2.31
    σ = 1.81
    dAdx = (DFTK.magnetic_field_produced_by_ρref(x+ε, y, M, σ) - DFTK.magnetic_field_produced_by_ρref(x, y, M, σ))/ε
    dAdy = (DFTK.magnetic_field_produced_by_ρref(x, y+ε, M, σ) - DFTK.magnetic_field_produced_by_ρref(x, y, M, σ))/ε
    curlA = dAdx[2] - dAdy[1]
    divA = dAdx[1] + dAdy[2]
    @test norm(curlA - 2π*DFTK.ρref_real(x, y, M, σ)) < 1e-4
    @test abs(divA) < 1e-6
end
