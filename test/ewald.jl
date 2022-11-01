using Test
using DFTK: energy_ewald
using LinearAlgebra

@testset "Ewald in 1D" begin
    lattice = [1.0 0 0; 0 0 0; 0 0 0]
    charges = [1]
    positions = [[0,0,0]]

    ref = -4.67579376684418
    γ_E = energy_ewald(lattice, charges, positions)
    @test abs(γ_E - ref) < 1e-8
end

@testset "Hydrogen atom" begin
    lattice = 16 * Diagonal(ones(3))
    charges = [1]
    positions = [[0,0,0]]

    ref = -0.088665545  # TODO source?
    γ_E = energy_ewald(lattice, charges, positions)
    @test abs(γ_E - ref) < 1e-8
end

@testset "Silicon diamond structure" begin
    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0]
    positions = [ones(3)/8, -ones(3)/8]
    charges = [14, 14]

    ref = -102.8741963352893
    γ_E = energy_ewald(lattice, charges, positions)
    @test abs(γ_E - ref) < 1e-8
end

@testset "Boron molecule" begin
    lattice = 16 * Diagonal(ones(3))
    positions = [[0,0,0], [0.14763485355139283, 0, 0]]
    charges = [5, 5]

    ref = 1.790634595  # TODO source?
    γ_E = energy_ewald(lattice, charges, positions)
    @test abs(γ_E - ref) < 1e-7
end

@testset "Hydrogen molecule" begin
    lattice = 16 * Diagonal(ones(3))
    positions = [
        [0.45312500031210007, 1/2, 1/2],
        [0.5468749996028622, 1/2, 1/2],
    ]
    charges = [1, 1]

    ref = 0.31316999  # TODO source?
    γ_E = energy_ewald(lattice, charges, positions)
    @test abs(γ_E - ref) < 1e-7
end

@testset "Forces" begin
    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0]
    # perturb positions away from equilibrium to get nonzero force
    positions = [ones(3)/8+rand(3)/20, -ones(3)/8]
    charges = [14, 14]

    forces = zeros(Vec3{Float64}, 2)
    γ1 = energy_ewald(lattice, charges, positions; forces)

    # Compare forces to finite differences
    disp = [rand(3)/20, rand(3)/20]
    ε = 1e-8
    γ2 = energy_ewald(lattice, charges, positions .+ ε .* disp)
    @test (γ2-γ1)/ε ≈ -dot(disp, forces) atol=abs(γ1*1e-6)
end
