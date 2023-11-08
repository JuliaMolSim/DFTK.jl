@testitem "Hydrogen atom" begin
    using DFTK: energy_forces_ewald
    using LinearAlgebra

    lattice = 16 * Diagonal(ones(3))
    charges = [1]
    positions = [[0,0,0]]

    ref = -0.088665545  # TODO source?
    γ_E = energy_forces_ewald(lattice, charges, positions).energy
    @test abs(γ_E - ref) < 1e-8
end

@testitem "Silicon diamond structure" begin
    using DFTK: energy_forces_ewald

    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0]
    positions = [ones(3)/8, -ones(3)/8]
    charges = [14, 14]

    ref = -102.8741963352893
    γ_E = energy_forces_ewald(lattice, charges, positions).energy
    @test abs(γ_E - ref) < 1e-8
end

@testitem "Boron molecule" begin
    using DFTK: energy_forces_ewald
    using LinearAlgebra

    lattice = 16 * Diagonal(ones(3))
    positions = [[0,0,0], [0.14763485355139283, 0, 0]]
    charges = [5, 5]

    ref = 1.790634595  # TODO source?
    γ_E = energy_forces_ewald(lattice, charges, positions).energy
    @test abs(γ_E - ref) < 1e-7
end

@testitem "Hydrogen molecule" begin
    using DFTK: energy_forces_ewald
    using LinearAlgebra

    lattice = 16 * Diagonal(ones(3))
    positions = [
        [0.45312500031210007, 1/2, 1/2],
        [0.5468749996028622, 1/2, 1/2],
    ]
    charges = [1, 1]

    ref = 0.31316999  # TODO source?
    γ_E = energy_forces_ewald(lattice, charges, positions).energy
    @test abs(γ_E - ref) < 1e-7
end

@testitem "Forces" begin
    using DFTK
    using DFTK: energy_forces_ewald
    using LinearAlgebra

    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0]
    # perturb positions away from equilibrium to get nonzero force
    positions = Vec3.([ones(3)/8+rand(3)/20, -ones(3)/8])
    charges = [14, 14]

    γ1, forces = energy_forces_ewald(lattice, charges, positions)

    # Compare forces to finite differences
    disp = [rand(3)/20, rand(3)/20]
    ε = 1e-8
    γ2 = energy_forces_ewald(lattice, charges, positions .+ ε .* disp).energy
    @test (γ2-γ1)/ε ≈ -dot(disp, forces) atol=abs(γ1*1e-6)
end
