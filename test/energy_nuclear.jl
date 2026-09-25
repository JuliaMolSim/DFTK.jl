@testitem "energy_forces_ewald Lithium hydride" begin
    using DFTK
    using PseudoPotentialData
    using LinearAlgebra: Diagonal

    lattice = 16 * Diagonal(ones(3))
    H  = ElementCoulomb(1)
    Li = ElementPsp(:Li, PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth"))
    atoms = [H, Li]
    positions = [
        [1/2, 1/2, 0.5953697526034847],
        [1/2, 1/2, 0.40463024613039883],
    ]

    ref = -0.02196861  # TODO source?
    γ_E = DFTK.energy_forces_ewald(lattice, charge_ionic.(atoms), positions).energy
    @test abs(γ_E - ref) < 1e-8
end

@testitem "energy_forces_ewald silicon" begin
    using DFTK
    using PseudoPotentialData

    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0]
    Si = ElementPsp(:Si, PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth"))
    atoms     = [Si, Si]
    positions = [[1/8, 1/8, 1/8], [-1/8, -1/8, -1/8]]

    ref = -8.39789357839024  # from ABINIT
    γ_E = DFTK.energy_forces_ewald(lattice, charge_ionic.(atoms), positions).energy
    @test abs(γ_E - ref) < 1e-10
end

@testitem "energy_forces_ewald cutoffs and derivatives" begin
    using DFTK
    using DFTK: Vec3, energy_forces_ewald
    using ForwardDiff
    using FiniteDifferences
    using LinearAlgebra

    lattice = [5.3 0.7 -0.2; 0.1 6.1 0.6; -0.3 0.2 7.2]
    charges = [1.0, 2.0, 3.0]
    positions = [Vec3(0.13, 0.21, 0.34), Vec3(0.57, 0.42, 0.76),
                 Vec3(0.82, 0.69, 0.15)]
    ηs = (0.2, 0.35, 0.6)
    reference = energy_forces_ewald(lattice, charges, positions; η=ηs[2])

    for η in ηs
        result = energy_forces_ewald(lattice, charges, positions; η)
        # Check that the convenience method actually passes η to the summation.
        @test result == energy_forces_ewald(Float64, lattice, charges, positions,
                                            zero(Vec3{Float64}), nothing; η)
        @test result.energy ≈ reference.energy atol=1e-12
        @test stack(result.forces) ≈ stack(reference.forces) atol=1e-11
    end
    @test norm(sum(reference.forces)) < 1e-12

    shifts = [Vec3(2, -3, 1), Vec3(-4, 1, 2), Vec3(1, 0, -2)]
    shifted = energy_forces_ewald(lattice, charges, positions .+ shifts; η=ηs[2])
    @test shifted.energy ≈ reference.energy atol=1e-12
    @test stack(shifted.forces) ≈ stack(reference.forces) atol=1e-11

    forces_ad = -ForwardDiff.gradient(vcat(positions...)) do coordinates
        T = eltype(coordinates)
        displaced = [Vec3(coordinates[3i-2:3i]) for i = 1:length(positions)]
        energy_forces_ewald(T.(lattice), charges, displaced; η=ηs[2]).energy
    end
    @test vec(stack(reference.forces)) ≈ forces_ad atol=1e-11

    # Lattice derivatives used for stresses must also be independent of the splitting.
    strain = [0.2 0.3 -0.1; 0.3 -0.4 0.2; -0.1 0.2 0.1]
    strain_energy(ε, η) = energy_forces_ewald((I + ε * strain) * lattice, charges,
                                              positions; η).energy
    derivative_ref = central_fdm(5, 1)(ε -> strain_energy(ε, ηs[2]), 0.0)
    for η in ηs
        derivative = ForwardDiff.derivative(ε -> strain_energy(ε, η), 0.0)
        @test derivative ≈ derivative_ref atol=1e-9
    end

    # Phonon displacements require complex-analytic forces, including at zero displacement.
    q = Vec3(0.2, -0.15, 1/3)
    displacement = [Vec3(0.2 + 0.1im, -0.3 + 0.2im, 0.1 - 0.4im),
                    Vec3(-0.1 + 0.3im, 0.2 - 0.1im, 0.4 + 0.2im),
                    Vec3(0.3 - 0.2im, 0.1 + 0.4im, -0.2 + 0.1im)]
    phonon_forces(ε) = stack(energy_forces_ewald(lattice, charges, positions, q,
                                                ε .* displacement; η=ηs[2]).forces)
    response = ForwardDiff.derivative(phonon_forces, 0.0)
    response_ref = central_fdm(5, 1)(phonon_forces, 0.0)
    @test response ≈ response_ref atol=1e-9
end

@testitem "energy_psp_correction silicon" begin
    using DFTK
    using PseudoPotentialData

    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0]
    Si = ElementPsp(:Si, PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth"))
    atoms     = [Si, Si]
    positions = [[1/8, 1/8, 1/8], [-1/8, -1/8, -1/8]]
    model = Model(lattice, atoms, positions; terms=[PspCorrection()])

    ref = -0.294622067023269  # from ABINIT
    e_corr = DFTK.energy_psp_correction(model)
    @test abs(e_corr - ref) < 1e-10
end
