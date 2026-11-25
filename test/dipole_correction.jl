@testitem "Bengtsson dipole correction" tags=[:minimal] begin
    using DFTK
    using ForwardDiff
    using Test
    using Unitful
    using LinearAlgebra
    using PseudoPotentialData
    using DFTK: _dipole_correction_sawtooth

    function frozen_problem(direction=3; zpositions=[[0.30], [0.60]], kwargs...)
        lengths = [6.0, 7.0, 60.0]
        if direction == 1
            lengths = lengths[[3, 1, 2]]
        elseif direction == 2
            lengths = lengths[[1, 3, 2]]
        end
        lattice = Matrix(Diagonal(lengths))

        atoms = [ElementCoulomb(:H), ElementCoulomb(:H)]
        positions = map(only.(zpositions)) do z
            position = [0.5, 0.5, z]
            if direction == 1
                position = position[[3, 1, 2]]
            elseif direction == 2
                position = position[[1, 3, 2]]
            end
            position
        end
        model = Model(lattice, atoms, positions;
                      terms=[DipoleCorrection(direction; kwargs...)], symmetries=false)
        fft_size = [8, 8, 8]
        fft_size[direction] = 64
        basis = PlaneWaveBasis(model; Ecut=2, kgrid=[1, 1, 1], fft_size=Tuple(fft_size))
        ρ = fill(2 / model.unit_cell_volume, basis.fft_size..., 1)
        (; model, basis, ρ)
    end

    @testset "Construction" begin
        correction = DipoleCorrection(2; discontinuity_position=0.2,
                                      smoothening_width=1u"Å")
        @test correction.direction == 2
        @test correction.discontinuity_position == 0.2
        @test correction.smoothening_width == DFTK.austrip(1u"Å")
        @test_throws ArgumentError DipoleCorrection(0)
        @test_throws ArgumentError DipoleCorrection(; smoothening_width=-1)
        @test_throws ArgumentError DipoleCorrection(; discontinuity_position=NaN)
    end

    @testset "Sawtooth properties" begin
        halfwidth = 0.08
        for x in range(-0.5, 1.5; length=101)
            @test _dipole_correction_sawtooth(x, halfwidth) ≈
                  _dipole_correction_sawtooth(x + 1, halfwidth) atol=1e-14
            @test _dipole_correction_sawtooth(x, halfwidth) + _dipole_correction_sawtooth(1 - x, halfwidth) ≈ 0 atol=1e-14
        end
        sawtooth = _dipole_correction_sawtooth.(range(0, 1; length=129)[1:end-1], halfwidth)
        @test sum(sawtooth) ≈ 0 atol=1e-14
        @test _dipole_correction_sawtooth(0, halfwidth) == 0
    end

    @testset "Properties on toy model" begin
        (; model, basis, ρ) = frozen_problem()
        term = only(basis.terms)
        eh = DFTK.ene_ops(term, basis, nothing, nothing; ρ)
        L = norm(model.lattice[:, 3])
        Ω = model.unit_cell_volume
        # The uniform electronic density integrates to zero against the zero-mean
        # sawtooth. Sharp-sawtooth values apply at both ionic positions, so the
        # total dipole moment (electronic minus ionic) is p = L * 0.1.
        p = L * 0.1
        @test eh.E ≈ 2π * p^2 / Ω atol=2e-13
        expected_potential = 4π * p / Ω .* term.sawtooth_scaled
        @test only(eh.ops).potential ≈ expected_potential atol=2e-13

        symmetric = frozen_problem(; zpositions=[[0.40], [0.60]])
        eh_symmetric = DFTK.ene_ops(only(symmetric.basis.terms), symmetric.basis,
                                    nothing, nothing; symmetric.ρ)
        @test eh_symmetric.E ≈ 0 atol=1e-14
        @test maximum(abs, only(eh_symmetric.ops).potential) ≈ 0 atol=1e-14

        mirrored = frozen_problem(; zpositions=[[0.70], [0.40]])
        eh_mirrored = DFTK.ene_ops(only(mirrored.basis.terms), mirrored.basis,
                                   nothing, nothing; mirrored.ρ)
        @test eh_mirrored.E ≈ eh.E atol=2e-13
        # Mirror symmetry: the total dipole moment (electronic minus ionic) flips sign.
        mirrored_term = only(mirrored.basis.terms)
        mirrored_p = (sum(total_density(mirrored.ρ) .* mirrored_term.sawtooth_scaled)
                      * mirrored.basis.dvol - mirrored_term.ionic_dipole_moment)
        @test mirrored_p ≈ -p atol=2e-13

        for direction = 1:3
            axial = frozen_problem(direction)
            eh_axial = DFTK.ene_ops(only(axial.basis.terms), axial.basis,
                                    nothing, nothing; axial.ρ)
            @test eh_axial.E ≈ eh.E atol=2e-13
        end
    end

    @testset "Argument errors when instantiating" begin
        (; model, basis, ρ) = frozen_problem()
        skew_lattice = Matrix(model.lattice)
        skew_lattice[1, 3] = 1
        skew_model = Model(skew_lattice, model.atoms, model.positions;
                           terms=[DipoleCorrection()], symmetries=false)
        @test_throws ArgumentError PlaneWaveBasis(skew_model; Ecut=2, kgrid=[1, 1, 1])

        smoothening_width = 2norm(model.lattice[:, 3])
        wide_model = Model(model.lattice, model.atoms, model.positions;
                           terms=[DipoleCorrection(; smoothening_width)],
                           symmetries=false)
        @test_throws ArgumentError PlaneWaveBasis(wide_model; Ecut=2, kgrid=[1, 1, 1])

        charged_model = Model(model; n_electrons=model.n_electrons - 1,
                              disable_electrostatics_check=true)
        @test_throws ArgumentError PlaneWaveBasis(charged_model; Ecut=2, kgrid=[1, 1, 1])
    end

    @testset "Test forces" begin
        (; model, basis, ρ) = frozen_problem()
        term = only(basis.terms)
        forces = DFTK.compute_forces(term, basis, nothing, nothing; ρ)
        displacement = [0.03, -0.02, 0.05]
        function position_energy(ε)
            positions = copy(model.positions)
            positions[1] += ε * displacement
            new_model = Model(model; positions, symmetries=false)
            new_basis = PlaneWaveBasis(new_model; Ecut=2, kgrid=[1, 1, 1], basis.fft_size)
            DFTK.ene_ops(only(new_basis.terms), new_basis, nothing, nothing; ρ).E
        end
        ε = 1e-6
        reference = -(position_energy(ε) - position_energy(-ε)) / 2ε
        @test dot(displacement, forces[1]) ≈ reference atol=1e-10
        @test forces[2] ≈ forces[1] atol=1e-14
    end
end

@testitem "Bengtsson dipole single versus mirrored water layers" tags=[:slow] begin
    using DFTK
    using LinearAlgebra
    using PseudoPotentialData
    using Statistics
    using Test
    using Unitful

    lateral = 6.0
    height = 30.0
    lattice = Matrix(Diagonal([lateral, lateral, height]))
    bond_length = DFTK.austrip(0.95858u"Å")
    half_angle = deg2rad(104.45) / 2
    bond_y = bond_length * sin(half_angle)
    bond_z = bond_length * cos(half_angle)
    center = [lateral, lateral, height] / 2
    cartesian_positions = [center,
                           center + [0, bond_y, -bond_z],
                           center + [0, -bond_y, -bond_z]]
    positions = [r ./ [lateral, lateral, height] for r in cartesian_positions]
    pseudopotentials = PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth")
    atoms = [ElementPsp(:O, pseudopotentials),
             ElementPsp(:H, pseudopotentials),
             ElementPsp(:H, pseudopotentials)]

    model = model_DFT(lattice, atoms, positions;
                      symmetries=false, functionals=LDA(),
                      extra_terms=[DipoleCorrection()])
    basis = PlaneWaveBasis(model; Ecut=20, kgrid=[1, 1, 1])
    scfres = self_consistent_field(basis; tol=1e-8)

    # Use two oppositely oriented layers in a doubled cell as a dipole-free reference.
    double_positions = vcat([r ./ [1, 1, 2] for r in positions],
                            [[r[1], r[2], 1 - r[3] / 2] for r in positions])
    double_lattice = Matrix(Diagonal([lateral, lateral, 2height]))
    double_model = model_DFT(double_lattice, repeat(atoms, 2), double_positions;
                             functionals=LDA(), symmetries=false)
    double_basis = PlaneWaveBasis(double_model; Ecut=20, kgrid=[1, 1, 1])
    double_scfres = self_consistent_field(double_basis; tol=1e-8)

    corrected_error = scfres.energies.total - double_scfres.energies.total / 2
    @test abs(corrected_error) < 8e-4
end

@testitem "GPU Bengtsson dipole correction" tags=[:gpu] begin
    using DFTK
    using CUDA
    using AMDGPU
    using LinearAlgebra

    lattice = Matrix(Diagonal([6.0, 7.0, 60.0]))
    atoms = [ElementCoulomb(:H), ElementCoulomb(:H)]
    positions = [[0.5, 0.5, 0.30], [0.5, 0.5, 0.60]]
    model = Model(lattice, atoms, positions;
                  terms=[DipoleCorrection()], symmetries=false)
    ρ_cpu = fill(2 / model.unit_cell_volume, (8, 8, 64)..., 1)
    δρ_cpu = randn(size(ρ_cpu)) / model.unit_cell_volume

    function run_problem(architecture)
        basis = PlaneWaveBasis(model; Ecut=2, kgrid=[1, 1, 1], fft_size=(8, 8, 64),
                               architecture)
        term = only(basis.terms)
        ρ = DFTK.to_device(architecture, ρ_cpu)
        δρ = DFTK.to_device(architecture, δρ_cpu)
        eh = DFTK.ene_ops(term, basis, nothing, nothing; ρ)
        (; E=eh.E, potential=only(eh.ops).potential,
         δV=DFTK.apply_kernel(term, basis, δρ).δVρ,
         forces=DFTK.compute_forces(term, basis, nothing, nothing; ρ))
    end

    cpu = run_problem(DFTK.CPU())
    function test_against_cpu(gpu)
        @test gpu.E ≈ cpu.E rtol=1e-12
        @test norm(DFTK.to_cpu(gpu.potential) - cpu.potential) < 1e-12
        @test norm(DFTK.to_cpu(gpu.δV) - cpu.δV) < 1e-12
        @test norm(stack(gpu.forces) - stack(cpu.forces)) < 1e-12
    end

    @testset "CUDA" begin
        if CUDA.has_cuda() && CUDA.has_cuda_gpu()
            test_against_cpu(run_problem(DFTK.GPU(CuArray)))
        end
    end
    @testset "AMDGPU" begin
        if AMDGPU.has_rocm_gpu()
            test_against_cpu(run_problem(DFTK.GPU(ROCArray)))
        end
    end
end
