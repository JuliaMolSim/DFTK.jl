# These are not yet the best tests, but just to ensure our GPU support
# does not just break randomly
#
# TODO Refactor this file to reduce code duplication
#
# TODO Test Hamiltonian application on GPU
# TODO Direct minimisation
# TODO Float32

@testitem "GPU silicon functionality test" tags=[:gpu] setup=[TestCases] begin
    using DFTK
    using CUDA
    using AMDGPU
    using LinearAlgebra
    silicon = TestCases.silicon

    function run_problem(; architecture)
        model = model_DFT(silicon.lattice, silicon.atoms, silicon.positions;
                          functionals=PBE())
        basis = PlaneWaveBasis(model; Ecut=10, kgrid=(3, 3, 3), architecture)
        self_consistent_field(basis; tol=1e-9, solver=scf_damping_solver(damping=1.0))
    end

    # Bad copy and paste for now ... think of something more clever later
    scfres_cpu = run_problem(; architecture=DFTK.CPU())
    @testset "CUDA" begin
    if CUDA.has_cuda() && CUDA.has_cuda_gpu()
        scfres_cuda = run_problem(; architecture=DFTK.GPU(CuArray))
        @test scfres_cuda.ρ isa CuArray
        @test abs(scfres_cpu.energies.total - scfres_cuda.energies.total) < 1e-9
        @test norm(scfres_cpu.ρ - Array(scfres_cuda.ρ)) < 1e-9
        @test norm(compute_forces(scfres_cuda)) < 1e-6  # Symmetric structure
    end
    end

    @testset "AMDGPU" begin
    if AMDGPU.has_rocm_gpu()
        scfres_rocm = run_problem(; architecture=DFTK.GPU(ROCArray))
        @test scfres_rocm.ρ isa ROCArray
        @test abs(scfres_cpu.energies.total - scfres_rocm.energies.total) < 1e-9
        @test norm(scfres_cpu.ρ - Array(scfres_rocm.ρ)) < 1e-9
        @test norm(compute_forces(scfres_rocm)) < 1e-6  # Symmetric structure
    end
    end
end

@testitem "GPU iron functionality test" tags=[:gpu] setup=[TestCases] begin
    using DFTK
    using CUDA
    using AMDGPU
    using LinearAlgebra
    iron_bcc = TestCases.iron_bcc

    function run_problem(; architecture)
        magnetic_moments = [4.0]
        model = model_DFT(iron_bcc.lattice, iron_bcc.atoms, iron_bcc.positions;
                          functionals=PBE(), magnetic_moments,
                          smearing=Smearing.Gaussian(), temperature=1e-3)
        basis = PlaneWaveBasis(model; Ecut=20, kgrid=(4, 4, 4), architecture)
        ρ = guess_density(basis, magnetic_moments)

        # TODO Bump tolerance a bit here ... still leads to NaNs unfortunately
        self_consistent_field(basis; ρ, tol=1e-7, mixing=KerkerMixing())
    end

    scfres_cpu  = run_problem(; architecture=DFTK.CPU())
    @testset "CUDA" begin
    if CUDA.has_cuda() && CUDA.has_cuda_gpu()
        scfres_cuda = run_problem(; architecture=DFTK.GPU(CuArray))
        @test abs(scfres_cpu.energies.total - scfres_cuda.energies.total) < 1e-7
        @test norm(scfres_cpu.ρ - Array(scfres_cuda.ρ)) < 1e-6
        # Test that forces compute: symmetric structure, forces are zero
        @test norm(compute_forces(scfres_cpu) - compute_forces(scfres_cuda)) < 1e-9
    end
    end

    @testset "AMDGPU" begin
    if AMDGPU.has_rocm_gpu()
        scfres_rocm = run_problem(; architecture=DFTK.GPU(ROCArray))
        @test abs(scfres_cpu.energies.total - scfres_rocm.energies.total) < 1e-7
        @test norm(scfres_cpu.ρ - Array(scfres_rocm.ρ)) < 1e-6
        @test norm(compute_forces(scfres_cpu) - compute_forces(scfres_rocm)) < 1e-9
    end
    end
end

@testitem "GPU aluminium forces test" tags=[:gpu] setup=[TestCases] begin
    using DFTK
    using CUDA
    using AMDGPU
    using LinearAlgebra
    aluminium = TestCases.aluminium
    positions = copy(aluminium.positions)
    # Perturb equilibrium position for non-zero forces
    positions[1] += [0.01, 0.0, -0.01]

    function run_problem(; architecture)
        # Test with a core-corrected PSP for maximal coverage
        Al = ElementPsp(aluminium.atnum, load_psp(aluminium.psp_upf))
        atoms = fill(Al, length(aluminium.atoms))
        model = model_DFT(aluminium.lattice, atoms, positions;
                          functionals=PBE(), temperature=0.01)
        basis = PlaneWaveBasis(model; Ecut=32, kgrid=(1, 1, 1), architecture)
        self_consistent_field(basis; tol=1e-10)
    end

    scfres_cpu  = run_problem(; architecture=DFTK.CPU())
    @testset "CUDA" begin
    if CUDA.has_cuda() && CUDA.has_cuda_gpu()
        scfres_cuda = run_problem(; architecture=DFTK.GPU(CuArray))
        @test abs(scfres_cpu.energies.total - scfres_cuda.energies.total) < 1e-10
        @test norm(scfres_cpu.ρ - Array(scfres_cuda.ρ)) < 1e-8
        @test norm(compute_forces(scfres_cpu) - compute_forces(scfres_cuda)) < 1e-7
    end
    end

    @testset "AMDGPU" begin
    if AMDGPU.has_rocm_gpu()
        scfres_rocm = run_problem(; architecture=DFTK.GPU(ROCArray))
        @test abs(scfres_cpu.energies.total - scfres_rocm.energies.total) < 1e-10
        @test norm(scfres_cpu.ρ - Array(scfres_rocm.ρ)) < 1e-8
        @test norm(compute_forces(scfres_cpu) - compute_forces(scfres_rocm)) < 1e-7
    end
    end
end
