### TODO: make sure the ForwardDiffWrappers help timings in this way, by measuring const
###       of all FD tests at once. Worry is that recompilation occurs for wach file

### ForwardDiff tests related to goemetry and symmetry perturbations. Each test is defined
### in its own explicitly named module, implementing a run_test(architecture) function. This
### allows easy testing of the same feature on CPU and GPU. @testitem are located immediatly
### below the @testmodule definition. Note that the order of the test modules in the @testitem 
### setup matters for successful compilation.
@testmodule ForceDerivatives begin
using DFTK
using Test
using LinearAlgebra
using ..TestCases: silicon
using ..ForwardDiffWrappers: tagged_derivative, tagged_gradient, tagged_jacobian

function run_test(architecture)
    function compute_force(ε1, ε2; metal=false, tol=1e-10, atoms=silicon.atoms)
        T = promote_type(typeof(ε1), typeof(ε2))
        pos = [[1.01, 1.02, 1.03] / 8, -ones(3) / 8 + ε1 * [1., 0, 0] + ε2 * [0, 1., 0]]
        if metal
            # Silicon reduced HF is metallic
            model = model_DFT(Matrix{T}(silicon.lattice), atoms, pos;
                              functionals=[], temperature=1e-3)
        else
            model = model_DFT(Matrix{T}(silicon.lattice), atoms, pos;
                              functionals=LDA())
        end
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2], architecture)

        response     = ResponseOptions(; verbose=true)
        is_converged = DFTK.ScfConvergenceForce(tol)
        scfres = self_consistent_field(basis; is_converged, response)
        compute_forces_cart(scfres)
    end

    F = compute_force(0.0, 0.0)
    derivative_ε1_fd = let ε1 = 1e-5
        (compute_force(ε1, 0.0) - F) / ε1
    end
    derivative_ε1 = tagged_derivative(ε1 -> compute_force(ε1, 0.0), 0.0)
    @test norm(derivative_ε1 - derivative_ε1_fd) < 1e-4

    derivative_ε2_fd = let ε2 = 1e-5
        (compute_force(0.0, ε2) - F) / ε2
    end
    derivative_ε2 = tagged_derivative(ε2 -> compute_force(0.0, ε2), 0.0)
    @test norm(derivative_ε2 - derivative_ε2_fd) < 1e-4

    @testset "Multiple partials" begin
        grad = tagged_gradient(v -> compute_force(v...)[1][1], [0.0, 0.0])
        @test abs(grad[1] - derivative_ε1[1][1]) < 1e-4
        @test abs(grad[2] - derivative_ε2[1][1]) < 1e-4

        jac = tagged_jacobian(v -> compute_force(v...)[1], [0.0, 0.0])
        @test norm(grad - jac[1, :]) < 1e-9
    end

    @testset "Derivative for metals" begin
        metal = true
        derivative_ε1_fd = let ε1 = 1e-5
            (compute_force(ε1, 0.0; metal) - compute_force(-ε1, 0.0; metal)) / 2ε1
        end
        derivative_ε1 = tagged_derivative(ε1 -> compute_force(ε1, 0.0; metal), 0.0)
        @test norm(derivative_ε1 - derivative_ε1_fd) < 1e-4
    end

    @testset "Using PspUpf" begin
        Si = ElementPsp(:Si, load_psp(silicon.psp_upf))
        atoms = [Si, Si]

        derivative_ε1_fd = let ε1 = 1e-5
            (compute_force(ε1, 0.0; atoms) - compute_force(-ε1, 0.0; atoms)) / 2ε1
        end
        derivative_ε1 = tagged_derivative(ε1 -> compute_force(ε1, 0.0; atoms), 0.0)
        @test norm(derivative_ε1 - derivative_ε1_fd) < 1e-4
    end

end
end

@testitem "Force derivatives using ForwardDiff" tags=[:dont_test_mpi, :minimal] #=
    =#    setup=[TestCases, ForwardDiffWrappers, ForceDerivatives] begin
    using DFTK
    ForceDerivatives.run_test(DFTK.CPU())
end

@testitem "Force derivatives using ForwardDiff (GPU)" tags=[:gpu] #=
    =#    setup=[TestCases, ForwardDiffWrappers, ForceDerivatives] begin
    using DFTK
    using CUDA
    using AMDGPU
    if CUDA.has_cuda() && CUDA.has_cuda_gpu()
        ForceDerivatives.run_test(DFTK.GPU(CuArray))
    end
    if AMDGPU.has_rocm_gpu()
        ForceDerivatives.run_test(DFTK.GPU(ROCArray))
    end
end

@testmodule StrainSensitivity begin
using DFTK
using Test
using LinearAlgebra
using ComponentArrays
using ..TestCases: aluminium
using ..ForwardDiffWrappers: tagged_derivative

function run_test(architecture)
    Ecut = 5
    kgrid = [2, 2, 2]
    model = model_DFT(aluminium.lattice, aluminium.atoms, aluminium.positions;
                      functionals=LDA(), temperature=1e-2, smearing=Smearing.Gaussian(),
                      kinetic_blowup=BlowupCHV())
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    nbandsalg = FixedBands(; n_bands_converge=10)

    function compute_properties(lattice)
        model_strained = Model(model; lattice)
        basis = PlaneWaveBasis(model_strained; Ecut, kgrid, architecture)
        scfres = self_consistent_field(basis; tol=1e-11, nbandsalg)
        ComponentArray(
           eigenvalues=stack([ev[1:10] for ev in DFTK.to_cpu(scfres.eigenvalues)]),
           ρ=DFTK.to_cpu(scfres.ρ),
           energies=collect(values(scfres.energies)),
           εF=scfres.εF,
           occupation=reduce(vcat, DFTK.to_cpu(scfres.occupation)),
        )
    end

    strain_isotropic(ε) = (1 + ε) * model.lattice
    function strain_anisotropic(ε)
        DFTK.voigt_strain_to_full([ε, 0., 0., 0., 0., 0.]) * model.lattice
    end

    @testset "$strain_fn" for strain_fn in (strain_isotropic, strain_anisotropic)
        f(ε) = compute_properties(strain_fn(ε))
        dx = tagged_derivative(f, 0.)

        h = 1e-4
        x1 = f(-h)
        x2 = f(+h)
        dx_findiff = (x2 - x1) / 2h
        @test norm(dx.ρ - dx_findiff.ρ) * sqrt(basis.dvol) < 1e-6
        @test maximum(abs, dx.eigenvalues - dx_findiff.eigenvalues) < 1e-6
        @test maximum(abs, dx.energies - dx_findiff.energies) < 3e-5
        @test dx.εF - dx_findiff.εF < 1e-6
        @test maximum(abs, dx.occupation - dx_findiff.occupation) < 1e-4
    end 
end
end

@testitem "Anisotropic strain sensitivity using ForwardDiff" tags=[:dont_test_mpi, :minimal] #=
    =#    setup=[TestCases, ForwardDiffWrappers, StrainSensitivity] begin
    using DFTK
    StrainSensitivity.run_test(DFTK.CPU())
end

@testitem "Anisotropic strain sensitivity using ForwardDiff (GPU)" tags=[:gpu] #=
    =#    setup=[TestCases, ForwardDiffWrappers, StrainSensitivity] begin
    using DFTK
    using CUDA
    using AMDGPU
    if CUDA.has_cuda() && CUDA.has_cuda_gpu()
        StrainSensitivity.run_test(DFTK.GPU(CuArray))
    end
    if AMDGPU.has_rocm_gpu()
        StrainSensitivity.run_test(DFTK.GPU(ROCArray))
    end
end

@testmodule FilterBrokenSymmetries begin
using DFTK
using Test
using LinearAlgebra
using ForwardDiff

function run_test(architecture)
    lattice = [2. 0. 0.; 0. 1. 0.; 0. 0. 1.]
    positions = [[0., 0., 0.], [0.5, 0., 0.]]
    gauss = ElementGaussian(1.0, 0.5)
    atoms = [gauss, gauss]
    atom_groups = [findall(Ref(pot) .== atoms) for pot in Set(atoms)]

    # Select some "interesting" subset of the symmetries
    # Rotation in the yz plane by 90 degrees
    rotyz = SymOp([1 0 0; 0 0 1; 0 -1 0], [0., 0., 0.])
    mirroryz = rotyz * rotyz
    # Mirror y
    mirrory = SymOp([1 0 0; 0 -1 0; 0 0 1], [0., 0., 0.])
    # Translation by 0.5 in the x direction
    transx = SymOp(diagm([1, 1, 1]), [0.5, 0., 0.])

    # Generate the full group
    symmetries_full = DFTK.complete_symop_group([rotyz, mirrory, transx])
    @test length(symmetries_full) == 16

    DFTK._check_symmetries(symmetries_full, lattice, atom_groups, positions)

    function check_symmetries(filtered_symmetries, expected_symmetries)
        expected_symmetries = DFTK.complete_symop_group(expected_symmetries)

        @test length(filtered_symmetries) == length(expected_symmetries)
        for s in expected_symmetries
            @test DFTK.is_approx_in(s, filtered_symmetries)
        end
        for s in filtered_symmetries
            @test DFTK.is_approx_in(s, expected_symmetries)
        end
    end

    # Instantiate Dual to test with perturbations
    # We need to call the `Tag` constructor to trigger the `ForwardDiff.tagcount`
    ε = ForwardDiff.Dual{typeof(ForwardDiff.Tag(Val(:mytag), Float64))}(0.0, 1.0)

    @testset "Atom movement" begin
        # Moving the second atom should break the transx symmetry, but not the others
        positions_modified = [[0., 0., 0.], [0.5 + ε, 0., 0.]]
        symmetries_filtered = DFTK.remove_dual_broken_symmetries(lattice, atoms, positions_modified, symmetries_full)

        @test length(symmetries_filtered) == 8
        check_symmetries(symmetries_filtered, [rotyz, mirrory])
    end

    @testset "Lattice strain" begin
        # Straining the lattice along z should break the rotyz symmetry, but not the others
        # In particular it should not break mirroryz which is normally generated by rotyz.
        lattice_modified = diagm([2., 1., 1. + ε])
        symmetries_filtered = DFTK.remove_dual_broken_symmetries(lattice_modified, atoms, positions, symmetries_full)

        @test length(symmetries_filtered) == 8
        check_symmetries(symmetries_filtered, [mirrory, transx, mirroryz])
    end

    @testset "Atom movement + lattice strain" begin
        # Only the mirrory and mirroryz symmetries should be left
        positions_modified = [[0., 0., 0.], [0.5 + ε, 0., 0.]]
        lattice_modified = diagm([2., 1., 1. + ε])
        symmetries_filtered = DFTK.remove_dual_broken_symmetries(lattice_modified, atoms, positions_modified, symmetries_full)

        @test length(symmetries_filtered) == 4
        check_symmetries(symmetries_filtered, [mirrory, mirroryz])
    end

    @testset "Isotropic lattice scaling" begin
        # Isotropic scaling should not break any symmetry
        lattice_modified = (1 + ε) * lattice
        symmetries_filtered = DFTK.remove_dual_broken_symmetries(lattice_modified, atoms, positions, symmetries_full)

        @test length(symmetries_filtered) == length(symmetries_full)
        check_symmetries(symmetries_filtered, symmetries_full)
    end
end
end

@testitem "Symmetries broken by perturbation are filtered out" tags=[:dont_test_mpi, :minimal] #=
    =#    setup=[FilterBrokenSymmetries] begin
    using DFTK
    FilterBrokenSymmetries.run_test(DFTK.CPU())
end

@testitem "Symmetries broken by perturbation are filtered out (GPU)" tags=[:gpu] #=
    =#    setup=[FilterBrokenSymmetries] begin
    using DFTK
    using CUDA
    using AMDGPU
    if CUDA.has_cuda() && CUDA.has_cuda_gpu()
        FilterBrokenSymmetries.run_test(DFTK.GPU(CuArray))
    end
    if AMDGPU.has_rocm_gpu()
        FilterBrokenSymmetries.run_test(DFTK.GPU(ROCArray))
    end
end

@testmodule SymmetryBreakingPerturbation begin
using DFTK
using Test
using LinearAlgebra
using ..TestCases: aluminium
using ..ForwardDiffWrappers: tagged_derivative

function run_test(architecture)
    @testset for perturbation in (:lattice, :positions)
        function run_scf(ε)
            lattice = if perturbation == :lattice
                v = ε * [0., 0., 0., 0., 0., 1.]
                DFTK.voigt_strain_to_full(v) * aluminium.lattice
            else
                # Lattice has to be a dual for position perturbations to work
                aluminium.lattice * one(typeof(ε))
            end
            pos = if perturbation == :lattice
                aluminium.positions
            else
                map(enumerate(aluminium.positions)) do (i, x)
                    i == 1 ? x + ε * [1., 0, 0] : x
                end
            end

            model = model_DFT(lattice, aluminium.atoms, pos;
                              functionals=LDA(), temperature=1e-2,
                              smearing=Smearing.Gaussian())
            basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2], architecture)
            self_consistent_field(basis; tol=1e-10)
        end

        δρ = tagged_derivative(ε -> run_scf(ε).ρ, 0.)

        h = 1e-5
        scfres1 = run_scf(-h)
        scfres2 = run_scf(+h)
        δρ_finitediff = (scfres2.ρ - scfres1.ρ) / 2h

        rtol = 1e-4
        @test norm(δρ - δρ_finitediff, 1) < rtol * norm(δρ, 1)
    end
end
end

@testitem "Symmetry-breaking perturbation using ForwardDiff" tags=[:dont_test_mpi, :minimal] #=
    =#    setup=[TestCases, ForwardDiffWrappers, SymmetryBreakingPerturbation] begin
    using DFTK
   SymmetryBreakingPerturbation.run_test(DFTK.CPU())
end

@testitem "Symmetry-breaking perturbation using ForwardDiff (GPU)" tags=[:gpu] #=
    =#    setup=[TestCases, ForwardDiffWrappers, SymmetryBreakingPerturbation] begin
    using DFTK
    using CUDA
    using AMDGPU
    if CUDA.has_cuda() && CUDA.has_cuda_gpu()
        SymmetryBreakingPerturbation.run_test(DFTK.GPU(CuArray))
    end
    if AMDGPU.has_rocm_gpu()
        SymmetryBreakingPerturbation.run_test(DFTK.GPU(ROCArray))
    end
end
