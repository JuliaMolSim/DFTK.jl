### ForwardDiff tests related to perturbations of model parameters. Each test is defined
### in its own explicitly named module, implementing a run_test(architecture) function. This
### allows easy testing of the same feature on CPU and GPU. @testitem are located immediatly
### below the @testmodule definition. Note that the order of the test modules in the @testitem 
### setup matters for successful compilation. 

@testmodule PspSensitivity begin
using DFTK
using Test
using LinearAlgebra
using PseudoPotentialData
using ComponentArrays
using ..TestCases: aluminium
using ..ForwardDiffWrappers: tagged_derivative

function run_test(; architecture)
    function compute_band_energies(ε::T) where {T}
        psp  = load_psp(PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth"), :Al)
        rloc = convert(T, psp.rloc)

        pspmod = PspHgh(psp.Zion, rloc,
                        psp.cloc, psp.rp .+ [0, ε], psp.h;
                        psp.identifier, psp.description)
        atoms = fill(ElementPsp(aluminium.atnum; psp=pspmod), length(aluminium.positions))
        model = model_DFT(Matrix{T}(aluminium.lattice), atoms, aluminium.positions;
                          functionals=LDA(), temperature=1e-2, smearing=Smearing.Gaussian())
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2], architecture)

        is_converged = DFTK.ScfConvergenceDensity(1e-10)
        scfres = self_consistent_field(basis; is_converged, mixing=KerkerMixing(),
                                       nbandsalg=FixedBands(; n_bands_converge=10),
                                       damping=0.6, response=ResponseOptions(; verbose=true))

        ComponentArray(
           eigenvalues=stack([ev[1:10] for ev in DFTK.to_cpu(scfres.eigenvalues)]),
           ρ=DFTK.to_cpu(scfres.ρ),
           energies=collect(values(scfres.energies)),
           εF=scfres.εF,
           occupation=reduce(vcat, DFTK.to_cpu(scfres.occupation)),
        )
    end

    derivative_ε = let ε = 1e-4
        (compute_band_energies(ε) - compute_band_energies(-ε)) / 2ε
    end
    derivative_fd = tagged_derivative(compute_band_energies, 0.0)
    @test norm(derivative_fd - derivative_ε) < 5e-4
end
end

@testitem "scfres PSP sensitivity using ForwardDiff" tags=[:dont_test_mpi, :minimal] #=
    =#    setup=[TestCases, ForwardDiffWrappers, PspSensitivity] begin
    using DFTK
    PspSensitivity.run_test(; architecture=DFTK.CPU())
end

@testitem "scfres PSP sensitivity using ForwardDiff (GPU)" tags=[:gpu] #=
    =#    setup=[TestCases, ForwardDiffWrappers, PspSensitivity] begin
    using DFTK
    using CUDA
    using AMDGPU
    if CUDA.has_cuda() && CUDA.has_cuda_gpu()
        PspSensitivity.run_test(; architecture=DFTK.GPU(CuArray))
    end
    if AMDGPU.has_rocm_gpu()
        PspSensitivity.run_test(; architecture=DFTK.GPU(ROCArray))
    end
end

@testmodule ForceSensitivity begin
using DFTK
using Test
using LinearAlgebra
using ComponentArrays
using DftFunctionals
using ..TestCases: silicon
using ..ForwardDiffWrappers: tagged_derivative

function run_test(; architecture)
    function compute_force(ε1::T) where {T}
        pos = [[1.01, 1.02, 1.03] / 8, -ones(3) / 8]
        pbec = DftFunctional(:gga_c_pbe)
        pbex = DftFunctional(:gga_x_pbe)
        pbex = change_parameters(pbex, parameters(pbex) + ComponentArray(κ=0, μ=ε1))

        model = model_DFT(Matrix{T}(silicon.lattice), silicon.atoms, pos;
                          functionals=[pbex, pbec])
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2], architecture)

        is_converged = DFTK.ScfConvergenceDensity(1e-10)
        scfres = self_consistent_field(basis; is_converged,
                                       response=ResponseOptions(; verbose=true))
        compute_forces_cart(scfres)
    end

    derivative_ε = let ε = 1e-5
        (compute_force(ε) - compute_force(-ε)) / 2ε
    end
    derivative_fd = tagged_derivative(compute_force, 0.0)
    @test norm(derivative_ε - derivative_fd) < 1e-4
end
end

@testitem "Functional force sensitivity using ForwardDiff" tags=[:dont_test_mpi, :minimal] #=
    =#    setup=[TestCases, ForwardDiffWrappers, ForceSensitivity] begin
    using DFTK
    ForceSensitivity.run_test(; architecture=DFTK.CPU())
end

@testitem "Functional force sensitivity using ForwardDiff (GPU)" tags=[:gpu] #=
    =#    setup=[TestCases, ForwardDiffWrappers, ForceSensitivity] begin
    using DFTK
    using CUDA
    using AMDGPU
    if CUDA.has_cuda() && CUDA.has_cuda_gpu()
        ForceSensitivity.run_test(; architecture=DFTK.GPU(CuArray))
    end
    if AMDGPU.has_rocm_gpu()
        ForceSensitivity.run_test(; architecture=DFTK.GPU(ROCArray))
    end
end

@testmodule LocalNonlinearitySensitivity begin
using DFTK
using Test
using LinearAlgebra
using ..ForwardDiffWrappers: tagged_derivative

function run_test(; architecture)
    function compute_force(ε::T) where {T}
        # solve the 1D Gross-Pitaevskii equation with ElementGaussian potential
        lattice = 10.0 .* [[1 0 0.]; [0 0 0]; [0 0 0]]
        positions = [[0.2, 0, 0], [0.8, 0, 0]]
        gauss = ElementGaussian(1.0, 0.5)
        atoms = [gauss, gauss]
        n_electrons = 1
        terms = [Kinetic(), AtomicLocal(), LocalNonlinearity(ρ -> (1.0 + ε) * ρ^2)]
        model = Model(Matrix{T}(lattice), atoms, positions;
                      n_electrons, terms, spin_polarization=:spinless)
        basis = PlaneWaveBasis(model; Ecut=500, kgrid=(1, 1, 1), architecture)
        ρ = DFTK.to_device(architecture, zeros(Float64, basis.fft_size..., 1))
        is_converged = DFTK.ScfConvergenceDensity(1e-10)
        scfres = self_consistent_field(basis; ρ, is_converged,
                                       response=ResponseOptions(; verbose=true))
        compute_forces_cart(scfres)
    end
    derivative_ε = let ε = 1e-5
        (compute_force(ε) - compute_force(-ε)) / 2ε
    end
    derivative_fd = tagged_derivative(compute_force, 0.0)
    @test norm(derivative_ε - derivative_fd) < 1e-4
end
end

@testitem "LocalNonlinearity sensitivity using ForwardDiff" tags=[:dont_test_mpi, :minimal] #=
    =#    setup=[ForwardDiffWrappers, LocalNonlinearitySensitivity] begin
    using DFTK
    LocalNonlinearitySensitivity.run_test(; architecture=DFTK.CPU())
end

@testitem "LocalNonlinearity sensitivity using ForwardDiff (GPU)" tags=[:gpu] #=
    =#    setup=[ForwardDiffWrappers, LocalNonlinearitySensitivity] begin
    using DFTK
    using CUDA
    using AMDGPU
    if CUDA.has_cuda() && CUDA.has_cuda_gpu()
        LocalNonlinearitySensitivity.run_test(; architecture=DFTK.GPU(CuArray))
    end
    if AMDGPU.has_rocm_gpu()
        LocalNonlinearitySensitivity.run_test(; architecture=DFTK.GPU(ROCArray))
    end
end

@testmodule ScfresParameterConsistency begin
using DFTK
using Test
using ForwardDiff
using ..TestCases: silicon
function run_test(; architecture)
    # Make silicon primal model
    model = model_DFT(silicon.lattice, silicon.atoms, silicon.positions;
                      functionals=LDA(), temperature=1e-3, smearing=Smearing.Gaussian())
    
    # Make silicon dual model
    # We need to call the `Tag` constructor to trigger the `ForwardDiff.tagcount`
    T = typeof(ForwardDiff.Tag(Val(:mytag), Float64))
    x_dual = ForwardDiff.Dual{T}(1.0, 1.0)
    model_dual = Model(model; lattice=x_dual * model.lattice)

    # Construct the primal and dual basis
    basis = PlaneWaveBasis(model; Ecut=5, kgrid=(1,1,1), architecture)
    basis_dual = PlaneWaveBasis(model_dual; Ecut=5, kgrid=(1,1,1), architecture)

    # Compute scfres with primal and dual basis
    scfres = self_consistent_field(basis; tol=1e-5)
    scfres_dual = self_consistent_field(basis_dual; tol=1e-5)
    
    # Check that scfres_dual has the same parameters as scfres
    @test isempty(setdiff(keys(scfres), keys(scfres_dual)))
end
end

@testitem "Test scfres dual has the same params as scfres primal" tags=[:dont_test_mpi, :minimal] #=
    =#    setup=[TestCases, ScfresParameterConsistency] begin
    using DFTK
    ScfresParameterConsistency.run_test(; architecture=DFTK.CPU())
end

@testitem "Test scfres dual has the same params as scfres primal (GPU)" tags=[:gpu] #=
    =#    setup=[TestCases, ScfresParameterConsistency] begin
    using DFTK
    using CUDA
    using AMDGPU
    if CUDA.has_cuda() && CUDA.has_cuda_gpu()
        ScfresParameterConsistency.run_test(; architecture=DFTK.GPU(CuArray))
    end
    if AMDGPU.has_rocm_gpu()
        ScfresParameterConsistency.run_test(; architecture=DFTK.GPU(ROCArray))
    end
end

@testmodule TemperatureSensitivity begin
using DFTK
using Test
using LinearAlgebra
using PseudoPotentialData
using ..ForwardDiffWrappers: tagged_derivative
function run_test(; architecture)
    a = 10.26  # Silicon lattice constant in Bohr
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    Si = ElementPsp(:Si, PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
    atoms     = [Si, Si]
    positions = [ones(3)/8, -ones(3)/8]

    function get(T)
        model = model_DFT(lattice, atoms, positions; functionals=LDA(), temperature=T)
        basis = PlaneWaveBasis(model; Ecut=10, kgrid=[1, 1, 1], architecture)
        scfres = self_consistent_field(basis, tol=1e-12)
        scfres.energies.Entropy
    end
    T0 = .01
    derivative_ε = let ε = 1e-5
        (get(T0+ε) - get(T0-ε)) / 2ε
    end
    derivative_fd = tagged_derivative(get, T0)
    @test norm(derivative_ε - derivative_fd) < 1e-4
end
end

@testitem "ForwardDiff wrt temperature" tags=[:dont_test_mpi, :minimal] #=
    =#    setup=[ForwardDiffWrappers, TemperatureSensitivity] begin
    using DFTK
    TemperatureSensitivity.run_test(; architecture=DFTK.CPU())
end

@testitem "ForwardDiff wrt temperature (GPU)" tags=[:gpu] #=
    =#    setup=[ForwardDiffWrappers, TemperatureSensitivity] begin
    using DFTK
    using CUDA
    using AMDGPU
    if CUDA.has_cuda() && CUDA.has_cuda_gpu()
        TemperatureSensitivity.run_test(; architecture=DFTK.GPU(CuArray))
    end
    if AMDGPU.has_rocm_gpu()
        TemperatureSensitivity.run_test(; architecture=DFTK.GPU(ROCArray))
    end
end