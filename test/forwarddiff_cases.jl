@testmodule ForwardDiffCases begin
using DFTK
using Test
using ForwardDiff
using LinearAlgebra
using ComponentArrays
using PseudoPotentialData
using DftFunctionals
using FiniteDifferences
using SpecialFunctions
using ..TestCases: silicon, aluminium

# Wrappers around ForwardDiff to fix tags and reduce compilation time.
# Warning: fixing types without care can lead to perturbation confusion,
# this should only be used within the testing framework. Risk of perturbation
# confusion arises when nested derivatives of different functions are taken
# with a fixed tag. Only use these wrappers at the top-level call.
struct DerivativeTag end
function tagged_derivative(f, x::T; custom_tag=DerivativeTag) where T
    # explicit call to ForwardDiff.Tag() to trigger ForwardDiff.tagcount
    TagType = typeof(ForwardDiff.Tag(custom_tag(), T))
    x_dual = ForwardDiff.Dual{TagType, T, 1}(x, ForwardDiff.Partials((one(T),)))

    res = ForwardDiff.extract_derivative(TagType, f(x_dual))
    return res
end

struct GradientTag end
GradientConfig(f, x, ::Type{Tag}) where {Tag} =
    ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk(x), Tag())
function tagged_gradient(f, x::AbstractArray{T}; custom_tag=GradientTag) where T
    # explicit call to ForwardDiff.Tag() to trigger ForwardDiff.tagcount
    TagType = typeof(ForwardDiff.Tag(custom_tag(), T))

    cfg = GradientConfig(f, x, TagType)
    ForwardDiff.gradient(f, x, cfg, Val{false}())
end

struct JacobianTag end
JacobianConfig(f, x, ::Type{Tag}) where {Tag} =
    ForwardDiff.JacobianConfig(f, x, ForwardDiff.Chunk(x), Tag())
function tagged_jacobian(f, x::AbstractArray{T}; custom_tag=JacobianTag) where T
    # explicit call to ForwardDiff.Tag() to trigger ForwardDiff.tagcount
    TagType = typeof(ForwardDiff.Tag(custom_tag(), T))

    cfg = JacobianConfig(f, x, TagType)
    ForwardDiff.jacobian(f, x, cfg, Val{false}())
end

function test_FD_force_derivatives(architecture)
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
        Si = ElementPsp(:Si; psp=load_psp(silicon.psp_upf))
        atoms = [Si, Si]

        derivative_ε1_fd = let ε1 = 1e-5
            (compute_force(ε1, 0.0; atoms) - compute_force(-ε1, 0.0; atoms)) / 2ε1
        end
        derivative_ε1 = tagged_derivative(ε1 -> compute_force(ε1, 0.0; atoms), 0.0)
        @test norm(derivative_ε1 - derivative_ε1_fd) < 1e-4
    end

end

function test_FD_strain_sensitivity(architecture)
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

function test_FD_psp_sensitivity(architecture)
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

function test_FD_force_sensitivity(architecture)
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

function test_FD_local_nonlinearity_sensitivity(architecture)
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

function test_FD_filter_broken_symmetries(architecture)
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

function test_FD_symmetry_breaking_perturbation(architecture)
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

function test_FD_scfres_parameter_consistency(architecture)
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

function test_FD_wrt_temperature(architecture)
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

### Generic ForwardDiff tests (independent of DFTK architecture)
function test_FD_complex_function_derivative()
    α = randn(ComplexF64)
    erfcα = x -> erfc(α * x)

    x0  = randn()
    fd1 = tagged_derivative(erfcα, x0)
    fd2 = FiniteDifferences.central_fdm(5, 1)(erfcα, x0)
    @test norm(fd1 - fd2) < 1e-8
end

function test_FD_fermi_dirac_higher_derivatives()
    smearing = Smearing.FermiDirac()
    f(x) = Smearing.occupation(smearing, x)

    function compute_nth_derivative(n, f, x)
        (n == 0) && return f(x)
        ForwardDiff.derivative(x -> compute_nth_derivative(n - 1, f, x), x)
    end

    @testset "Avoid NaN from exp-overflow for large x" begin
        T = Float64
        x = log(floatmax(T)) / 2 + 1
        for n in 0:8
            @testset "Derivative order $n" begin
                y = compute_nth_derivative(n, f, x)
                @test isfinite(y)
                @test abs(y) ≤ eps(T)
            end
        end
    end
end


end