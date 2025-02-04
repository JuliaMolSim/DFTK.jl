@testitem "Force derivatives using ForwardDiff" #=
    =#    tags=[:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using ForwardDiff
    using LinearAlgebra
    silicon = TestCases.silicon

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
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2], kshift=[0, 0, 0])

        response     = ResponseOptions(; verbose=true)
        is_converged = DFTK.ScfConvergenceForce(tol)
        scfres = self_consistent_field(basis; is_converged, response)
        compute_forces_cart(scfres)
    end

    F = compute_force(0.0, 0.0)
    derivative_ε1_fd = let ε1 = 1e-5
        (compute_force(ε1, 0.0) - F) / ε1
    end
    derivative_ε1 = ForwardDiff.derivative(ε1 -> compute_force(ε1, 0.0), 0.0)
    @test norm(derivative_ε1 - derivative_ε1_fd) < 1e-4

    derivative_ε2_fd = let ε2 = 1e-5
        (compute_force(0.0, ε2) - F) / ε2
    end
    derivative_ε2 = ForwardDiff.derivative(ε2 -> compute_force(0.0, ε2), 0.0)
    @test norm(derivative_ε2 - derivative_ε2_fd) < 1e-4

    @testset "Multiple partials" begin
        grad = ForwardDiff.gradient(v -> compute_force(v...)[1][1], [0.0, 0.0])
        @test abs(grad[1] - derivative_ε1[1][1]) < 1e-4
        @test abs(grad[2] - derivative_ε2[1][1]) < 1e-4

        jac = ForwardDiff.jacobian(v -> compute_force(v...)[1], [0.0, 0.0])
        @test norm(grad - jac[1, :]) < 1e-9
    end

    @testset "Derivative for metals" begin
        metal = true
        derivative_ε1_fd = let ε1 = 1e-5
            (compute_force(ε1, 0.0; metal) - compute_force(-ε1, 0.0; metal)) / 2ε1
        end
        derivative_ε1 = ForwardDiff.derivative(ε1 -> compute_force(ε1, 0.0; metal), 0.0)
        @test norm(derivative_ε1 - derivative_ε1_fd) < 1e-4
    end

    @testset "Using PspUpf" begin
        Si = ElementPsp(:Si; psp=load_psp(silicon.psp_upf))
        atoms = [Si, Si]

        derivative_ε1_fd = let ε1 = 1e-5
            (compute_force(ε1, 0.0; atoms) - compute_force(-ε1, 0.0; atoms)) / 2ε1
        end
        derivative_ε1 = ForwardDiff.derivative(ε1 -> compute_force(ε1, 0.0; atoms), 0.0)
        @test norm(derivative_ε1 - derivative_ε1_fd) < 1e-4
    end
end

@testitem "Strain sensitivity using ForwardDiff" #=
    =#    tags=[:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using ForwardDiff
    using LinearAlgebra
    using ComponentArrays
    using PseudoPotentialData
    aluminium = TestCases.aluminium
    model = model_DFT(aluminium.lattice, aluminium.atoms, aluminium.positions;
                      functionals=LDA(), temperature=1e-2, smearing=Smearing.Gaussian())
    nbandsalg = FixedBands(; n_bands_converge=10)
    response = ResponseOptions(; verbose=true)

    function compute_properties(ε)
        model_strained = Model(model; lattice=(1 + ε) * model.lattice)
        basis = PlaneWaveBasis(model_strained; Ecut=5, kgrid=[2, 2, 2])
        scfres = self_consistent_field(basis; tol=1e-10, nbandsalg, response)
        ComponentArray(
           eigenvalues=stack([ev[1:10] for ev in scfres.eigenvalues]),
           ρ=scfres.ρ,
           energies=collect(values(scfres.energies)),
           εF=scfres.εF,
           occupation=reduce(vcat, scfres.occupation),
        )
    end

    dx = ForwardDiff.derivative(compute_properties, 0.)

    h = 1e-4
    x1 = compute_properties(-h)
    x2 = compute_properties(+h)
    dx_findiff = (x2 - x1) / 2h
    @test norm(dx.ρ - dx_findiff.ρ) < 1e-8
    @test maximum(abs, dx.eigenvalues - dx_findiff.eigenvalues) < 1e-8
    @test maximum(abs, dx.energies - dx_findiff.energies) < 1e-8
    @test dx.εF - dx_findiff.εF < 1e-8
    @test dx.occupation - dx_findiff.occupation < 1e-6
end

@testitem "scfres PSP sensitivity using ForwardDiff" #=
    =#    tags=[:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using ForwardDiff
    using LinearAlgebra
    using ComponentArrays
    using PseudoPotentialData
    aluminium = TestCases.aluminium

    function compute_band_energies(ε::T) where {T}
        psp  = load_psp(PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth"), :Al)
        rloc = convert(T, psp.rloc)

        pspmod = PspHgh(psp.Zion, rloc,
                        psp.cloc, psp.rp .+ [0, ε], psp.h;
                        psp.identifier, psp.description)
        atoms = fill(ElementPsp(aluminium.atnum; psp=pspmod), length(aluminium.positions))
        model = model_DFT(Matrix{T}(aluminium.lattice), atoms, aluminium.positions;
                          functionals=LDA(), temperature=1e-2, smearing=Smearing.Gaussian())
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2], kshift=[0, 0, 0])

        is_converged = DFTK.ScfConvergenceDensity(1e-10)
        scfres = self_consistent_field(basis; is_converged, mixing=KerkerMixing(),
                                       nbandsalg=FixedBands(; n_bands_converge=10),
                                       damping=0.6, response=ResponseOptions(; verbose=true))

        ComponentArray(
           eigenvalues=stack([ev[1:10] for ev in scfres.eigenvalues]),
           ρ=scfres.ρ,
           energies=collect(values(scfres.energies)),
           εF=scfres.εF,
           occupation=reduce(vcat, scfres.occupation),
        )
    end

    derivative_ε = let ε = 1e-4
        (compute_band_energies(ε) - compute_band_energies(-ε)) / 2ε
    end
    derivative_fd = ForwardDiff.derivative(compute_band_energies, 0.0)
    @test norm(derivative_fd - derivative_ε) < 5e-4
end

@testitem "Functional force sensitivity using ForwardDiff" #=
    =#    tags=[:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using ForwardDiff
    using LinearAlgebra
    using ComponentArrays
    using DftFunctionals
    silicon = TestCases.silicon

    function compute_force(ε1::T) where {T}
        pos = [[1.01, 1.02, 1.03] / 8, -ones(3) / 8]
        pbec = DftFunctional(:gga_c_pbe)
        pbex = DftFunctional(:gga_x_pbe)
        pbex = change_parameters(pbex, parameters(pbex) + ComponentArray(κ=0, μ=ε1))

        model = model_DFT(Matrix{T}(silicon.lattice), silicon.atoms, pos;
                          functionals=[pbex, pbec])
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2], kshift=[0, 0, 0])

        is_converged = DFTK.ScfConvergenceDensity(1e-10)
        scfres = self_consistent_field(basis; is_converged,
                                       response=ResponseOptions(; verbose=true))
        compute_forces_cart(scfres)
    end

    derivative_ε = let ε = 1e-5
        (compute_force(ε) - compute_force(-ε)) / 2ε
    end
    derivative_fd = ForwardDiff.derivative(compute_force, 0.0)
    @test norm(derivative_ε - derivative_fd) < 1e-4
end

@testitem "Derivative of complex function" tags=[:dont_test_mpi] begin
    using DFTK
    using ForwardDiff
    using LinearAlgebra
    using SpecialFunctions, FiniteDifferences

    α = randn(ComplexF64)
    erfcα = x -> erfc(α * x)

    x0  = randn()
    fd1 = ForwardDiff.derivative(erfcα, x0)
    fd2 = FiniteDifferences.central_fdm(5, 1)(erfcα, x0)
    @test norm(fd1 - fd2) < 1e-8
end

@testitem "Higher derivatives of Fermi-Dirac occupation" tags=[:dont_test_mpi] begin
    using DFTK
    using ForwardDiff

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

@testitem "LocalNonlinearity sensitivity using ForwardDiff" tags=[:dont_test_mpi] begin
    using DFTK
    using ForwardDiff
    using LinearAlgebra

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
        basis = PlaneWaveBasis(model; Ecut=500, kgrid=(1, 1, 1))
        ρ = zeros(Float64, basis.fft_size..., 1)
        is_converged = DFTK.ScfConvergenceDensity(1e-10)
        scfres = self_consistent_field(basis; ρ, is_converged,
                                       response=ResponseOptions(; verbose=true))
        compute_forces_cart(scfres)
    end
    derivative_ε = let ε = 1e-5
        (compute_force(ε) - compute_force(-ε)) / 2ε
    end
    derivative_fd = ForwardDiff.derivative(compute_force, 0.0)
    @test norm(derivative_ε - derivative_fd) < 1e-4
end
