@testitem "Kernels" setup=[TestCases] begin
    using DFTK
    using LinearAlgebra

    function test_kernel(spin_polarization, termtype; test_compute=true, psp=TestCases.silicon.psp_gth)
        kgrid  = MonkhorstPack([2, 2, 2]; kshift = ones(3) / 2)
        testcase = TestCases.silicon
        Si = ElementPsp(TestCases.silicon.atnum, load_psp(psp))
        atoms = [Si, Si]
        ε   = 1e-8
        tol = 1e-5

        xcsym = (termtype isa Xc) ? join(string.(termtype.functionals), " ") : ""
        @testset "Kernel $(typeof(termtype)) $xcsym ($spin_polarization) $(psp)" begin
            magnetic_moments = []
            n_spin = 1
            if spin_polarization == :collinear
                magnetic_moments = 2rand(2)
                n_spin = 2
            end

            model = Model(testcase.lattice, atoms, testcase.positions;
                          terms=[termtype], magnetic_moments, spin_polarization)
            @test model.n_spin_components == n_spin
            basis = PlaneWaveBasis(model; Ecut=2, kgrid)
            term  = only(basis.terms)

            ρ0 = guess_density(basis, magnetic_moments)
            δρ = randn(size(ρ0))
            ρ_minus     = ρ0 - ε * δρ
            ρ_plus      = ρ0 + ε * δρ
            ops_minus = DFTK.ene_ops(term, basis, nothing, nothing; ρ=ρ_minus).ops
            ops_plus  = DFTK.ene_ops(term, basis, nothing, nothing; ρ=ρ_plus).ops
            δV = zero(ρ0)

            for iσ = 1:model.n_spin_components
                # Index of the first spin-up or spin-down k-point
                ifirst = first(DFTK.krange_spin(basis, iσ))
                δV[:, :, :, iσ] = (ops_plus[ifirst].potential - ops_minus[ifirst].potential) / (2ε)
            end

            δV_apply = DFTK.apply_kernel(term, basis, δρ; ρ=ρ0)
            @test norm(δV - δV_apply) < tol
            if test_compute
                kernel = DFTK.compute_kernel(term, basis; ρ=ρ0)
                δV_matrix = reshape(kernel * vec(δρ), size(δρ))
                @test norm(δV - δV_matrix) < tol
            end

            @testset "Self-adjointness" begin
                δρ2 = randn(size(ρ0))
                left  = dot(δρ, DFTK.apply_kernel(term, basis, δρ2; ρ=ρ0)) * basis.dvol
                right = dot(DFTK.apply_kernel(term, basis, δρ; ρ=ρ0), δρ2) * basis.dvol
                @test isapprox(left, right; atol=1e-11)
            end
        end
    end


    function test_kernel_collinear_vs_noncollinear(termtype)
        Ecut = 2
        kgrid  = MonkhorstPack([2, 2, 2]; kshift = ones(3) / 2)
        testcase = TestCases.silicon

        xcsym = (termtype isa Xc) ? join(string.(termtype.functionals), " ") : ""
        @testset "Kernel $(typeof(termtype)) $xcsym (coll == noncoll)" begin
            model = Model(testcase.lattice, testcase.atoms, testcase.positions;
                          terms=[termtype])
            basis = PlaneWaveBasis(model; Ecut, kgrid)
            term  = only(basis.terms)

            model_col = Model(testcase.lattice, testcase.atoms, testcase.positions;
                              terms=[termtype], spin_polarization=:collinear)
            basis_col = PlaneWaveBasis(model_col; Ecut, kgrid)
            term_col  = only(basis_col.terms)

            ρ0 = guess_density(basis)
            δρ = randn(size(ρ0))
            δV = DFTK.apply_kernel(term, basis, δρ; ρ=ρ0)

            ρ0_col = cat(0.5ρ0, 0.5ρ0, dims=4)
            δρ_col = cat(0.5δρ, 0.5δρ, dims=4)
            δV_pol = DFTK.apply_kernel(term_col, basis_col, δρ_col; ρ=ρ0_col)

            @test norm(δV_pol[:, :, :, 1] - δV_pol[:, :, :, 2]) < 1e-12
            @test norm(δV - δV_pol[:, :, :, 1:1]) < 1e-11
        end
    end

    test_kernel(:none, LocalNonlinearity(ρ -> 1.2 * ρ^2))
    test_kernel(:none, Hartree())
    test_kernel(:none, Xc([:lda_xc_teter93]))
    test_kernel(:none, Xc([:gga_c_pbe]), test_compute=false)
    test_kernel(:none, Xc([:gga_x_pbe]), test_compute=false)

    test_kernel_collinear_vs_noncollinear(Hartree())
    test_kernel_collinear_vs_noncollinear(Xc([:lda_xc_teter93]))
    test_kernel_collinear_vs_noncollinear(Xc([:gga_c_pbe]))
    test_kernel_collinear_vs_noncollinear(Xc([:gga_x_pbe]))

    test_kernel(:collinear, Hartree())
    test_kernel(:collinear, LocalNonlinearity(ρ -> 1.2 * ρ^2.5))
    test_kernel(:collinear, Xc([:lda_xc_teter93]))
    test_kernel(:collinear, Xc([:gga_c_pbe]), test_compute=false)
    test_kernel(:collinear, Xc([:gga_x_pbe]), test_compute=false)
    test_kernel(:collinear, Xc([:gga_x_pbe, :gga_c_pbe]), test_compute=false)

    @testset "Non-linear core correction (NLCC)" begin
        psp = TestCases.silicon.psp_upf  # PseudoDojo v0.4.1 Si includes NLCC
        @test DFTK.has_core_density(load_psp(psp))
        test_kernel(:none, Xc([:lda_xc_teter93]); psp)
    end
end

# Tests the derivatives of the libxc potential_terms,
# especially the energy density e which is not tested by the apply_kernel tests.
@testitem "ForwardDiff potential_terms for libxc" tags=[:minimal] setup=[TestCases] begin
    using DFTK
    using DftFunctionals
    using DftFunctionals: potential_terms
    using ForwardDiff
    using LinearAlgebra
    import ForwardDiff
    import ForwardDiff: Dual, partials

    for spin in [:none, :collinear]
        @testset "Spin polarization: $spin" begin
            # Build a reasonable density from a silicon model
            testcase = TestCases.silicon
            Si = ElementPsp(testcase.atnum, load_psp(testcase.psp_gth))
            magnetic_moments = spin == :collinear ? [0.5, -0.5] : []
            model = model_DFT(testcase.lattice, [Si, Si], testcase.positions;
                              functionals=r2SCAN(), magnetic_moments)
            basis = PlaneWaveBasis(model; Ecut=2, kgrid=MonkhorstPack([2, 2, 2]))
            scfres = self_consistent_field(basis; ρ=guess_density(basis, magnetic_moments))
            ρ0 = scfres.ρ
            τ0 = scfres.τ

            # LibxcDensities with max_derivative=2 gives ρ, σ = |∇ρ|², and Δρ
            density = DFTK.LibxcDensities(basis, 2, ρ0, τ0)
            ρ  = reshape(density.ρ_real, size(density.ρ_real, 1), :)
            σ  = reshape(density.σ_real, size(density.σ_real, 1), :)
            Δρ = reshape(density.Δρ_real, size(density.Δρ_real, 1), :)
            τ  = reshape(density.τ_real, size(density.τ_real, 1), :)

            ε_ad = Dual{typeof(ForwardDiff.Tag(nothing, Float64))}(0.0, 1.0)
            function do_ad(f)
                map(f(ε_ad)) do y
                    partials.(y, 1)
                end
            end

            ε_fd = 1e-4
            function do_fd(f)
                f_m2ε = f(-2ε_fd)
                f_m1ε = f(-ε_fd)
                f_p1ε = f(ε_fd)
                f_p2ε = f(2ε_fd)
                map(f_m2ε, f_m1ε, f_p1ε, f_p2ε) do y_m2ε, y_m1ε, y_p1ε, y_p2ε
                    (-y_p2ε + 8*y_p1ε - 8*y_m1ε + y_m2ε) / 12ε_fd
                end
            end

            # Random δρ with consistent δσ and δΔρ
            δρ0 = randn(size(ρ0)) / model.unit_cell_volume
            δdens = DFTK.LibxcDensities(basis, 2, ρ0.+ε_ad.*δρ0, nothing)
            δσ_real = partials.(δdens.σ_real, 1)
            δΔρ_real = partials.(δdens.Δρ_real, 1)
            δρ = reshape(δρ0, size(ρ)...)
            δσ = reshape(δσ_real, size(σ)...)
            δΔρ = reshape(δΔρ_real, size(Δρ)...)
            # And a random δτ
            δτ = randn(size(τ)) / model.unit_cell_volume

            @testset "LDA" begin
                func = DFTK.LibxcFunctional(:lda_xc_teter93)

                f = ε -> potential_terms(func, ρ .+ ε .* δρ)

                δe_ad, δVρ_ad = do_ad(f)
                δe_fd, δVρ_fd = do_fd(f)

                @test δe_ad  ≈ δe_fd  rtol=1e-6
                @test δVρ_ad ≈ δVρ_fd rtol=1e-6
            end

            @testset "GGA" begin
                func = DFTK.LibxcFunctional(:gga_x_pbe)

                f = ε -> potential_terms(func, ρ .+ ε .* δρ, σ .+ ε .* δσ)

                δe_ad, δVρ_ad, δVσ_ad = do_ad(f)
                δe_fd, δVρ_fd, δVσ_fd = do_fd(f)

                @test δe_ad  ≈ δe_fd  rtol=1e-6
                @test δVρ_ad ≈ δVρ_fd rtol=1e-6
                @test δVσ_ad ≈ δVσ_fd rtol=1e-6
            end

            @testset "MGGA" begin
                func = DFTK.LibxcFunctional(:mgga_x_r2scan)

                f = ε -> potential_terms(func, ρ .+ ε .* δρ, σ .+ ε .* δσ, τ .+ ε .* δτ)

                δe_ad, δVρ_ad, δVσ_ad, δVτ_ad = do_ad(f)
                δe_fd, δVρ_fd, δVσ_fd, δVτ_fd = do_fd(f)

                @test δe_ad  ≈ δe_fd  rtol=1e-6
                @test δVρ_ad ≈ δVρ_fd rtol=1e-6
                # Seems more sensitive to noise, use a slightly looser tolerance:
                @test δVσ_ad ≈ δVσ_fd rtol=3e-6
                @test δVτ_ad ≈ δVτ_fd rtol=1e-6
            end

            @testset "MGGAL" begin
                # Need a (∇²ρ, τ)-dependent MGGA, seems more stable than the original br89
                func = DFTK.LibxcFunctional(:mgga_x_br89_explicit)
                @assert func isa DFTK.LibxcFunctional{:mggal}

                f = ε -> potential_terms(func, ρ .+ ε .* δρ, σ .+ ε .* δσ,
                                               τ .+ ε .* δτ, Δρ .+ ε .* δΔρ)

                δe_ad, δVρ_ad, δVσ_ad, δVτ_ad, δVl_ad = do_ad(f)
                δe_fd, δVρ_fd, δVσ_fd, δVτ_fd, δVl_fd = do_fd(f)

                @test δe_ad  ≈ δe_fd  rtol=1e-6
                @test δVρ_ad ≈ δVρ_fd rtol=1e-6
                @test δVσ_ad ≈ δVσ_fd rtol=1e-6
                @test δVτ_ad ≈ δVτ_fd rtol=1e-6
                @test δVl_ad ≈ δVl_fd rtol=1e-6
            end
        end
    end
end
