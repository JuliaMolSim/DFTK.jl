@testmodule mPspUpf begin  # PspUpf already exported by DFTK
    using DFTK
    using PseudoPotentialData

    pd_lda_family = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
    pd_pbe_family = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
    upf_pseudos = Dict(
        # Converted from cp2k repo (in GTH format) to UPF
        :Si => load_psp(joinpath(@__DIR__, "pseudos", "gth", "Si.pbe-hgh.upf")),
        :Tl => load_psp(joinpath(@__DIR__, "pseudos", "gth", "Tl.pbe-d-hgh.upf")),
        # No NLCC
        :Li => load_psp(pd_lda_family[:Li]),
        :Mg => load_psp(pd_lda_family[:Mg]),
        # With NLCC
        :Co => load_psp(pd_pbe_family[:Co]; rcut=10.0),
        :Ge => load_psp(pd_pbe_family[:Ge]),
        # With cutoff
        :Cu => load_psp(pd_pbe_family[:Cu]; rcut=9.0),
        :Cr => load_psp(pd_pbe_family[:Cr]; rcut=12.0)
    )
    gth_pseudos = [
        (; gth=load_psp(joinpath(@__DIR__, "pseudos", "gth", "Si-q4.gth")),  upf=upf_pseudos[:Si]),
        (; gth=load_psp(joinpath(@__DIR__, "pseudos", "gth", "Tl-q13.gth")), upf=upf_pseudos[:Tl]),
    ]
    psp8_pseudos = Dict(
        :Li_pbe => load_psp(joinpath(@__DIR__, "pseudos", "Li.psp8")),
    )
end


@testitem "Check reading PseudoDojo Li UPF" tags=[:psp] setup=[mPspUpf] begin
    psp = mPspUpf.upf_pseudos[:Li]

    @test psp.lmax == 1
    @test psp.Zion == 3
    @test length(psp.rgrid) == 1944
    @test length(psp.vloc) == 1944
    for m in psp.h
        @test size(m) == (2, 2)
    end

    @test psp.vloc[1] ≈ -1.2501238567E+01 / 2
    @test psp.h[1][1,1] ≈ -9.7091222353E+0 * 2
    @test psp.r2_projs[1][1][1] ≈ psp.rgrid[1] * -7.5698070034E-10 / 2
end

@testitem "Check reading PseudoDojo Li PSP8" tags=[:psp] setup=[mPspUpf] begin
    psp = mPspUpf.psp8_pseudos[:Li_pbe]

    @test psp.lmax == 1
    @test psp.Zion == 3
    @test length(psp.rgrid) == 400
    @test length(psp.vloc) == 400
    for m in psp.h
        @test size(m) == (2, 2)
    end

    @test psp.vloc[1] ≈ -6.2945684552403
    @test psp.h[1][1,1] ≈ -4.9515440588302 * 4
    @test psp.r2_projs[1][1][1] ≈ psp.rgrid[1] * -6.2444638349035e-10
end

@testitem "Real potentials are consistent with HGH" tags=[:psp] setup=[mPspUpf] begin
    using DFTK: eval_psp_local_real

    for psp_pair in mPspUpf.gth_pseudos
        upf = psp_pair.upf
        gth = psp_pair.gth
        rand_r = rand(5) .* abs(upf.rgrid[end] - upf.rgrid[1]) .+ upf.rgrid[1]
        for r in [upf.rgrid[1], rand_r..., upf.rgrid[end]]
            reference_gth = eval_psp_local_real(gth, r)
            @test reference_gth ≈ eval_psp_local_real(upf, r) rtol=1e-2 atol=1e-2
        end
    end
end

@testitem "Fourier potentials are consistent with HGH" tags=[:psp] setup=[mPspUpf] begin
    using DFTK: eval_psp_local_fourier

    for psp_pair in mPspUpf.gth_pseudos
        upf = psp_pair.upf
        gth = psp_pair.gth
        for p in (0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.)
            reference_gth = eval_psp_local_fourier(gth, p)
            @test reference_gth ≈ eval_psp_local_fourier(upf, p) rtol=1e-3 atol=1e-3
        end
    end
end

@testitem "Projectors are consistent with HGH in real and Fourier space" #=
    =#    tags=[:psp] setup=[mPspUpf] begin
    using DFTK: eval_psp_projector_fourier, eval_psp_projector_real, eval_psp_local_fourier
    using DFTK: count_n_proj_radial

    for psp_pair in mPspUpf.gth_pseudos
        upf = psp_pair.upf
        gth = psp_pair.gth

        @test upf.lmax == gth.lmax
        for l = 0:upf.lmax
            @test count_n_proj_radial(upf, l) == count_n_proj_radial(gth, l)
        end

        for l = 0:upf.lmax, i in count_n_proj_radial(upf, l)
            ircut = length(upf.r2_projs[l+1][i])
            for p in (0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.)
                reference_gth = eval_psp_projector_fourier(gth, i, l, p)
                proj_upf = eval_psp_projector_fourier(upf, i, l, p)
                @test reference_gth ≈ proj_upf atol=1e-5 rtol=1e-5
            end
            for r in [upf.rgrid[1], upf.rgrid[ircut]]
                reference_gth = eval_psp_projector_real(gth, i, l, r)
                proj_upf = eval_psp_projector_real(upf, i, l, r)
                @test reference_gth ≈ proj_upf atol=1e-5 rtol=1e-5
            end
        end
    end
end

@testitem "Energy correction is consistent with HGH" tags=[:psp] setup=[mPspUpf] begin
    using DFTK: eval_psp_energy_correction

    for psp_pair in mPspUpf.gth_pseudos
        upf = psp_pair.upf
        gth = psp_pair.gth
        reference_gth = eval_psp_energy_correction(gth)
        @test reference_gth ≈ eval_psp_energy_correction(upf) atol=1e-3 rtol=1e-3
    end
end

@testitem "Potentials are consistent in real and Fourier space" #=
    =#    tags=[:psp] setup=[mPspUpf] begin
    using DFTK: eval_psp_local_real, eval_psp_local_fourier
    using QuadGK

    function integrand(psp, p, r)
        4π * (eval_psp_local_real(psp, r) + psp.Zion / r) * sin(p * r) / (p * r) * r^2
    end
    for psp in values(mPspUpf.upf_pseudos)
        for p in (0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.)
            reference = quadgk(r -> integrand(psp, p, r), psp.rgrid[begin],
                               psp.rgrid[psp.ircut])[1]
            correction = 4π * psp.Zion / p^2
            @test (reference - correction) ≈ eval_psp_local_fourier(psp, p) atol=1e-2 rtol=1e-2
        end
    end
end

@testitem "Projectors are consistent in real and Fourier space" #=
    =#    tags=[:psp] setup=[mPspUpf] begin
    using DFTK: eval_psp_projector_fourier, eval_psp_projector_real, count_n_proj_radial
    using SpecialFunctions: sphericalbesselj
    using QuadGK

    # The integrand for performing the spherical Hankel transform,
    # i.e. compute the radial part of the projector in Fourier space
    function integrand(psp, i, l, p, r)
        4π * r^2 * eval_psp_projector_real(psp, i, l, r) * sphericalbesselj(l, p * r)
    end

    for psp in values(mPspUpf.upf_pseudos)
        ir_start = iszero(psp.rgrid[1]) ? 2 : 1
        for l = 0:psp.lmax, i in count_n_proj_radial(psp, l)
            ir_cut = min(psp.ircut, length(psp.r2_projs[l+1][i]))
            for p in (0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.)
                # DFTK uses a modified Hankel, hence divide by 1/p^l
                reference = 1/p^l * quadgk(r -> integrand(psp, i, l, p, r),
                                           psp.rgrid[ir_start], psp.rgrid[ir_cut])[1]
                @test reference ≈ eval_psp_projector_fourier(psp, i, l, p) atol=1e-2 rtol=1e-2
            end
        end
    end
end

@testitem "Valence charge densities are consistent in real and Fourier space" #=
    =#    tags=[:psp] setup=[mPspUpf] begin
    using DFTK: eval_psp_valence_density_real, eval_psp_valence_density_fourier
    using SpecialFunctions: sphericalbesselj
    using QuadGK

    function integrand(psp, p, r)
        4π * r^2 * eval_psp_valence_density_real(psp, r) * sphericalbesselj(0, p * r)
    end
    for psp in values(mPspUpf.upf_pseudos)
        ir_start = iszero(psp.rgrid[1]) ? 2 : 1
        for p in (0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.)
            reference = quadgk(r -> integrand(psp, p, r), psp.rgrid[ir_start],
                               psp.rgrid[psp.ircut])[1]
            @test reference  ≈ eval_psp_valence_density_fourier(psp, p) atol=1e-2 rtol=1e-2
        end
    end
end

@testitem "Core charge densities are consistent in real and Fourier space" #=
    =#    tags=[:psp] setup=[mPspUpf] begin
    using DFTK: eval_psp_core_density_real, eval_psp_core_density_fourier
    using SpecialFunctions: sphericalbesselj
    using QuadGK

    function integrand(psp, p, r)
        4π * r^2 * eval_psp_core_density_real(psp, r) * sphericalbesselj(0, p * r)
    end
    for psp in values(mPspUpf.upf_pseudos)
        ir_start = iszero(psp.rgrid[1]) ? 2 : 1
        for p in (0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.)
            reference = quadgk(r -> integrand(psp, p, r), psp.rgrid[ir_start],
                               psp.rgrid[psp.ircut])[1]
            @test reference  ≈ eval_psp_core_density_fourier(psp, p) atol=1e-2 rtol=1e-2
        end
    end
end

@testitem "Core kinetic energy densities are consistent in real and Fourier space" #=
    =#    tags=[:psp] begin
    using DFTK: eval_psp_core_kinetic_energy_density_real,
                eval_psp_core_kinetic_energy_density_fourier,
                load_psp
    using SpecialFunctions: sphericalbesselj
    using QuadGK

    function integrand(psp, p, r)
        4π * r^2 * eval_psp_core_kinetic_energy_density_real(psp, r) * sphericalbesselj(0, p * r)
    end
    for psp in [load_psp(joinpath(@__DIR__, "pseudos", "C_m.upf"))]
        ir_start = iszero(psp.rgrid[1]) ? 2 : 1
        for p in (0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.)
            reference = quadgk(r -> integrand(psp, p, r), psp.rgrid[ir_start],
                               psp.rgrid[psp.ircut])[1]
            @test reference ≈ eval_psp_core_kinetic_energy_density_fourier(psp, p) atol=1e-2 rtol=1e-2
        end
    end
end

@testitem "PSP energy correction is consistent with fourier-space potential" #=
    =#    tags=[:psp] setup=[mPspUpf] begin
    using DFTK: eval_psp_local_fourier, eval_psp_energy_correction

    p_small = 1e-2    # We are interested in p→0 term
    for psp in values(mPspUpf.upf_pseudos)
        coulomb = -4π * (psp.Zion) / p_small^2
        reference = eval_psp_local_fourier(psp, p_small) - coulomb
        @test reference ≈ eval_psp_energy_correction(psp) atol=1e-2
    end
end

@testitem "PSP guess density is positive" tags=[:psp] setup=[mPspUpf] begin
    using DFTK
    using LinearAlgebra

    lattice = 5 * I(3)
    positions = [zeros(3)]
    for (element, psp) in mPspUpf.upf_pseudos
        atoms = [ElementPsp(element, psp)]
        model = model_DFT(lattice, atoms, positions; functionals=LDA())
        basis = PlaneWaveBasis(model; Ecut=22, kgrid=[2, 2, 2])
        ρ_val = guess_density(basis, ValenceDensityPseudo())
        ρ_val_neg = abs(sum(ρ_val[ρ_val .< 0]))
        @test ρ_val_neg * model.unit_cell_volume / prod(basis.fft_size) < 1e-6
    end
end

@testitem "PSP total guess density gives Z-valence" tags=[:psp] setup=[mPspUpf] begin
    using DFTK
    using LinearAlgebra

    lattice = 5 * I(3)
    positions = [zeros(3)]
    for (element, psp) in mPspUpf.upf_pseudos
        if sum(psp.r2_ρion) > 0  # Otherwise, it's all 0 in the UPF as a placeholder
            atoms = [ElementPsp(element, psp)]
            model = model_DFT(lattice, atoms, positions; functionals=LDA())
            basis = PlaneWaveBasis(model; Ecut=22, kgrid=[2, 2, 2])
            ρ_val = guess_density(basis, ValenceDensityPseudo())
            Z_valence = sum(ρ_val) * model.unit_cell_volume / prod(basis.fft_size)
            @test Z_valence ≈ charge_ionic(psp) rtol=1e-5 atol=1e-5
        end
    end
end

@testitem "All pseudopotentials from common UPF families can be loaded" begin
    using PseudoPotentialData
    using DFTK

    for key in ("dojo.nc.sr.lda.v0_4_1.standard.upf",
                "dojo.nc.sr.pbe.v0_5.standard.upf",
                "dojo.nc.sr.pbesol.v0_4_1.standard.upf")
        pseudopotentials = PseudoFamily(key)
        for element in keys(pseudopotentials)
            psp = load_psp(pseudopotentials, element)
            @test psp isa PspUpf
        end
    end
end

@testitem "Fourier tables agree with an accurate quadrature" #=
    =#    tags=[:psp] setup=[mPspUpf] begin
    using DFTK
    using DFTK: eval_psp_local_fourier, eval_psp_projector_fourier, eval_psp_pswfc_fourier
    using DFTK: eval_psp_valence_density_fourier, eval_psp_core_density_fourier
    using QuadGK
    using ForwardDiff
    using SpecialFunctions: erf
    upf_pseudos = mPspUpf.upf_pseudos
    psp8_pseudos = mPspUpf.psp8_pseudos

    # The tables replace a Simpson quadrature over the radial mesh, and are *more* accurate
    # than it (the mesh is coarse where the pseudo has a cutoff kink), so Simpson is not a
    # usable reference here. Integrate the very same radial spline adaptively instead: that
    # isolates the error of the transform from the error of the radial interpolation.
    # Build the spline once per quantity -- rebuilding it inside the integrand (it costs a
    # tridiagonal solve) makes this test minutes long instead of seconds.
    function reference(rgrid, r2_f, l, p)
        r2f = DFTK.radial_spline(rgrid, r2_f)
        4π / p^l * quadgk(r -> r2f(r) * DFTK.sphericalbesselj_fast(l, p * r),
                          rgrid[1], rgrid[end]; rtol=1e-8)[1]
    end
    # Exact derivative rule H̃'_l(p) = -4π/p^l ∫ r² f(r) r j_{l+1}(p r) dr
    function dreference(rgrid, r2_f, l, p)
        r2f = DFTK.radial_spline(rgrid, r2_f)
        -4π / p^l * quadgk(r -> r * r2f(r) * DFTK.sphericalbesselj_fast(l + 1, p * r),
                           rgrid[1], rgrid[end]; rtol=1e-8)[1]
    end

    ps = [0.3, 2.7, 15.0]
    # :Si is on a logarithmic mesh, the others are linear; :Co carries an rcut, so its
    # pswfcs are transformed on a different grid than the rest of its quantities.
    for psp in [upf_pseudos[:Si], upf_pseudos[:Li], upf_pseudos[:Co],
                psp8_pseudos[:Li_pbe]]
        rcut = 1:psp.ircut
        # (name, rgrid, r2_f, l, evaluator)
        quantities = Any[("ρion", view(psp.rgrid, rcut), view(psp.r2_ρion, rcut), 0,
                          p -> eval_psp_valence_density_fourier(psp, p))]
        if DFTK.has_core_density(psp)
            push!(quantities, ("ρcore", view(psp.rgrid, rcut), view(psp.r2_ρcore, rcut), 0,
                               p -> eval_psp_core_density_fourier(psp, p)))
        end
        for l = 0:psp.lmax, i = 1:length(psp.r2_projs[l+1])
            icut = min(psp.ircut, length(psp.r2_projs[l+1][i]))
            push!(quantities, ("projector l=$l i=$i", view(psp.rgrid, 1:icut),
                               view(psp.r2_projs[l+1][i], 1:icut), l,
                               p -> eval_psp_projector_fourier(psp, i, l, p)))
        end
        for l = 0:psp.lmax, i = 1:DFTK.count_n_pswfc_radial(psp, l)
            push!(quantities, ("pswfc l=$l i=$i", psp.rgrid, psp.r2_pswfcs[l+1][i], l,
                               p -> eval_psp_pswfc_fourier(psp, i, l, p)))
        end

        for (name, rgrid, r2_f, l, evaluator) in quantities
            refs = [reference(rgrid, r2_f, l, p) for p in ps]
            scale = maximum(abs, refs)
            scale < 1e-10 && continue  # Quantity absent from this pseudo
            for (p, ref) in zip(ps, refs)
                @test abs(evaluator(p) - ref) < 1e-5 * scale
            end
            # ForwardDiff differentiates the interpolant: check it against the exact rule.
            dad = ForwardDiff.derivative(evaluator, 2.7)
            @test abs(dad - dreference(rgrid, r2_f, l, 2.7)) < 1e-4 * scale
        end

        # The local potential is tabulated with its Coulomb tail taken out; the tail is
        # added back analytically and p = 0 stays zero (compensating charge background).
        @test iszero(eval_psp_local_fourier(psp, 0.0))
        r2_vloc_corrected = psp.rgrid[rcut] .^ 2 .* psp.vloc[rcut] .+
                            psp.Zion .* psp.rgrid[rcut] .* erf.(psp.rgrid[rcut])
        for p in ps
            ref = reference(view(psp.rgrid, rcut), r2_vloc_corrected, 0, p) +
                  4π * -psp.Zion / p^2 * exp(-p^2 / 4)
            # vloc(p) spans four orders of magnitude over this p range and crosses zero, so
            # judge it against the scale of its Coulomb tail rather than its own value.
            @test abs(eval_psp_local_fourier(psp, p) - ref) < 1e-7 * 4π * psp.Zion / p^2
        end
    end
end

@testitem "Fourier tables: small-p series, smoothness, vectorized paths" #=
    =#    tags=[:psp] setup=[mPspUpf] begin
    using DFTK
    using DFTK: eval_psp_local_fourier, eval_psp_projector_fourier
    using DFTK: eval_psp_valence_density_fourier
    using ForwardDiff

    psp = mPspUpf.upf_pseudos[:Li]

    # Below pcut the table evaluates its Taylor series, which is exact at p = 0: the valence
    # density integrates to the number of valence electrons.
    @test eval_psp_valence_density_fourier(psp, 0.0) ≈ psp.Zion rtol=1e-6
    # ... and matches the spline it hands over to, on the other side of the crossover.
    pcut = DFTK.HANKEL_TABLE_PCUT
    series = eval_psp_valence_density_fourier(psp, prevfloat(pcut))
    spline = eval_psp_valence_density_fourier(psp, nextfloat(pcut))
    @test series ≈ spline rtol=1e-8

    # The interpolation is a cubic spline, so the second derivative is continuous across a
    # node of the table (a Lagrange stencil would jump here). This is what the stress and
    # response code paths differentiate through.
    (; logpmin, Δlogp) = psp.r2_ρion_table
    pnode = exp(logpmin + 100Δlogp)
    d2(p) = ForwardDiff.derivative(q -> ForwardDiff.derivative(
                r -> eval_psp_valence_density_fourier(psp, r), q), p)
    @test d2(prevfloat(pnode, 10)) ≈ d2(nextfloat(pnode, 10)) rtol=1e-6

    # Vectorized (GPU) paths agree with the scalar ones.
    ps = [0.0, 0.7, 3.0, 30.0]
    @test eval_psp_projector_fourier(psp, 1, 0, ps) ≈
          [eval_psp_projector_fourier(psp, 1, 0, p) for p in ps]
    @test eval_psp_local_fourier(psp, ps) ≈ [eval_psp_local_fourier(psp, p) for p in ps]
end

@testitem "Fourier tables against analytic transforms" tags=[:psp] begin
    using DFTK
    using DFTK: hankel_table_plan, build_hankel_table, hankel
    using ForwardDiff

    # Radial quantities whose modified Hankel transform H̃_l(p) = 4π/p^l ∫ r² f j_l(pr) dr is
    # known in closed form -- along with its p-derivatives -- sampled on a typical UPF mesh
    # (linear, h = 0.01, out to 15).
    #
    #   Gaussian  f = r^l e^{-r²}  →  π^{3/2} / (2^l e^{p²/4})   -- smooth, dies at both ends
    #   Slater    f = e^{-2r}      →  8π·2 / (4 + p²)²  (l = 0)  -- cusp at r = 0, p⁻⁴ tail,
    #                                                               i.e. real signal at high p
    rgrid = collect(0.0:0.01:15.0)
    gaussian_exact(l, p) = π^(3/2) / (2^l * exp(p^2 / 4))
    slater_exact(p)   =   8π * 2 / (4 + p^2)^2
    slater_exact′(p)  = -32π * 2 * p / (4 + p^2)^3
    slater_exact″(p)  = -32π * 2 * (4 - 5p^2) / (4 + p^2)^4
    table_of(f, l) = build_hankel_table(hankel_table_plan(rgrid[end], l), rgrid,
                                        [r^2 * f(r) for r in rgrid], l)

    # l = 0 is the delicate case for the *radial* spline: r²f then has curvature 2f(0) ≠ 0 at
    # the origin, so a natural interpolating spline (y′′ = 0) misrepresents the first mesh cell
    # by O(1). That error sits within one spacing h of r = 0, i.e. it is nearly a delta function
    # -- and the transform of a delta is *flat*, so it does not decay but sets a ~6e-7 floor on
    # H̃ out to p ~ 1/h. It perturbs no p = 0 quantity, so norms and charges stay right and it
    # hides from every other test in this file, surfacing only as a spuriously negative core
    # density in real space. Hence: check the tail, not just the norm.
    for l = 0:2
        table = table_of(r -> r^l * exp(-r^2), l)
        for p in (0.5, 1.0, 3.0)
            @test table(p) ≈ gaussian_exact(l, p) rtol=1e-11
        end
    end

    slater = table_of(r -> exp(-2r), 0)
    r2_slater = [r^2 * exp(-2r) for r in rgrid]
    for p in (1.0, 5.0, 10.0, 20.0, 40.0)
        @test slater(p) ≈ slater_exact(p) rtol=1e-6
        # The tables replace a Simpson quadrature over the radial mesh; with a high-order
        # radial spline they beat it everywhere, by ~4 orders of magnitude.
        exact = slater_exact(p)
        @test abs(slater(p) - exact) < abs(hankel(rgrid, r2_slater, 0, p) - exact)
    end

    # Derivatives, which is what the stress and response paths actually differentiate. These
    # are the reason the spline in log p is of order 6 rather than cubic: an order-k spline is
    # only C^{k-2}, so a cubic one has a merely piecewise-linear second derivative and lands at
    # rtol ~ 1e-6 on H̃″ below, where order 6 gives ~1e-9.
    for p in (1.0, 5.0, 10.0)
        d1 = ForwardDiff.derivative(slater, p)
        d2 = ForwardDiff.derivative(q -> ForwardDiff.derivative(slater, q), p)
        @test d1 ≈ slater_exact′(p) rtol=1e-6
        @test d2 ≈ slater_exact″(p) rtol=1e-7
    end
end

@testitem "Fourier tables of a quantity cut while still nonzero" tags=[:psp] setup=[mPspUpf] begin
    using DFTK: eval_psp_pswfc_fourier, radial_spline, sphericalbesselj_fast
    using QuadGK

    # The pseudo-atomic wavefunctions are the one family still at ~1e-2 of their peak where the
    # radial mesh is cut. By Euler-Maclaurin the error of the (uniform-grid) quadrature inside
    # the transform *is* that boundary term, so these are the quantities -- and the only ones --
    # that the order of `gregory_weights` binds: a 4th-order rule floors them near 1e-9, well
    # above the ~1e-13 the order-6 splines around it are worth. Guard the 6th-order rule.
    psp = mPspUpf.upf_pseudos[:Ge]
    l, i = 1, 1                                  # its l=1 pswfc ends at 9.8e-3 of its peak
    r2_f = psp.r2_pswfcs[l+1][i]
    @test abs(r2_f[end]) / maximum(abs, r2_f) > 1e-3   # ... i.e. this test is testing something

    # Reference: adaptive quadrature of the very same radial spline, which isolates the error of
    # the transform (quadrature + interpolation in p) from that of the radial interpolation.
    spline = radial_spline(psp.rgrid, r2_f)
    reference(p) = 4π / p^l * quadgk(r -> spline(r) * sphericalbesselj_fast(l, p * r),
                                     psp.rgrid[1], psp.rgrid[end]; rtol=1e-12)[1]
    scale = abs(4π / 3 * quadgk(r -> spline(r) * r^l, psp.rgrid[1], psp.rgrid[end];
                                rtol=1e-12)[1])   # 3 = (2l+1)!! for l = 1
    for p in (2.7, 5.0)
        @test abs(eval_psp_pswfc_fourier(psp, i, l, p) - reference(p)) < 1e-10 * scale
    end
end

@testitem "Fourier tables decay at large p" tags=[:psp] setup=[mPspUpf] begin
    using DFTK: eval_psp_core_density_fourier

    # As above, but on a real pseudopotential: the core density is sharply peaked and l = 0,
    # so it is the quantity a bad end condition at the origin damages most.
    psp = mPspUpf.upf_pseudos[:Ge]  # NLCC, linear mesh
    ρcore_max = eval_psp_core_density_fourier(psp, 0.0)
    for p in (20.0, 30.0, 40.0)
        @test abs(eval_psp_core_density_fourier(psp, p)) < 1e-8 * ρcore_max
    end
end

@testitem "Basis construction rejects a psp tabulated too coarsely" #=
    =#    tags=[:psp] setup=[mPspUpf] begin
    using DFTK
    using LinearAlgebra

    # A basis needs the Fourier transforms at every |G| of its density cube. The table cannot
    # raise on an out-of-range p (it is evaluated in a GPU kernel), so the basis checks the
    # largest |G| against the table's range up front. With the real HANKEL_TABLE_PMAX this is
    # unreachable at any sane Ecut, so shrink a copy of the table to provoke it.
    psp = mPspUpf.upf_pseudos[:Si]
    table = psp.vloc_table
    small = DFTK.HankelTable{DFTK.HANKEL_TABLE_ORDER_P,Float64,Vector{Float64}}(
        table.coefficients, table.logpmin, table.Δlogp, 1.0,
        table.moment0, table.moment2, table.moment4)
    fields = [f === :vloc_table ? small : getfield(psp, f) for f in fieldnames(DFTK.PspUpf)]
    psp_small = typeof(psp)(fields...)
    @test DFTK.max_momentum_fourier(psp_small) == 1.0

    model = model_DFT(diagm([10.0, 10.0, 10.0]), [ElementPsp(:Si, psp_small)], [zeros(3)];
                      functionals=LDA())
    @test_throws ErrorException PlaneWaveBasis(model; Ecut=20, kgrid=[1, 1, 1])
    # The same basis is fine with the pseudopotential's real table.
    model_ok = model_DFT(diagm([10.0, 10.0, 10.0]), [ElementPsp(:Si, psp)], [zeros(3)];
                         functionals=LDA())
    @test PlaneWaveBasis(model_ok; Ecut=20, kgrid=[1, 1, 1]) isa PlaneWaveBasis
end
