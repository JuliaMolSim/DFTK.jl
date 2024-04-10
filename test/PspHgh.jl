@testitem "Check reading 'C-lda-q4'" tags=[:psp] begin
    using LinearAlgebra
    using DFTK: load_psp

    psp = load_psp("hgh/lda/C-q4")

    @test psp.identifier == "hgh/lda/c-q4"
    @test occursin("c", lowercase(psp.description))
    @test occursin("pade", lowercase(psp.description))
    @test psp.Zion == 4
    @test psp.rloc == 0.34883045
    @test psp.cloc == [-8.51377110, 1.22843203, 0, 0]
    @test psp.lmax == 1
    @test psp.rp == [0.30455321, 0.2326773]
    @test psp.h[1] == 9.52284179 * ones(1, 1)
    @test psp.h[2] == zeros(0, 0)
end

@testitem "Check reading 'Ni-lda-q18'" tags=[:psp] begin
    using LinearAlgebra
    using DFTK: load_psp

    psp = load_psp("hgh/lda/Ni-q18")

    @test psp.identifier == "hgh/lda/ni-q18"
    @test occursin("ni", lowercase(psp.description))
    @test occursin("pade", lowercase(psp.description))
    @test psp.Zion == 18
    @test psp.rloc == 0.35000000
    @test psp.cloc == [3.61031072, 0.44963832, 0, 0]
    @test psp.lmax == 2
    @test psp.rp == [0.24510489, 0.23474136, 0.21494950]
    @test psp.h[1] == [[12.16113071, 3.51625420] [3.51625420, -9.07892931]]
    @test psp.h[2] == [[-0.82062357, 2.54774737] [2.54774737, -6.02907069]]
    @test psp.h[3] == -13.39506212 * ones(1, 1)
end

@testitem "Check evaluating 'Si-lda-q4'" tags=[:psp] begin
    using LinearAlgebra
    using DFTK: load_psp, eval_psp_projector_fourier, eval_psp_local_fourier

    psp = load_psp("hgh/lda/Si-q4")

    # Test local part evaluation
    @test eval_psp_local_fourier(psp, norm([0.1,   0,    0])) ≈ -400.395448865164*4π
    @test eval_psp_local_fourier(psp, norm([0.1, 0.2,    0])) ≈ -80.39317320182417*4π
    @test eval_psp_local_fourier(psp, norm([0.1, 0.2, -0.3])) ≈ -28.95951714682582*4π
    @test eval_psp_local_fourier(psp, norm([1.0, -2.0, 3.0])) ≈ -0.275673388844235*4π
    @test eval_psp_local_fourier(psp, norm([10.0, 0.0, 0.0])) ≈ -5.1468909215285576e-5*4π

    # Test nonlocal part evaluation
    psq = [0, 0.01, 0.1, 0.3, 1, 10]
    pnorms = sqrt.([0, 0.01, 0.1, 0.3, 1, 10])
    @test map(p -> eval_psp_projector_fourier(psp, 1, 0, p), pnorms) ≈ [
        6.503085484692629, 6.497277328372439, 6.445236803354619,
        6.331078654802208, 5.947214691896995, 2.661098803299718,
    ]
    @test map(p -> eval_psp_projector_fourier(psp, 2, 0, p), pnorms) ≈ [
        10.074536712471094, 10.059542796942894, 9.925438587886482,
        9.632787375976731, 8.664551612201326, 1.666783598475508
    ]
    @test map(p -> eval_psp_projector_fourier(psp, 3, 0, p), pnorms) ≈ [
        12.692723197804167, 12.666281142268161, 12.430208137727789,
        11.917710279480355, 10.249557409656868, 0.11180299205602792
    ]

    @test map(p -> eval_psp_projector_fourier(psp, 1, 1, p), pnorms) ≈ [
        0.0, 0.3149163627204332, 0.9853983576555614,
        1.667197861646941, 2.8039993470553535, 3.0863036233824626,
    ]
    @test map(p -> eval_psp_projector_fourier(psp, 2, 1, p), pnorms) ≈ [
        0.0, 0.5320561290084422, 1.657814585041487,
        2.778424038171201, 4.517311337690638, 2.7698566262467117,
    ]

    @test map(p -> eval_psp_projector_fourier(psp, 3, 1, p), pnorms) ≈ [
         0.0, 0.7482799478933317, 2.321676914155303,
         3.8541542745249706, 6.053770711942623, 1.6078748819430986,
    ]


end

@testitem "Check pcut routines" tags=[:psp] begin
    using LinearAlgebra
    using DFTK: load_psp, eval_psp_projector_fourier, eval_psp_local_fourier
    using DFTK: pcut_psp_projector, pcut_psp_local

    psp = load_psp("hgh/pbe/au-q11.hgh")
    ε = 1e-6

    let
        pcut = pcut_psp_local(psp)
        res = eval_psp_local_fourier.(psp, [pcut - ε, pcut, pcut + ε])
        @test (res[1] < res[2]) == (res[3] < res[2])
    end

    for i = 1:2, l = 0:2
        pcut = pcut_psp_projector(psp, i, l)
        res = eval_psp_projector_fourier.(psp, i, l, [pcut - ε, pcut, pcut + ε])
        @test (res[1] < res[2]) == (res[3] < res[2])
    end
end

@testitem "Agreement of polynomial implementation and eval functions" tags=[:psp] begin
    using LinearAlgebra
    using DFTK: load_psp, eval_psp_projector_fourier, eval_psp_local_fourier
    using DFTK: psp_local_polynomial, psp_projector_polynomial, count_n_proj_radial

    let
        psp = load_psp("hgh/lda/Si-q4")
        Qloc = psp_local_polynomial(Float64, psp)
        evalQloc(p) = let t = p * psp.rloc; Qloc(t) * exp(-t^2 / 2) / t^2; end
        for p in abs.(randn(10))
            @test evalQloc(p) ≈ eval_psp_local_fourier(psp, p)
        end
    end

    for pspfile in ["Au-q11", "Ba-q10"]
        psp = load_psp("hgh/lda/" * pspfile)
        for l = 0:psp.lmax, i = 1:count_n_proj_radial(psp, l)
            Qproj = psp_projector_polynomial(Float64, psp, i, l)
            evalQproj(p) = let t = p * psp.rp[l + 1]; Qproj(t) * exp(-t^2 / 2); end
            for p in abs.(randn(10))
                @test evalQproj(p) ≈ eval_psp_projector_fourier(psp, i, l, p)
            end
        end
    end
end

@testitem "Projectors are consistent in real and Fourier space" tags=[:psp] begin
    using LinearAlgebra
    using DFTK: load_psp, eval_psp_projector_fourier, eval_psp_projector_real
    using DFTK: count_n_proj_radial
    using SpecialFunctions: besselj
    using QuadGK

    # The spherical bessel function of the first kind in terms of ordinary bessels:
    function j(n, x::T) where {T}
        x == 0 ? zero(T) : sqrt(π/2x) * besselj(n+1/2, x)
    end

    # The integrand for performing the spherical Hankel transform,
    # i.e. compute the radial part of the projector in Fourier space
    function integrand(psp, i, l, p, x)
        4π * x^2 * eval_psp_projector_real(psp, i, l, x) * j(l, p*x)
    end

    for pspfile in ["Au-q11", "Ba-q10"]
        psp = load_psp("hgh/lda/" * pspfile)
        for l = 0:psp.lmax, i = 1:count_n_proj_radial(psp, l)
            for p in [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10]
                reference = quadgk(r -> integrand(psp, i, l, p, r), 0, Inf)[1]
                @test reference ≈ eval_psp_projector_fourier(psp, i, l, p) atol=5e-15 rtol=1e-8
            end
        end
    end
end

@testitem "Potentials are consistent in real and Fourier space" tags=[:psp] begin
    using LinearAlgebra
    using DFTK: load_psp, eval_psp_local_fourier, eval_psp_local_real
    using QuadGK

    reg_param = 1e-3  # divergent integral, needs regularization
    function integrand(psp, p, r)
        4π * eval_psp_local_real(psp, r) * exp(-reg_param * r) * sin(p*r) / p * r
    end

    for pspfile in ["Au-q11", "Ba-q10"]
        psp = load_psp("hgh/lda/" * pspfile)
        for p in [0.01, 0.2, 1, 1.3]
            reference = quadgk(r -> integrand(psp, p, r), 0, Inf)[1]
            @test reference ≈ eval_psp_local_fourier(psp, p) rtol=.1 atol = .1
        end
    end
end

@testitem "PSP energy correction is consistent with real-space potential" tags=[:psp] begin
    using LinearAlgebra
    using DFTK: load_psp, eval_psp_local_real, eval_psp_energy_correction
    using QuadGK

    reg_param = 1e-6  # divergent integral, needs regularization
    p_small = 1e-6    # We are interested in p→0 term
    function integrand(psp, n_electrons, r)
        # Difference of potential of point-like atom (what is assumed in Ewald)
        # versus actual structure of the pseudo potential
        coulomb = -psp.Zion / r
        diff = n_electrons * (eval_psp_local_real(psp, r) - coulomb)
        4π * diff * exp(-reg_param * r) * sin(p_small*r) / p_small * r
    end

    n_electrons = 20
    for pspfile in ["Au-q11", "Ba-q10"]
        psp = load_psp("hgh/lda/" * pspfile)
        reference = quadgk(r -> integrand(psp, n_electrons, r), 0, Inf)[1]
        @test reference ≈ eval_psp_energy_correction(psp, n_electrons) atol=1e-2
    end
end

@testitem "PSP energy correction is consistent with fourier-space potential" #=
    =#    tags=[:psp] begin
    using LinearAlgebra
    using DFTK: load_psp, eval_psp_local_fourier, eval_psp_energy_correction

    p_small = 1e-3    # We are interested in p→0 term
    for pspfile in ["Au-q11", "Ba-q10"]
        psp = load_psp("hgh/lda/" * pspfile)
        coulomb = -4π * psp.Zion / p_small^2
        reference = eval_psp_local_fourier(psp, p_small) - coulomb
        @test reference ≈ eval_psp_energy_correction(psp, 1) atol=1e-3
    end
end
