using Test
using DFTK: load_psp, eval_psp_projector_fourier, eval_psp_local_fourier
using DFTK: eval_psp_projector_real, eval_psp_local_real
using SpecialFunctions: besselj
using QuadGK

@testset "Check reading 'Si.nc.z_4.oncvpsp3.dojo.v4-std'" begin
    psp = load_psp("upf/pbe/si-nc-sr-standard-04.upf")

    @test psp.lmax == 2
    @test psp.Zion == 4
    @test length(psp.rgrid) == 1510
    @test length(psp.vloc) == 1510
    for m in psp.h
        @test size(m) == (2, 2)
    end

    @test psp.projs[1][1][1] ≈ -5.6328824383E-09
end

@testset "Real potentials are consistent with HGH" begin
    for pspfile in ["si-q4", "b-q3", "he-q2", "fe-q16", "mo-q14", "tl-q3", "tl-q13"]
        upf = load_psp("upf/pbe/$(pspfile).upf")
        hgh = load_psp("hgh/pbe/$(pspfile).hgh")
        for r in [upf.rgrid[1], upf.rgrid[end]]
            reference_hgh = eval_psp_local_real(hgh, r)
            @test reference_hgh ≈ eval_psp_local_real(upf, r) rtol=1e-8 atol=1e-8
        end
    end
end

@testset "Fourier potentials are consistent with HGH" begin
    for pspfile in ["si-q4", "b-q3", "he-q2", "fe-q16", "mo-q14", "tl-q3", "tl-q13"]
        upf = load_psp("upf/pbe/$(pspfile).upf")
        hgh = load_psp("hgh/pbe/$(pspfile).hgh")
        for q in [0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.]
            reference_hgh = eval_psp_local_fourier(hgh, q)
            @test reference_hgh ≈ eval_psp_local_fourier(upf, q) rtol=1e-5 atol=1e-5
        end
    end
end

@testset "Projectors are consistent with HGH in real and Fourier space" begin
    for pspfile in ["si-q4", "b-q3", "he-q2", "fe-q16", "mo-q14", "tl-q3", "tl-q13"]
        upf = load_psp("upf/pbe/$(pspfile).upf")
        hgh = load_psp("hgh/pbe/$(pspfile).hgh")
        for (l, i) in [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3),
                       (2, 1), (2, 2), (3, 1)]
            l > upf.lmax - 1 && continue  # Overshooting available AM
            i > length(upf.projs[l+1]) && continue  # Overshooting available projectors at AM
            ircut = length(upf.projs[l+1][i])
            for q in [0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.]
                reference_hgh = eval_psp_projector_fourier(hgh, i, l, q)
                @test reference_hgh ≈ eval_psp_projector_fourier(upf, i, l, q) / 2 atol=1e-7 rtol=1e-7
            end
            for r in [upf.rgrid[1], upf.rgrid[ircut]]
                reference_hgh = eval_psp_projector_real(hgh, i, l, r)
                @test reference_hgh ≈ eval_psp_projector_real(upf, i, l, r) / 2 atol=1e-7 rtol=1e-7
            end
        end
    end
end

@testset "Energy correction is consistent with HGH" begin
    for pspfile in ["si-q4", "b-q3", "he-q2", "fe-q16", "mo-q14", "tl-q3", "tl-q13"]
        upf = load_psp("upf/pbe/$(pspfile).upf")
        hgh = load_psp("hgh/pbe/$(pspfile).hgh")
        n_electrons = 3
        reference_hgh = eval_psp_energy_correction(hgh, n_electrons)
        @test reference_hgh ≈ eval_psp_energy_correction(upf, n_electrons) rtol=1e-5 atol=1e-5
    end
end

@testset "Potentials are consistent in real and Fourier space" begin
    reg_param = 0.25  # divergent integral, needs regularization
    function integrand(psp_local_real, q, r)
        4π * psp_local_real(r) * exp(-reg_param * r) * sin(q*r) / q * r
    end
    for pspfile in ["si-nc-sr-standard-04.upf"]
        psp = load_psp("upf/pbe/" * pspfile)
        psp_local_real = linear_interpolation((psp.rgrid,), psp.vloc ./ 2)

        for q in [0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.]
            reference = quadgk(r -> integrand(psp_local_real, q, r), psp.rgrid[1], psp.rgrid[end])[1]
            @test reference ≈ eval_psp_local_fourier(psp, q) rtol=1. atol=1.
        end
    end
end

@testset "Projectors are consistent in real and Fourier space" begin
    # The spherical bessel function of the first kind in terms of ordinary bessels:
    function j(n, x::T) where {T}
        x == 0 ? zero(T) : sqrt(π/2x) * besselj(n+1/2, x)
    end

    # The integrand for performing the spherical Hankel transfrom,
    # i.e. compute the radial part of the projector in Fourier space
    # Note: eval_psp_projector_real returns rβ, not just β
    function integrand(psp, i, l, q, x)
        4π * x * eval_psp_projector_real(psp, i, l, x) * j(l, q*x)
    end

    for pspfile in ["si-nc-sr-standard-04.upf"]
        psp = load_psp("upf/pbe/" * pspfile)
        for (l, i) in [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3),
                       (2, 1), (2, 2), (3, 1)]
            l > psp.lmax - 1 && continue  # Overshooting available AM
            i > length(psp.projs[l+1]) && continue  # Overshooting available projectors at AM
            ircut = length(psp.projs[l+1][i])
            for q in [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10]
                reference = quadgk(r -> integrand(psp, i, l, q, r), psp.rgrid[1], psp.rgrid[ircut])[1]
                @test reference ≈ eval_psp_projector_fourier(psp, i, l, q) atol=1e-8 rtol=1e-8
            end
        end
    end
end

@testset "PSP energy correction is consistent with real-space potential" begin
    reg_param = 1e-6  # divergent integral, needs regularization
    q_small = 1e-8    # We are interested in q→0 term
    function integrand(psp, n_electrons, r)
        # Difference of potential of point-like atom (what is assumed in Ewald)
        # versus actual structure of the pseudo potential
        coulomb = -psp.Zion / r
        diff = n_electrons * (eval_psp_local_real(psp, r) - coulomb)
        4π * diff * exp(-reg_param * r) * sin(q_small*r) / q_small * r
    end

    n_electrons = 3
    for pspfile in ["si-nc-sr-standard-04.upf"]
        psp = load_psp("upf/pbe/" * pspfile)
        reference = quadgk(r -> integrand(psp, n_electrons, r), psp.rgrid[1], psp.rgrid[end])[1]
        @test reference ≈ eval_psp_energy_correction(psp, n_electrons) atol=1e-1
    end
end

@testset "PSP energy correction is consistent with fourier-space potential" begin
    q_small = 1e-3    # We are interested in q→0 term
    for pspfile in ["si-nc-sr-standard-04.upf"]
        psp = load_psp("upf/pbe/" * pspfile)
        coulomb = -4π * (psp.Zion * 2) / q_small^2
        reference = eval_psp_local_fourier(psp, q_small) - coulomb
        @test reference ≈ eval_psp_energy_correction(psp, 1) atol=1e-3
    end
end