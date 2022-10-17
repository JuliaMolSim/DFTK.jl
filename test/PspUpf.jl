using Test
using DFTK: load_psp, eval_psp_projector_fourier, eval_psp_local_fourier
using DFTK: eval_psp_projector_real, eval_psp_local_real, eval_psp_energy_correction
using DFTK: parse_upf_file, count_n_proj_radial
using SpecialFunctions: sphericalbesselj
using QuadGK

hgh_upf_files = ["si-q4", "tl-q13"]
oncv_upf_files = ["si-nc-sr-standard-04", "hf-sp-oncvpsp"]
all_upf_files = vcat(hgh_upf_files, oncv_upf_files)

@testset "Check reading 'si-nc-sr-standard-04.upf'" begin
    psp = parse_upf_file("psp/si-nc-sr-standard-04.upf")

    @test psp.lmax == 2
    @test psp.Zion == 4
    @test length(psp.rgrid) == 1510
    @test length(psp.vloc) == 1510
    for m in psp.h
        @test size(m) == (2, 2)
    end

    @test psp.projs[1][1][1] ≈ -5.6328824383E-09 / 2
end

@testset "Real potentials are consistent with HGH" begin
    for pspfile in hgh_upf_files
        upf = parse_upf_file("psp/$(pspfile).upf")
        hgh = load_psp("hgh/pbe/$(pspfile).hgh")
        rand_r = rand(5) .* abs(upf.rgrid[end] - upf.rgrid[1]) .+ upf.rgrid[1]
        for r in [upf.rgrid[1], rand_r..., upf.rgrid[end]]
            reference_hgh = eval_psp_local_real(hgh, r)
            @test reference_hgh ≈ eval_psp_local_real(upf, r) rtol=1e-3 atol=1e-3
        end
    end
end

@testset "Fourier potentials are consistent with HGH" begin
    for pspfile in hgh_upf_files
        upf = parse_upf_file("psp/$(pspfile).upf")
        hgh = load_psp("hgh/pbe/$(pspfile).hgh")
        for q in (0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.)
            reference_hgh = eval_psp_local_fourier(hgh, q)
            @test reference_hgh ≈ eval_psp_local_fourier(upf, q) rtol=1e-5 atol=1e-5
        end
    end
end

@testset "Projectors are consistent with HGH in real and Fourier space" begin
    for pspfile in hgh_upf_files
        upf = parse_upf_file("psp/$(pspfile).upf")
        hgh = load_psp("hgh/pbe/$(pspfile).hgh")

        @test upf.lmax == hgh.lmax
        for l in 0:upf.lmax
            @test count_n_proj_radial(upf, l) == count_n_proj_radial(hgh, l)
        end

        for l in 0:upf.lmax, i in count_n_proj_radial(upf, l)
            ircut = length(upf.projs[l+1][i])
            for q in (0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.)
                reference_hgh = eval_psp_projector_fourier(hgh, i, l, q)
                proj_upf = eval_psp_projector_fourier(upf, i, l, q)
                @test reference_hgh ≈ proj_upf atol=1e-7 rtol=1e-7
            end
            for r in [upf.rgrid[1], upf.rgrid[ircut]]
                reference_hgh = eval_psp_projector_real(hgh, i, l, r)
                proj_upf = eval_psp_projector_real(upf, i, l, r)
                @test reference_hgh ≈ proj_upf atol=1e-7 rtol=1e-7
            end
        end
    end
end

@testset "Energy correction is consistent with HGH" begin
    for pspfile in hgh_upf_files
        upf = parse_upf_file("psp/$(pspfile).upf")
        hgh = load_psp("hgh/pbe/$(pspfile).hgh")
        n_electrons = 3
        reference_hgh = eval_psp_energy_correction(hgh, n_electrons)
        @test reference_hgh ≈ eval_psp_energy_correction(upf, n_electrons) rtol=1e-5 atol=1e-5
    end
end

@testset "Potentials are consistent in real and Fourier space" begin
    function integrand(psp, q, r)
        4π * (eval_psp_local_real(psp, r) + psp.Zion / r) * sin(q * r) / (q * r) * r^2
    end
    for pspfile in all_upf_files
        psp = parse_upf_file("psp/$(pspfile).upf")
        for q in (0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.)
            reference = quadgk(r -> integrand(psp, q, r), psp.rgrid[begin], psp.rgrid[end])[1]
            correction = 4π * psp.Zion / q^2
            @test (reference - correction) ≈ eval_psp_local_fourier(psp, q) rtol=1. atol=1.
        end
    end
end

@testset "Projectors are consistent in real and Fourier space" begin
    # The integrand for performing the spherical Hankel transfrom,
    # i.e. compute the radial part of the projector in Fourier space
    function integrand(psp, i, l, q, x)
        4π * x^2 * eval_psp_projector_real(psp, i, l, x) * sphericalbesselj(l, q*x)
    end

    for pspfile in all_upf_files
        psp = parse_upf_file("psp/$(pspfile).upf")
        ir_start = iszero(psp.rgrid[1]) ? 2 : 1 
        for l in 0:psp.lmax, i in count_n_proj_radial(psp, l)
            ir_cut = length(psp.projs[l+1][i])
            for q in (0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.)
                reference = quadgk(r -> integrand(psp, i, l, q, r),
                                   psp.rgrid[ir_start], psp.rgrid[ir_cut])[1]
                @test reference ≈ eval_psp_projector_fourier(psp, i, l, q) atol=1e-2 rtol=1e-2
            end
        end
    end
end

@testset "PSP energy correction is consistent with fourier-space potential" begin
    q_small = 1e-3    # We are interested in q→0 term
    for pspfile in all_upf_files
        psp = parse_upf_file("psp/$(pspfile).upf")
        coulomb = -4π * (psp.Zion) / q_small^2
        reference = eval_psp_local_fourier(psp, q_small) - coulomb
        @test reference ≈ eval_psp_energy_correction(psp, 1) atol=1e-3
    end
end
