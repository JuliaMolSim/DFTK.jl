using Test
using Downloads
using DFTK: eval_psp_projector_fourier, eval_psp_local_fourier
using DFTK: eval_psp_projector_real, eval_psp_local_real, eval_psp_energy_correction
using DFTK: count_n_proj_radial
using SpecialFunctions: sphericalbesselj
using QuadGK

base_url = "https://raw.githubusercontent.com/JuliaMolSim/PseudoLibrary/main/pseudos/"

upf_urls = [
    joinpath(base_url, "hgh_pbe_upf/Si.pbe-hgh.UPF"),
    joinpath(base_url, "hgh_pbe_upf/Tl.pbe-d-hgh.UPF"),
    joinpath(base_url, "pd_nc_sr_lda_standard_04_upf/Li.upf"),
    joinpath(base_url, "pd_nc_sr_pbe_standard_04_upf/Mg.upf")
]
hgh_pseudos = [
    (hgh="hgh/pbe/si-q4.hgh", upf=upf_urls[1]),
    (hgh="hgh/pbe/tl-q13.hgh", upf=upf_urls[2])
]

@testset "Check reading PseudoDojo Li UPF" begin
    psp = load_psp(Downloads.download(upf_urls[3], joinpath(tempdir(), "psp.upf")))

    @test psp.lmax == 1
    @test psp.Zion == 3
    @test length(psp.rgrid) == 1944
    @test length(psp.vloc) == 1944
    for m in psp.h
        @test size(m) == (2, 2)
    end

    @test psp.vloc[1] ≈ -1.2501238567E+01 / 2
    @test psp.h[1][1,1] ≈ -9.7091222353E+0 * 2
    @test psp.r_projs[1][1][1] ≈ -7.5698070034E-10 / 2
end

@testset "Real potentials are consistent with HGH" begin
    for psp_pair in hgh_pseudos
        upf = load_psp(Downloads.download(psp_pair.upf, joinpath(tempdir(), "psp.upf")))
        hgh = load_psp(psp_pair.hgh)
        rand_r = rand(5) .* abs(upf.rgrid[end] - upf.rgrid[1]) .+ upf.rgrid[1]
        for r in [upf.rgrid[1], rand_r..., upf.rgrid[end]]
            reference_hgh = eval_psp_local_real(hgh, r)
            @test reference_hgh ≈ eval_psp_local_real(upf, r) rtol=1e-3 atol=1e-3
        end
    end
end

@testset "Fourier potentials are consistent with HGH" begin
    for psp_pair in hgh_pseudos
        upf = load_psp(Downloads.download(psp_pair.upf, joinpath(tempdir(), "psp.upf")))
        hgh = load_psp(psp_pair.hgh)
        for q in (0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.)
            reference_hgh = eval_psp_local_fourier(hgh, q)
            @test reference_hgh ≈ eval_psp_local_fourier(upf, q) rtol=1e-5 atol=1e-5
        end
    end
end

@testset "Projectors are consistent with HGH in real and Fourier space" begin
    for psp_pair in hgh_pseudos
        upf = load_psp(Downloads.download(psp_pair.upf, joinpath(tempdir(), "psp.upf")))
        hgh = load_psp(psp_pair.hgh)

        @test upf.lmax == hgh.lmax
        for l in 0:upf.lmax
            @test count_n_proj_radial(upf, l) == count_n_proj_radial(hgh, l)
        end

        for l in 0:upf.lmax, i in count_n_proj_radial(upf, l)
            ircut = length(upf.r_projs[l+1][i])
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
    for psp_pair in hgh_pseudos
        upf = load_psp(Downloads.download(psp_pair.upf, joinpath(tempdir(), "psp.upf")))
        hgh = load_psp(psp_pair.hgh)
        n_electrons = 3
        reference_hgh = eval_psp_energy_correction(hgh, n_electrons)
        @test reference_hgh ≈ eval_psp_energy_correction(upf, n_electrons) rtol=1e-5 atol=1e-5
    end
end

@testset "Potentials are consistent in real and Fourier space" begin
    function integrand(psp, q, r)
        4π * (eval_psp_local_real(psp, r) + psp.Zion / r) * sin(q * r) / (q * r) * r^2
    end
    for upf_url in upf_urls
        psp = load_psp(Downloads.download(upf_url, joinpath(tempdir(), "psp.upf")))
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

    for upf_url in upf_urls
        psp = load_psp(Downloads.download(upf_url, joinpath(tempdir(), "psp.upf")))
        ir_start = iszero(psp.rgrid[1]) ? 2 : 1 
        for l in 0:psp.lmax, i in count_n_proj_radial(psp, l)
            ir_cut = length(psp.r_projs[l+1][i])
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
    for upf_url in upf_urls
        psp = load_psp(Downloads.download(upf_url, joinpath(tempdir(), "psp.upf")))
        coulomb = -4π * (psp.Zion) / q_small^2
        reference = eval_psp_local_fourier(psp, q_small) - coulomb
        @test reference ≈ eval_psp_energy_correction(psp, 1) atol=1e-3
    end
end
