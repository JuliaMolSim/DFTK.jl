@testmodule mPspUpf begin  # PspUpf already exported by DFTK
using DFTK
using LazyArtifacts

upf_pseudos = Dict(
    # Converged from HGH
    :Si => load_psp(artifact"hgh_pbe_upf", "Si.pbe-hgh.UPF"),
    :Tl => load_psp(artifact"hgh_pbe_upf", "Tl.pbe-d-hgh.UPF"),
    # No NLCC
    :Li => load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf", "Li.upf"),
    :Mg => load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf", "Mg.upf"),
    # With NLCC
    :Co => load_psp(artifact"pd_nc_sr_pbe_standard_0.4.1_upf", "Co.upf"; rcut=10.0),
    :Ge => load_psp(artifact"pd_nc_sr_pbe_standard_0.4.1_upf", "Ge.upf"),
    # With cutoff
    :Cu => load_psp(artifact"pd_nc_sr_pbe_standard_0.4.1_upf", "Cu.upf"; rcut=9.0),
    :Cr => load_psp(artifact"pd_nc_sr_pbe_standard_0.4.1_upf", "Cu.upf"; rcut=12.0)
)
hgh_pseudos = [
    (; hgh=load_psp("hgh/pbe/si-q4.hgh"), upf=upf_pseudos[:Si]),
    (; hgh=load_psp("hgh/pbe/tl-q13.hgh"), upf=upf_pseudos[:Tl])
]
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

@testitem "Real potentials are consistent with HGH" tags=[:psp] setup=[mPspUpf] begin
    using DFTK: eval_psp_local_real

    for psp_pair in mPspUpf.hgh_pseudos
        upf = psp_pair.upf
        hgh = psp_pair.hgh
        rand_r = rand(5) .* abs(upf.rgrid[end] - upf.rgrid[1]) .+ upf.rgrid[1]
        for r in [upf.rgrid[1], rand_r..., upf.rgrid[end]]
            reference_hgh = eval_psp_local_real(hgh, r)
            @test reference_hgh ≈ eval_psp_local_real(upf, r) rtol=1e-2 atol=1e-2
        end
    end
end

@testitem "Fourier potentials are consistent with HGH" tags=[:psp] setup=[mPspUpf] begin
    using DFTK: eval_psp_local_fourier

    for psp_pair in mPspUpf.hgh_pseudos
        upf = psp_pair.upf
        hgh = psp_pair.hgh
        for p in (0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.)
            reference_hgh = eval_psp_local_fourier(hgh, p)
            @test reference_hgh ≈ eval_psp_local_fourier(upf, p) rtol=1e-3 atol=1e-3
        end
    end
end

@testitem "Projectors are consistent with HGH in real and Fourier space" #=
    =#    tags=[:psp] setup=[mPspUpf] begin
    using DFTK: eval_psp_projector_fourier, eval_psp_projector_real, eval_psp_local_fourier
    using DFTK: count_n_proj_radial

    for psp_pair in mPspUpf.hgh_pseudos
        upf = psp_pair.upf
        hgh = psp_pair.hgh

        @test upf.lmax == hgh.lmax
        for l = 0:upf.lmax
            @test count_n_proj_radial(upf, l) == count_n_proj_radial(hgh, l)
        end

        for l = 0:upf.lmax, i in count_n_proj_radial(upf, l)
            ircut = length(upf.r2_projs[l+1][i])
            for p in (0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.)
                reference_hgh = eval_psp_projector_fourier(hgh, i, l, p)
                proj_upf = eval_psp_projector_fourier(upf, i, l, p)
                @test reference_hgh ≈ proj_upf atol=1e-5 rtol=1e-5
            end
            for r in [upf.rgrid[1], upf.rgrid[ircut]]
                reference_hgh = eval_psp_projector_real(hgh, i, l, r)
                proj_upf = eval_psp_projector_real(upf, i, l, r)
                @test reference_hgh ≈ proj_upf atol=1e-5 rtol=1e-5
            end
        end
    end
end

@testitem "Energy correction is consistent with HGH" tags=[:psp] setup=[mPspUpf] begin
    using DFTK: eval_psp_energy_correction

    for psp_pair in mPspUpf.hgh_pseudos
        upf = psp_pair.upf
        hgh = psp_pair.hgh
        n_electrons = 3
        reference_hgh = eval_psp_energy_correction(hgh, n_electrons)
        @test reference_hgh ≈ eval_psp_energy_correction(upf, n_electrons) atol=1e-3 rtol=1e-3
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
                reference = quadgk(r -> integrand(psp, i, l, p, r),
                                   psp.rgrid[ir_start], psp.rgrid[ir_cut])[1]
                @test reference ≈ eval_psp_projector_fourier(psp, i, l, p) atol=1e-2 rtol=1e-2
            end
        end
    end
end

@testitem "Valence charge densities are consistent in real and Fourier space" #=
    =#    tags=[:psp] setup=[mPspUpf] begin
    using DFTK: eval_psp_density_valence_real, eval_psp_density_valence_fourier
    using SpecialFunctions: sphericalbesselj
    using QuadGK

    function integrand(psp, p, r)
        4π * r^2 * eval_psp_density_valence_real(psp, r) * sphericalbesselj(0, p * r)
    end
    for psp in values(mPspUpf.upf_pseudos)
        ir_start = iszero(psp.rgrid[1]) ? 2 : 1
        for p in (0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.)
            reference = quadgk(r -> integrand(psp, p, r), psp.rgrid[ir_start],
                               psp.rgrid[psp.ircut])[1]
            @test reference  ≈ eval_psp_density_valence_fourier(psp, p) atol=1e-2 rtol=1e-2
        end
    end
end

@testitem "Core charge densities are consistent in real and Fourier space" #=
    =#    tags=[:psp] setup=[mPspUpf] begin
    using DFTK: eval_psp_density_core_real, eval_psp_density_core_fourier
    using SpecialFunctions: sphericalbesselj
    using QuadGK

    function integrand(psp, p, r)
        4π * r^2 * eval_psp_density_core_real(psp, r) * sphericalbesselj(0, p * r)
    end
    for psp in values(mPspUpf.upf_pseudos)
        ir_start = iszero(psp.rgrid[1]) ? 2 : 1
        for p in (0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.)
            reference = quadgk(r -> integrand(psp, p, r), psp.rgrid[ir_start],
                               psp.rgrid[psp.ircut])[1]
            @test reference  ≈ eval_psp_density_core_fourier(psp, p) atol=1e-2 rtol=1e-2
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
        @test reference ≈ eval_psp_energy_correction(psp, 1) atol=1e-2
    end
end

@testitem "PSP guess density is positive" tags=[:psp] setup=[mPspUpf] begin
    using DFTK
    using LinearAlgebra

    lattice = 5 * I(3)
    positions = [zeros(3)]
    for (element, psp) in mPspUpf.upf_pseudos
        atoms = [ElementPsp(element; psp)]
        model = model_LDA(lattice, atoms, positions)
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
            atoms = [ElementPsp(element; psp)]
            model = model_LDA(lattice, atoms, positions)
            basis = PlaneWaveBasis(model; Ecut=22, kgrid=[2, 2, 2])
            ρ_val = guess_density(basis, ValenceDensityPseudo())
            Z_valence = sum(ρ_val) * model.unit_cell_volume / prod(basis.fft_size)
            @test Z_valence ≈ charge_ionic(psp) rtol=1e-5 atol=1e-5
        end
    end
end
