@testitem "Virtual crystal with combination 0.5+0.5" setup=[TestCases] tags=[:psp] begin
    using DFTK
    using DFTK: PspLinComb
    using LinearAlgebra
    silicon = TestCases.silicon
    pd_lda_family = TestCases.pd_lda_family

    Si  = ElementPsp(:Si, pd_lda_family)
    mix = virtual_crystal_approximation(0.5, Si, 0.5, Si)

    @testset "Basic properties" begin
        @test mix.psp isa PspLinComb
        @test length(mix.psp.coefficients) == 2
        @test mix.psp.pseudos[1] == Si.psp
        @test mix.psp.pseudos[2] == Si.psp

        @test mix.psp.lmax == Si.psp.lmax
        @test mix.family   == pd_lda_family
        @test charge_ionic(mix.psp) == charge_ionic(Si.psp)
        @test DFTK.count_n_proj_radial(mix.psp) == 2DFTK.count_n_proj_radial(Si.psp)
    end

    @testset "Local potentials / densities agree on element" begin
        functions = [
            DFTK.local_potential_fourier,
            DFTK.local_potential_real,
            DFTK.valence_charge_density_fourier,
            DFTK.core_charge_density_fourier,
        ]
        for fun in functions
            for p in (0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.)
                @test fun(mix, p) == fun(Si, p)
            end  # p
        end
    end

    @testset "Local potentials / densities agree on psp" begin
        functions = [
            DFTK.eval_psp_local_real
            DFTK.eval_psp_local_fourier
            DFTK.eval_psp_density_valence_real
            DFTK.eval_psp_density_valence_fourier
            DFTK.eval_psp_density_core_real
            DFTK.eval_psp_density_core_fourier
        ]
        for fun in functions
            for p in (0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.)
                @test fun(mix.psp, p) == fun(Si.psp, p)
            end  # p
        end
    end

    @testset "Hamiltonian application" begin
        Ecut  = 15
        kgrid = [2, 2, 2]
        n_bands = 6

        atoms_orig = [Si, Si]
        model_orig = model_DFT(silicon.lattice, atoms_orig, silicon.positions; functionals=LDA())
        basis_orig = PlaneWaveBasis(model_orig; Ecut, kgrid)
        ham_orig   = Hamiltonian(basis_orig; ρ=guess_density(basis_orig))

        atoms_mix  = [mix, mix]
        model_mix  = model_DFT(silicon.lattice, atoms_mix, silicon.positions; functionals=LDA())
        basis_mix  = PlaneWaveBasis(model_mix; Ecut, kgrid)
        ham_mix    = Hamiltonian(basis_mix; ρ=guess_density(basis_mix))

        ψ = DFTK.random_orbitals(ham_orig.basis, n_bands)
        @test maximum(Hk -> maximum(abs, Hk), ham_orig*ψ - ham_mix*ψ) < 1e-14
    end
end

@testitem "Test virtual_crystal_approximation" setup=[TestCases] tags=[:psp] begin
    using DFTK
    using DFTK: Element
    silicon = TestCases.silicon
    pd_lda_family = TestCases.pd_lda_family
    Ge_psp = load_psp(pd_lda_family, :Ge)
    Sn_psp = load_psp(pd_lda_family, :Sn)
    Ge = ElementPsp(:Ge, pd_lda_family)
    Sn = ElementPsp(:Sn, pd_lda_family)
    Pb = ElementPsp(:Pb, pd_lda_family)

    @testset "Pseudopotentials (2-psp form)" begin
        vpsp = virtual_crystal_approximation(0.7, Ge_psp, 0.3, Sn_psp; symbols=[:Ge, :Sn])
        @test vpsp.coefficients == [0.7, 0.3]
        @test vpsp.pseudos == [Ge_psp, Sn_psp]
        @test charge_ionic(vpsp) == charge_ionic(Ge_psp)
    end

    @testset "Elements (array form; properties)" begin
        coeffs   = [0.2, 0.4, 0.1, 0.3]
        elements = [ Ge,  Sn,  Pb,  Sn]
        virtual  = virtual_crystal_approximation(coeffs, elements)
        @test virtual.psp.coefficients == coeffs
        @test virtual.psp.pseudos      == [el.psp for el in elements]
        @test virtual.family  == pd_lda_family
        @test virtual.species == :X
        @test virtual.mass    == 0.2Ge.mass + 0.4Sn.mass + 0.1Pb.mass + 0.3Sn.mass

        @test charge_ionic(virtual)   == charge_ionic(Ge)
        @test n_elec_valence(virtual) == n_elec_valence(Ge)
        @test_broken n_elec_core(virtual) == n_elec_core(Ge)
    end

    @testset "Hamiltonian application" begin
        n_bands = 5
        Ecut    = 15
        kgrid   = (2, 2, 2)

        function make_ham(el::Element)
            model = model_atomic(silicon.lattice, [el, el], silicon.positions)
            basis = PlaneWaveBasis(model; Ecut, kgrid)
            Hamiltonian(basis; ρ=guess_density(basis))
        end

        vv = virtual_crystal_approximation(0.7, Ge, 0.3, Sn)
        H_vv = make_ham(vv)
        H_Ge = make_ham(Ge)
        H_Sn = make_ham(Sn)

        ψ = DFTK.random_orbitals(H_vv.basis, n_bands)
        Hψ_vv  = H_vv * ψ
        Hψ_Ge  = H_Ge * ψ
        Hψ_Sn  = H_Sn * ψ
        Hψ_ref = 0.7 * Hψ_Ge + 0.3 * Hψ_Sn

        @test maximum(Hk -> maximum(abs, Hk), Hψ_vv - Hψ_ref) < 1e-14
    end
end

@testitem "Potentials are consistent in real and Fourier space" tags=[:psp] setup=[TestCases] begin
    using DFTK
    using DFTK: eval_psp_local_real, eval_psp_local_fourier
    using QuadGK
    pd_lda_family = TestCases.pd_lda_family
    Ge_psp = load_psp(pd_lda_family, :Ge)
    Sn_psp = load_psp(pd_lda_family, :Sn)

    r_first = Ge_psp.rgrid[begin]
    r_last  = Ge_psp.rgrid[Ge_psp.ircut]
    @assert r_first == Sn_psp.rgrid[begin]
    @assert r_last  == Sn_psp.rgrid[Sn_psp.ircut]
    function integrand(psp, p, r)
        4π * (eval_psp_local_real(psp, r) + charge_ionic(psp) / r) * sin(p * r) / (p * r) * r^2
    end
    for c in [0.2, 0.5, 0.7]
        psp = virtual_crystal_approximation(c, Ge_psp, 1-c, Sn_psp; symbols=[:Ge, :Sn])
        for p in (0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10.)
            reference = quadgk(r -> integrand(psp, p, r), r_first, r_last)[1]
            correction = 4π * charge_ionic(psp) / p^2
            @test (reference - correction) ≈ eval_psp_local_fourier(psp, p) atol=1e-2 rtol=1e-2
        end
    end
end

@testitem "PSP energy correction is consistent with fourier-space potential" tags=[:psp] setup=[TestCases] begin
    using DFTK
    using DFTK: eval_psp_local_fourier, eval_psp_energy_correction
    pd_lda_family = TestCases.pd_lda_family
    Ge_psp = load_psp(pd_lda_family, :Ge)
    Sn_psp = load_psp(pd_lda_family, :Sn)

    p_small = 1e-2    # We are interested in p→0 term
    for c in [0.2, 0.5, 0.7]
        psp = virtual_crystal_approximation(c, Ge_psp, 1-c, Sn_psp; symbols=[:Ge, :Sn])
        coulomb = -4π * charge_ionic(psp) / p_small^2
        reference = eval_psp_local_fourier(psp, p_small) - coulomb
        @test reference ≈ eval_psp_energy_correction(psp) atol=1e-2
    end
end
