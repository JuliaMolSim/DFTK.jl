@testitem "apply_kernel_ad" setup=[TestCases] begin
    using DFTK
    using ForwardDiff
    using LinearAlgebra
    using Random

    testcase = TestCases.silicon
    atoms = fill(ElementPsp(testcase.atnum, load_psp(testcase.psp_upf)), 2)
    magnetic = true
    if magnetic
        magnetic_moments = [1, -1]
    else
        magnetic_moments = []
    end
    model = model_DFT(testcase.lattice, atoms, testcase.positions;
                      functionals=PBE(), magnetic_moments)
    basis = PlaneWaveBasis(model; Ecut=40, kgrid=MonkhorstPack([2, 2, 2]))

    if magnetic
        ρ = guess_density(basis, magnetic_moments)
    else
        ρ = guess_density(basis)
    end
    scfres = self_consistent_field(basis; ρ, tol=1e-6)
    ρ = scfres.ρ

    # Extract the XC term
    term = only(filter(t -> t isa DFTK.TermXc, basis.terms))

    # Random density perturbation
    Random.seed!(1234)
    δρ = randn(size(ρ))

    
    function man_kernel(ntimes)
        for i in 1:ntimes
            DFTK.apply_kernel(term, basis, δρ; ρ)
        end
        DFTK.apply_kernel(term, basis, δρ; ρ)
    end
    function ad_kernel(ntimes)
        for i in 1:ntimes
            DFTK.apply_kernel_ad(term, basis, δρ; ρ)
        end
        DFTK.apply_kernel_ad(term, basis, δρ; ρ)
    end

    N = 10
    man_kernel(N)
    ad_kernel(N)

    tman = @elapsed δV_manual = man_kernel(N)
    tad = @elapsed δV_ad     = ad_kernel(N)

    @show tman tad basis.fft_size

    @test norm(δV_manual - δV_ad) / norm(δV_manual) < 1e-6
end
