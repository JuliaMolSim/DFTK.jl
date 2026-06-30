@testitem "Test pack/split_gdensity" tags=[:minimal] begin
    using AtomsIO
    using PseudoPotentialData
    using Logging
    using DFTK
    system = load_system("structures/AlVac_4.extxyz")
    pseudopotentials = Dict(:Al => "pseudos/Al_m.upf")
    τW = DFTK.von_weizsaecker_kinetic_energy_density

    model = model_DFT(system; pseudopotentials, functionals=r2SCAN(),
                      temperature=1e-2, smearing=Smearing.Gaussian())
    basis = PlaneWaveBasis(model; Ecut=10, kgrid=[1, 1, 1])

    # Initial guess
    ρ  = guess_density(basis)
    τ  = DFTK.guess_kinetic_energy_density(basis, ρ)

    # Test basic density properties on initial guess
    @test all(0 .≤ τ)
    @test all(0 .≤ ρ)
    @test all(τW(basis, ρ) .≤ τ)  # Hoffman-Ostenhof

    # Check that pack/split are identities
    D = DFTK.pack_gdensity(basis, ρ, τ)
    let
        ρs, τs = DFTK.split_gdensity(basis, D)
        @test ρs ≈ ρ atol=1e-12
        @test τs ≈ τ atol=1e-12
    end

    # One SCF step to get another set of (ρ/τ)
    scfres = with_logger(NullLogger()) do
        self_consistent_field(basis; ρ, τ, tol=1e-1, maxiter=1, callback=identity)
    end

    # Test basic density properties on SCF result
    @test all(0 .≤ scfres.τ)
    @test all(0 .≤ scfres.ρ)
    @test all(τW(basis, scfres.ρ) .≤ scfres.τ)  # Hoffman-Ostenhof

    # Check that pack/split are identities
    Dout = DFTK.pack_gdensity(basis, scfres.ρ, scfres.τ)
    let 
        ρs, τs = DFTK.split_gdensity(basis, Dout)
        @test ρs ≈ scfres.ρ atol=1e-12
        @test τs ≈ scfres.τ atol=1e-12
    end
    let D = DFTK.pack_gdensity_flat_(basis, scfres.ρ, scfres.τ)
        ρs, τs = DFTK.split_gdensity_flat_(basis, D)
        @test ρs ≈ scfres.ρ atol=1e-12
        @test τs ≈ scfres.τ atol=1e-12
    end

    # Take a few random, convex linear combinations
    for α in (rand(), rand(), rand())
        Dnew = D + α * (Dout - D)
        ρnew, τnew = DFTK.split_gdensity(basis, Dnew)

        @test all(0 .≤ τnew)
        @test all(0 .≤ ρnew)
        @test all(τW(basis, ρnew) .≤ τnew)  # Hoffman-Ostenhof
    end

    # A few more SCF steps to see SCF is stable to (ρ/τ) properties
    let scfres = self_consistent_field(basis; ρ, τ, tol=1e-1)
        @test all(0 .≤ scfres.τ)
        @test all(0 .≤ scfres.ρ)
        @test all(τW(basis, scfres.ρ) .≤ scfres.τ)  # Hoffman-Ostenhof
    end
end
