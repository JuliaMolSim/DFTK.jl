@testitem "Testing HF/Hybrids using self_consistent_orbitals()" begin
using DFTK
using LinearAlgebra
using PseudoPotentialData
using AtomsBuilder

pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")

function subspace_difference(scfres1, scfres2)
    Δρ = 0
    for i in length(scfres1.ψ)
        nspace = Int(sum(scfres1.occupation[i])/2)
        ovlp = scfres1.ψ[i][:,1:nspace]' * scfres2.ψ[i][:,1:nspace]
        Δρ += abs((nspace-norm(ovlp)^2)/nspace)
    end
    Δρ = Δρ / length(scfres1.ψ)
    Δρ
end

function test_hybrid(testcase; Ecut=18, kgrid=[1,1,1], functionals = PBE0())
    model_lda = model_DFT(testcase;pseudopotentials, functionals=LDA())
    model_hf  = model_DFT(testcase;pseudopotentials, functionals)
    basis_lda = PlaneWaveBasis(model_lda;kgrid,Ecut)
    basis_hf  = PlaneWaveBasis(model_hf ;kgrid,Ecut)

    n_occ   = Int(model_hf.n_electrons / 2)
    n_bands = n_occ * 5

    scfres_lda = self_consistent_field(basis_lda)
    scfres_ref = self_consistent_field(basis_lda; maxiter=1, ρ=scfres_lda.ρ, nbandsalg=FixedBands(;n_bands_converge=n_bands))

    scfres = self_consistent_field(basis_hf; 
                                   tol         = 1e-8,
                                   maxiter     = 50,
                                   ρ           = scfres_lda.ρ,
                                   occupation  = scfres_lda.occupation,
                                   ψ           = scfres_lda.ψ)
    
    scfres_simple = self_consistent_orbitals(basis_hf; 
                                             tol         = 1e-12,
                                             maxiter     = 50,
                                             occupation  = scfres_lda.occupation,
                                             ψ           = scfres_lda.ψ,
                                             solver      = DFTK.OrbitalSimpleSolver())
    
    scfres_pcdiis = self_consistent_orbitals(basis_hf; 
                                             tol         = 1e-12,
                                             maxiter     = 50,
                                             occupation  = scfres_lda.occupation,
                                             ψ           = scfres_lda.ψ,
                                             solver      = DFTK.OrbitalPcdiisSolver(;reference=scfres_ref.ψ))

    @test subspace_difference(scfres, scfres_simple) < 1e-12
    @test subspace_difference(scfres_pcdiis, scfres_simple) < 1e-12
end

function test_hf(testcase; Ecut=18, kgrid=[1,1,1])
    model_lda = model_DFT(testcase;pseudopotentials, functionals=LDA())
    model_hf  = model_HF(testcase;pseudopotentials)
    basis_lda = PlaneWaveBasis(model_lda;kgrid,Ecut)
    basis_hf  = PlaneWaveBasis(model_hf ;kgrid,Ecut)

    n_occ   = Int(model_hf.n_electrons / 2)
    n_bands = n_occ * 5

    scfres_lda = self_consistent_field(basis_lda)
    scfres_ref = self_consistent_field(basis_lda; maxiter=1, ρ=scfres_lda.ρ, nbandsalg=FixedBands(;n_bands_converge=n_bands))

    scfres = self_consistent_field(basis_hf; 
                                   tol         = 1e-8,
                                   maxiter     = 50,
                                   ρ           = scfres_lda.ρ,
                                   occupation  = scfres_lda.occupation,
                                   ψ           = scfres_lda.ψ)
    
    scfres_simple = self_consistent_orbitals(basis_hf; 
                                             tol         = 1e-12,
                                             maxiter     = 50,
                                             occupation  = scfres_lda.occupation,
                                             ψ           = scfres_lda.ψ,
                                             solver      = DFTK.OrbitalSimpleSolver())

    scfres_pcdiis = self_consistent_orbitals(basis_hf; 
                                             tol         = 1e-12,
                                             maxiter     = 50,
                                             occupation  = scfres_lda.occupation,
                                             ψ           = scfres_lda.ψ,
                                             solver      = DFTK.OrbitalPcdiisSolver(;reference=scfres_ref.ψ))

    @test subspace_difference(scfres, scfres_simple) < 1e-12
    @test subspace_difference(scfres_pcdiis, scfres_simple) < 1e-12
end


@testset "silicon, PBE0" begin
    test_hybrid(bulk(:Si))
end

@testset "silicon, HF" begin
    test_hf(bulk(:Si))
end
end
