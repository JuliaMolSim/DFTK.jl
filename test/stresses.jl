using Test
using DFTK
using ForwardDiff
import FiniteDiff
include("testcases.jl")

# Hellmann-Feynman stress
# via ForwardDiff & custom FFTW overloads on ForwardDiff.Dual

@testset "ForwardDiff stresses on silicon" begin
    function make_basis(a)
        lattice = a / 2 * [[0 1 1.];
                           [1 0 1.];
                           [1 1 0.]]
        Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
        atoms = [Si => silicon.positions]
        model = model_DFT(lattice, atoms, [:lda_x, :lda_c_vwn], symmetries=false)
        kgrid = [1, 1, 1]
        Ecut = 7
        PlaneWaveBasis(model, Ecut; kgrid=kgrid)
    end

    function recompute_energy(a)
        basis = make_basis(a)
        scfres = self_consistent_field(basis, is_converged=DFTK.ScfConvergenceDensity(1e-13))
        energies, H = energy_hamiltonian(basis, scfres.ψ, scfres.occupation; ρ=scfres.ρ)
        energies.total
    end

    function hellmann_feynman_energy(scfres_ref, a)
        basis = make_basis(a)
        ρ = DFTK.compute_density(basis, scfres_ref.ψ, scfres_ref.occupation)
        energies, H = energy_hamiltonian(basis, scfres_ref.ψ, scfres_ref.occupation; ρ=ρ)
        energies.total
    end

    a = 10.26
    scfres = self_consistent_field(make_basis(a), is_converged=DFTK.ScfConvergenceDensity(1e-13))
    hellmann_feynman_energy(a) = hellmann_feynman_energy(scfres, a)

    stresses = compute_stresses(scfres)
    ref_recompute = FiniteDiff.finite_difference_derivative(recompute_energy, a)
    ref_hf = FiniteDiff.finite_difference_derivative(hellmann_feynman_energy, a)
    s_hf = ForwardDiff.derivative(hellmann_feynman_energy, a)


    @test isapprox(ref_hf, ref_recompute, atol=1e-4)
    @test isapprox(s_hf, ref_hf, atol=1e-8)
    @test isapprox(tr(stresses), ref_hf, atol=1e-8)
end
