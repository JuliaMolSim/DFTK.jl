# Hellmann-Feynman stress
# via ForwardDiff & custom FFTW overloads on ForwardDiff.Dual
using Test
using DFTK
using ForwardDiff
import FiniteDiff

@testset "ForwardDiff stresses on silicon" begin
    function make_basis(a)
        lattice = a / 2 * [[0 1 1.];
                        [1 0 1.];
                        [1 1 0.]]
        Si = ElementPsp(:Si, psp=load_psp(:Si, functional="lda"))
        atoms = [Si => [ones(3)/8, -ones(3)/8]]
        model = model_DFT(lattice, atoms, [:lda_x, :lda_c_vwn]; symmetries=false)
        kgrid = [1, 1, 1] # k-point grid (Regular Monkhorst-Pack grid)
        Ecut = 7          # kinetic energy cutoff in Hartree
        PlaneWaveBasis(model, Ecut; kgrid=kgrid)
    end

    function compute_energy(scfres_ref, a)
        basis = make_basis(a)
        energies, H = energy_hamiltonian(basis, scfres_ref.ψ, scfres_ref.occupation; ρ=scfres_ref.ρ)
        energies.total
    end

    a = 10.26
    scfres = self_consistent_field(make_basis(a), is_converged=DFTK.ScfConvergenceDensity(1e-13))
    compute_energy(a) = compute_energy(scfres, a)

    ref = FiniteDiff.finite_difference_derivative(compute_energy, a)
    @test isapprox(ForwardDiff.derivative(compute_energy, a), ref, atol=1e-8)
end
