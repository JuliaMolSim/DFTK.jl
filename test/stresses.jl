using Test
using DFTK
using ForwardDiff
import FiniteDiff
include("testcases.jl")

# Hellmann-Feynman stress
# via ForwardDiff & custom FFTW overloads on ForwardDiff.Dual

@testset "ForwardDiff stresses on silicon" begin
    function make_basis(lattice, symmetry)
        Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
        atoms = [Si => silicon.positions]
        model = model_DFT(lattice, atoms, [:lda_x, :lda_c_vwn]; symmetries=symmetry)
        kgrid = [1, 1, 1]
        Ecut = 7
        PlaneWaveBasis(model; Ecut, kgrid)
    end

    function recompute_energy(lattice, symmetry)
        basis = make_basis(lattice, symmetry)
        scfres = self_consistent_field(basis, is_converged=DFTK.ScfConvergenceDensity(1e-13))
        energies, H = energy_hamiltonian(basis, scfres.ψ, scfres.occupation; ρ=scfres.ρ)
        energies.total
    end

    function hellmann_feynman_energy(scfres, lattice, symmetry)
        basis = make_basis(lattice, symmetry)
        ρ = DFTK.compute_density(basis, scfres.ψ, scfres.occupation)
        energies, H = energy_hamiltonian(basis, scfres.ψ, scfres.occupation; ρ=ρ)
        energies.total
    end


    a = 10.0 # slightly compressed
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    scfres = self_consistent_field(make_basis(lattice, true), is_converged=DFTK.ScfConvergenceDensity(1e-13))
    scfres_nosym = self_consistent_field(make_basis(lattice, false), is_converged=DFTK.ScfConvergenceDensity(1e-13))
    stresses = compute_stresses(scfres)
    @test isapprox(stresses, compute_stresses(scfres_nosym), atol=1e-10)

    dir = randn(3, 3)

    dE_stresses = dot(dir, stresses) * scfres.basis.model.unit_cell_volume
    ref_recompute = FiniteDiff.finite_difference_derivative(ε -> recompute_energy(lattice + ε*dir*lattice, false), 0.0)
    ref_HF = FiniteDiff.finite_difference_derivative(ε -> hellmann_feynman_energy(scfres_nosym, lattice+ε*dir*lattice, false), 0.0)
    # FD_recompute = ForwardDiff.derivative(ε -> recompute_energy(lattice+ε*dir*lattice, false), 0.0)
    FD_HF = ForwardDiff.derivative(ε -> hellmann_feynman_energy(scfres_nosym, lattice+ε*(dir*lattice), false), 0.0)

    @test isapprox(ref_HF, ref_recompute, atol=1e-5)
    @test isapprox(ref_HF, FD_HF, atol=1e-5)
    @test isapprox(dE_stresses, ref_recompute, atol=1e-5)
end
