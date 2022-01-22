using Test
using DFTK
using ForwardDiff
import FiniteDiff
using MPI
include("testcases.jl")

# Hellmann-Feynman stress
# via ForwardDiff & custom FFTW overloads on ForwardDiff.Dual

@testset "ForwardDiff stresses on silicon" begin
    function make_basis(lattice, symmetries)
        Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
        atoms = [Si => silicon.positions]
        model = model_PBE(lattice, atoms; symmetries)
        kgrid = [3, 3, 1]
        Ecut = 7
        PlaneWaveBasis(model; Ecut, kgrid)
    end

    function recompute_energy(lattice, symmetries)
        basis = make_basis(lattice, symmetries)
        scfres = self_consistent_field(basis; is_converged=DFTK.ScfConvergenceDensity(1e-13))
        energies, H = energy_hamiltonian(basis, scfres.ψ, scfres.occupation; ρ=scfres.ρ)
        energies.total
    end

    function hellmann_feynman_energy(scfres, lattice, symmetries)
        basis = make_basis(lattice, symmetries)
        ρ = DFTK.compute_density(basis, scfres.ψ, scfres.occupation)
        energies, H = energy_hamiltonian(basis, scfres.ψ, scfres.occupation; ρ=ρ)
        energies.total
    end


    a = 10.0  # slightly compressed
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    is_converged = DFTK.ScfConvergenceDensity(1e-13)
    scfres = self_consistent_field(make_basis(lattice, true); is_converged)
    scfres_nosym = self_consistent_field(make_basis(lattice, false); is_converged)
    stresses = compute_stresses(scfres)
    @test isapprox(stresses, compute_stresses(scfres_nosym), atol=1e-10)

    dir = MPI.bcast(randn(3, 3), 0, MPI.COMM_WORLD)

    dE_stresses = dot(dir, stresses) * scfres.basis.model.unit_cell_volume
    ref_recompute = FiniteDiff.finite_difference_derivative(0.0) do ε
        recompute_energy(lattice + ε*dir*lattice, false)
    end
    ref_HF = FiniteDiff.finite_difference_derivative(0.0) do ε
        hellmann_feynman_energy(scfres_nosym, lattice+ε*dir*lattice, false)
    end
    FD_HF = ForwardDiff.derivative(0.0) do ε
        hellmann_feynman_energy(scfres_nosym, lattice+ε*(dir*lattice), false)
    end

    @test isapprox(ref_HF, ref_recompute, atol=1e-5)
    @test isapprox(ref_HF, FD_HF, atol=1e-5)
    @test isapprox(dE_stresses, ref_recompute, atol=1e-5)
end
