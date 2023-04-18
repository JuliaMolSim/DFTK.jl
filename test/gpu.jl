using Test
using CUDA
using DFTK
using StaticArrays
using Logging
import DFTK: mpi_sum
include("testcases.jl")

# These are not yet the best tests, but just to ensure our GPU support
# does not just break randomly

@testset "CUDA silicon functionality test" begin
    function run_problem(; architecture)
        model = model_PBE(silicon.lattice, silicon.atoms, silicon.positions)
        basis = PlaneWaveBasis(model; Ecut=10, kgrid=(3, 3, 3), architecture)

        # TODO Right now guess generation on the GPU does not work
        basis_cpu = PlaneWaveBasis(model; basis.Ecut, basis.kgrid)
        ρ_guess = guess_density(basis_cpu)
        ρ = DFTK.to_device(architecture, ρ_guess)

        self_consistent_field(basis; tol=1e-10, solver=scf_damping_solver(1.0), ρ)
    end

    scfres_cpu = run_problem(; architecture=DFTK.CPU())
    scfres_gpu = run_problem(; architecture=DFTK.GPU(CuArray))
    @test abs(scfres_cpu.energies.total - scfres_gpu.energies.total) < 1e-10
    @test norm(scfres_cpu.ρ - Array(scfres_gpu.ρ)) < 1e-10
end

@testset "CUDA iron functionality test" begin
    function run_problem(; architecture)
        magnetic_moments = [4.0]
        model = model_PBE(iron_bcc.lattice, iron_bcc.atoms, iron_bcc.positions;
                          magnetic_moments, smearing=Smearing.Gaussian(), temperature=1e-3)
        basis = PlaneWaveBasis(model; Ecut=20, kgrid=(4, 4, 4), architecture)

        # TODO Right now guess generation on the GPU does not work
        basis_cpu = PlaneWaveBasis(model; basis.Ecut, basis.kgrid)
        ρ_guess = guess_density(basis_cpu, magnetic_moments)
        ρ = DFTK.to_device(architecture, ρ_guess)
        # ρ = guess_density(basis, magnetic_moments)

        # TODO Bump tolerance a bit here ... still leads to NaNs unfortunately
        self_consistent_field(basis; ρ, tol=1e-7, mixing=KerkerMixing(),
                              solver=scf_damping_solver(1.0))
    end

    scfres_cpu = run_problem(; architecture=DFTK.CPU())
    scfres_gpu = run_problem(; architecture=DFTK.GPU(CuArray))
    @test abs(scfres_cpu.energies.total - scfres_gpu.energies.total) < 1e-7
    @test norm(scfres_cpu.ρ - Array(scfres_gpu.ρ)) < 1e-6
end

# TODO Fix direct minimization
# TODO Fix guess density generation
# TODO Float32
# TODO meta GGA
# TODO Aluminium with LdosMixing
# TODO Anderson acceleration
# TODO Norm-conserving pseudopotentials with non-linear core
