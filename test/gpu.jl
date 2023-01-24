using Test
using CUDA
using DFTK
include("testcases.jl")

# These are not yet the best tests, but just to ensure our GPU support
# does not just break randomly

@testset "CUDA silicon functionality test" begin
    function run_problem(; architecture)
        model = model_LDA(silicon.lattice, silicon.atoms, silicon.positions)
        basis = PlaneWaveBasis(model; Ecut=10, kgrid=(1, 1, 1), architecture)
        self_consistent_field(basis; tol=1e-10, mixing=SimpleMixing(),
                              solver=scf_damping_solver(0.4))
    end

    scfres_cpu = run_problem(; architecture=DFTK.CPU())
    scfres_gpu = run_problem(; architecture=DFTK.GPU(CuArray))
    @test abs(scfres_cpu.energies.total - scfres_gpu.energies.total) < 1e-10
    @test norm(scfres_cpu.ρ - Array(scfres_gpu.ρ)) < 1e-10
end

# TODO Float32

# TODO Disabled, because not yet supported ...
# @testset "CUDA silicon functionality test" begin
#     function run_problem(; architecture)
#         model = model_PBE(silicon.lattice, silicon.atoms, silicon.positions)
#         basis = PlaneWaveBasis(model; Ecut=10, kgrid=(3, 3, 3), architecture)
#         self_consistent_field(basis; is_converged=DFTK.ScfConvergenceDensity(1e-10),
#                               mixing=KerkerMixing(), solver=scf_damping_solver(1.0))
#     end
#
#     scfres_cpu = run_problem(; architecture=DFTK.CPU())
#     scfres_gpu = run_problem(; architecture=DFTK.GPU(CuArray))
#     @test abs(scfres_cpu.energies.total - scfres_gpu.energies.total) < 1e-10
#     @test norm(scfres_cpu.ρ - scfres_gpu.ρ) < 1e-9
# end

# TODO Disabled, because not yet supported ...
# @testset "CUDA iron functionality test" begin
#     function run_problem(; architecture)
#         magnetic_moments = [4.0]
#         model = model_PBE(iron_bcc.lattice, iron_bcc.atoms, iron_bcc.positions;
#                           magnetic_moments, smearing=Smearing.Gaussian(), temperature=1e-3)
#         basis = PlaneWaveBasis(model; Ecut=20, kgrid=(4, 4, 4), architecture)
#
#         ρ = guess_density(basis, magnetic_moments)
#         self_consistent_field(basis; ρ, is_converged=DFTK.ScfConvergenceDensity(1e-10),
#                               mixing=KerkerMixing(), solver=scf_damping_solver(1.0))
#     end
#
#     scfres_cpu = run_problem(; architecture=DFTK.CPU())
#     scfres_gpu = run_problem(; architecture=DFTK.GPU(CuArray))
#     @test abs(scfres_cpu.energies.total - scfres_gpu.energies.total) < 1e-10
#     @test norm(scfres_cpu.ρ - scfres_gpu.ρ) < 1e-9
# end
