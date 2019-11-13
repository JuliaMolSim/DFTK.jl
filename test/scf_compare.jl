using Test
using DFTK

include("testcases.jl")

@testset "Compare different SCF algorithms" begin
    Ecut = 2
    n_bands = 6
    fft_size = [9, 9, 9]
    tol = 1e-6

    Si = Species(silicon.atnum, psp=load_psp(silicon.psp))
    model = model_reduced_hf(silicon.lattice, Si => silicon.positions)
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)

    # Run nlsolve without guess
    scfres = self_consistent_field!(Hamiltonian(basis), n_bands, tol=tol,
                                    solver=scf_nlsolve_solver())
    ρ_nl = fourier(scfres.ρ)

    # Run other SCFs with SAD guess
    ρ0 = guess_density(basis, Si => silicon.positions)
    for solver in (scf_nlsolve_solver, scf_damping_solver, scf_anderson_solver,
                   scf_CROP_solver)
        println("Testing $solver")
        scfres = self_consistent_field!(Hamiltonian(basis, ρ0), n_bands, tol=tol, solver=solver())
        ρ_alg = fourier(scfres.ρ)
        @test maximum(abs.(ρ_alg - ρ_nl)) < 30tol
    end
end
