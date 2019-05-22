include("testcases_silicon.jl")

@testset "Diagonalisation of a free-electron Hamiltonian" begin
    # Construct a free-electron Hamiltonian
    Ecut = 5
    pw = PlaneWaveBasis(lattice, kpoints, kweights, Ecut)
    ham = Hamiltonian(pw)

    tol = 1e-8
    nev_per_k = 10
    ref_λ = [
        [0.00000000000, 0.56219939834, 0.56219939834, 0.56219939834, 0.56219939834,
         0.56219939834, 0.56219939834, 0.56219939834, 0.56219939834, 0.74959919778],
        [0.06246659981, 0.24986639926, 0.49973279852, 0.49973279852, 0.49973279852,
         0.56219939834, 0.56219939834, 0.56219939834, 0.74959919778, 0.74959919778],
        [0.08328879975, 0.33315519901, 0.39562179883, 0.39562179883, 0.39562179883,
         0.39562179883, 0.83288799753, 0.83288799754, 0.83288799754, 0.83288799754],
        [0.16657759951, 0.22904419932, 0.22904419932, 0.41644399877, 0.41644399877,
         0.66631039803, 0.72877699784, 0.72877699784, 0.72877699784, 0.72877699784],
    ]

    @test length(ref_λ) == length(kpoints)
    @testset "without Preconditioner" begin
        res = lobpcg(ham, nev_per_k, tol=tol)

        @test res.converged
        for ik in 1:length(kpoints)
            @test ref_λ[ik] ≈ res.λ[ik]
            @test maximum(res.residual_norms[ik]) < 100tol  # TODO Why the 100?
            @test res.iterations[ik] < 200
        end
    end

    @testset "with Preconditioner" begin
        res = lobpcg(ham, nev_per_k, tol=tol,
                     preconditioner=PreconditionerKinetic(ham, α=0.1))

        @test res.converged
        for ik in 1:length(kpoints)
            @test ref_λ[ik] ≈ res.λ[ik]
            @test maximum(res.residual_norms[ik]) < 100tol  # TODO Why the 100?
            @test res.iterations[ik] < 50
        end
    end
end
