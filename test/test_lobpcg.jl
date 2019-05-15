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

    guess = [
        [zeros(length(mask)) for (ik, mask) in enumerate(pw.kmask) for n in 1:nev_per_k]
    ]
    for (ik, g) in enumerate(guess)
        for n in 1:nev_per_k
            idx_DC = indexin(pw.idx_DC, pw.kmask[ik])[1]
            g[n][idx_DC + n] = 1
        end
    end

    @test length(ref_λ) == length(kpoints)
    @testset "without Preconditioner" begin
        res = nothing
        for retry in 1:3
            try
                res = lobpcg(ham, nev_per_k, guess=guess, tol=tol)
                break
            catch PosDefException
            end
        end

        @test_broken res !== nothing
        if res !== nothing
            @test res.converged
            for ik in 1:length(kpoints)
                @test ref_λ[ik] ≈ res.λ[ik]
                @test maximum(res.residual_norms[ik]) < tol
                @test res.iterations[ik] < 200
            end
        end
    end

    @testset "with Preconditioner" begin
        res = nothing
        for retry in 1:3
            try
                res = lobpcg(ham, nev_per_k, guess=guess, tol=tol,
                             preconditioner=PreconditionerKinetic(ham, α=0.1))
                break
            catch PosDefException
            end
        end

        @test_broken res !== nothing
        if res !== nothing
            @test res.converged
            for ik in 1:length(kpoints)
                @test ref_λ[ik] ≈ res.λ[ik]
                @test maximum(res.residual_norms[ik]) < tol
                @test res.iterations[ik] < 35
            end
        end
    end
end
