include("testcases_silicon.jl")

@testset "Diagonalisation of a free-electron Hamiltonian" begin
    # Construct a free-electron Hamiltonian
    Ecut = 5
    grid_size = [15, 15, 15]
    pw = PlaneWaveBasis(lattice, grid_size, Ecut, kpoints, kweights)
    ham = Hamiltonian(Kinetic(pw))

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

@testset "Diagonalisation of kinetic + local psp" begin
    if ! running_in_ci
        Ecut = 25
        grid_size = [33, 33, 33]
        pw = PlaneWaveBasis(lattice, grid_size, Ecut, kpoints, kweights)
        hgh = PspHgh("si-pade-q4")

        psp_local = PotLocal(pw, positions, G -> DFTK.eval_psp_local_fourier(hgh, G),
                             coords_are_cartesian=true)
        ham = Hamiltonian(pot_local=psp_local)
        res = lobpcg(ham, 6, tol=1e-8)

        ref = [
            [-4.087198659513310, -4.085326314828677, -0.506869382308294,
             -0.506869382280876, -0.506869381798614],
            [-4.085824585443292, -4.085418874576503, -0.509716820984169,
             -0.509716820267449, -0.508545832298541],
            [-4.086645155119840, -4.085209948598607, -0.514320642233337,
             -0.514320641863231, -0.499373272772206],
            [-4.085991608422304, -4.085039856878318, -0.517299903754010,
             -0.513805498246478, -0.497036479690380]
        ]
        for ik in 1:length(kpoints)
            @test res.λ[ik][1:5] ≈ ref[ik] atol=5e-7
        end
    else
        println("Skipping diagonalisation of kinetic + local psp since running from CI")
    end
end

@testset "Diagonalisation of a core Hamiltonian" begin
    Ecut = 10
    grid_size = [21, 21, 21]
    pw = PlaneWaveBasis(lattice, grid_size, Ecut, kpoints, kweights)
    hgh = PspHgh("si-pade-q4")

    psp_local = PotLocal(pw, positions, G -> DFTK.eval_psp_local_fourier(hgh, G),
                         coords_are_cartesian=true)
    psp_nonlocal = PotNonLocal(pw, "Si" => positions, "Si" => hgh)
    ham = Hamiltonian(pot_local=psp_local, pot_nonlocal=psp_nonlocal)
    res = lobpcg(ham, 5, tol=1e-8)

    ref = [
        [0.067955741977536, 0.470244204908046, 0.470244204920801,
         0.470244204998022, 0.578392222232969],
        [0.111089041747288, 0.304724122513625, 0.445322298067717,
         0.445322298101198, 0.584713217756577],
        [0.129419322499919, 0.293174377882115, 0.411932220567084,
         0.411932220611853, 0.594921264868345],
        [0.168662148987539, 0.238552367551507, 0.370743978236562,
         0.418387442903058, 0.619797227001203],
    ]
    for ik in 1:length(kpoints)
        @test res.λ[ik] ≈ ref[ik] atol=0.02
    end
end
