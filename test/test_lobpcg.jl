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

@testset "Diagonalisation of kinetic + local psp" begin
    Ecut = 15
    kpoints = [  # TODO Maybe this can be dropped later ... check
        [0,0,0],
        [0.229578295126352, 0.229578295126352, 0.000000000000000],
        [0.401762016471116, 0.401762016471116, 0.114789147563176],
        [0.280595694022912, 0.357121792398363, 0.280595694022912],
    ]
    pw = PlaneWaveBasis(lattice, kpoints, kweights, Ecut)
    hgh = PspHgh("si-pade-q4")

    psp_local = PotLocal(pw, positions, G -> DFTK.eval_psp_local_fourier(hgh, G))
    ham = Hamiltonian(pot_local=psp_local)
    res = lobpcg(ham, 5, tol=1e-8)

    ref = [
        [-3.974284745874937, -3.961611963017420, -0.440522571607315,
         -0.440522571440079, -0.440522571188859],
        [-3.973175753226023, -3.969984768978392, -0.454244667918821,
         -0.449473464486584, -0.446171055771318],
        [-3.969677220364769, -3.967842215005740, -0.457111205795918,
         -0.455519764434241, -0.435530921245836],
        [-3.973987865865504, -3.966377749500019, -0.455191427797800,
         -0.452082557539700, -0.443809546863165],
    ]
    for ik in 1:length(kpoints)
        @test res.λ[ik] ≈ ref[ik]
    end
end

@testset "Diagonalisation of a core Hamiltonian" begin
    Ecut = 15
    kpoints = [  # TODO Maybe this can be dropped later ... check
        [0,0,0],
        [0.229578295126352, 0.229578295126352, 0.000000000000000],
        [0.401762016471116, 0.401762016471116, 0.114789147563176],
        [0.280595694022912, 0.357121792398363, 0.280595694022912],
    ]
    pw = PlaneWaveBasis(lattice, kpoints, kweights, Ecut)
    hgh = PspHgh("si-pade-q4")

    psp_local = PotLocal(pw, positions, G -> DFTK.eval_psp_local_fourier(hgh, G))
    psp_nonlocal = PotNonLocal(pw, "Si" => positions, "Si" => hgh)
    ham = Hamiltonian(pot_local=psp_local, pot_nonlocal=psp_nonlocal)
    res = lobpcg(ham, 5, tol=1e-8)

    ref = [
        [0.067966083141126, 0.470570565964348, 0.470570565966131,
         0.470570565980086, 0.578593208202602],
        [0.105959302042882, 0.329211057388161, 0.410969129077501,
         0.451613404615669, 0.626861886537186],
        [0.158220020418481, 0.246761395395185, 0.383362969225928,
         0.422345289771740, 0.620994908900183],
        [0.138706889457309, 0.256605657080138, 0.431494061152506,
         0.437698454692923, 0.587160336593700]
    ]
    for ik in 1:length(kpoints)
        @test res.λ[ik] ≈ ref[ik]
    end
end
