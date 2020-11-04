using Test
using DFTK

include("./testcases.jl")


@testset "Diagonalization of a free-electron Hamiltonian" begin
    # Construct a free-electron Hamiltonian
    Ecut = 5
    fft_size = [15, 15, 15]
    model = Model(silicon.lattice, n_electrons=silicon.n_electrons, terms=[Kinetic()])  # free-electron model
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)
    ham = Hamiltonian(basis)

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

    @test length(ref_λ) == length(silicon.kcoords)
    @testset "without Preconditioner" begin
        res = diagonalize_all_kblocks(lobpcg_hyper, ham, nev_per_k, tol=tol,
                                      prec_type=nothing, interpolate_kpoints=false)

        @test res.converged
        for ik in 1:length(basis.kpoints)
            @test ref_λ[basis.krange_thisproc[ik]] ≈ res.λ[ik]
            @test maximum(res.residual_norms[ik]) < 100tol  # TODO Why the 100?
            @test res.iterations[ik] < 200
        end
    end

    @testset "with Preconditioner" begin
        res = diagonalize_all_kblocks(lobpcg_hyper, ham, nev_per_k, tol=tol,
                                      prec_type=PreconditionerTPA, interpolate_kpoints=false)

        @test res.converged
        for ik in 1:length(basis.kpoints)
            @test ref_λ[basis.krange_thisproc[ik]] ≈ res.λ[ik]
            @test maximum(res.residual_norms[ik]) < 100tol  # TODO Why the 100?
            @test res.iterations[ik] < 50
        end
    end
end

if !isdefined(Main, :FAST_TESTS) || !FAST_TESTS
    @testset "Diagonalization of kinetic + local PSP" begin
        Ecut = 25
        fft_size = [33, 33, 33]

        Si = ElementPsp(silicon.atnum, psp=load_psp("hgh/lda/si-q4"))
        model = Model(silicon.lattice; atoms=[Si => silicon.positions],
                      terms=[Kinetic(),AtomicLocal()])
        basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops;
                               fft_size=fft_size)
        ham = Hamiltonian(basis)

        res = diagonalize_all_kblocks(lobpcg_hyper, ham, 6, tol=1e-8)

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
        for ik in 1:length(basis.kpoints)
            @test res.λ[ik][1:5] ≈ ref[basis.krange_thisproc[ik]] atol=5e-7
        end
    end
end

@testset "Diagonalization of a core Hamiltonian" begin
    Ecut = 10
    fft_size = [21, 21, 21]

    Si = ElementPsp(silicon.atnum, psp=load_psp("hgh/lda/si-q4"))
    model = Model(silicon.lattice; atoms=[Si => silicon.positions],
                  terms=[Kinetic(), AtomicLocal(), AtomicNonlocal()])

    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)
    ham = Hamiltonian(basis)

    res = diagonalize_all_kblocks(lobpcg_hyper, ham, 5, tol=1e-8, interpolate_kpoints=false)
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
    for ik in 1:length(basis.kpoints)
        @test res.λ[ik] ≈ ref[basis.krange_thisproc[ik]] atol=0.02
    end
end

@testset "Full diagonalization of a LDA Hamiltonian" begin
    Ecut = 2

    Si = ElementPsp(silicon.atnum, psp=load_psp("hgh/lda/si-q4"))
    model = model_DFT(silicon.lattice, [Si => silicon.positions], :lda_xc_teter93)
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops)
    ham = Hamiltonian(basis; ρ=guess_density(basis))

    res1 = diagonalize_all_kblocks(lobpcg_hyper, ham, 5, tol=1e-8, interpolate_kpoints=false)
    res2 = diagonalize_all_kblocks(diag_full, ham, 5, tol=1e-8, interpolate_kpoints=false)
    for ik in 1:length(basis.kpoints)
        @test res1.λ[ik] ≈ res2.λ[ik] atol=1e-6
    end
end
