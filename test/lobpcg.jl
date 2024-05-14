@testitem "Diagonalization of a free-electron Hamiltonian" setup=[TestCases] begin
    using DFTK
    silicon = TestCases.silicon

    # Construct a free-electron Hamiltonian
    fft_size = [15, 15, 15]
    model = Model(silicon.lattice, silicon.atoms, silicon.positions; terms=[Kinetic()])
    basis = PlaneWaveBasis(model; Ecut=5, silicon.kgrid, fft_size)
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

    @test length(ref_λ) == length(silicon.kgrid)
    @testset "without Preconditioner" begin
        res = diagonalize_all_kblocks(lobpcg_hyper, ham, nev_per_k; tol=1e-4,
                                      prec_type=nothing, interpolate_kpoints=false)

        @test res.converged
        for ik = 1:length(basis.kpoints)
            @test ref_λ[basis.krange_thisproc_allspin[ik]] ≈ res.λ[ik] atol = 1e-4
            @test maximum(res.residual_norms[ik]) < 1e-4
            @test res.n_iter[ik] < 200
        end
    end

    @testset "with Preconditioner" begin
        res = diagonalize_all_kblocks(lobpcg_hyper, ham, nev_per_k; tol,
                                      prec_type=PreconditionerTPA, interpolate_kpoints=false)

        @test res.converged
        for ik = 1:length(basis.kpoints)
            @test ref_λ[basis.krange_thisproc_allspin[ik]] ≈ res.λ[ik]
            @test maximum(res.residual_norms[ik]) < 100tol  # TODO Why the 100?
            @test res.n_iter[ik] < 50
        end
    end
end

@testitem "Diagonalization of kinetic + local PSP" tags=[:slow] setup=[TestCases] begin
    using DFTK
    silicon = TestCases.silicon

    Si = ElementPsp(silicon.atnum; psp=load_psp("hgh/lda/si-q4"))
    model = Model(silicon.lattice, silicon.atoms, silicon.positions;
                  terms=[Kinetic(),AtomicLocal()])
    basis = PlaneWaveBasis(model; Ecut=25, silicon.kgrid, fft_size=[33, 33, 33])
    ham = Hamiltonian(basis)

    res = diagonalize_all_kblocks(lobpcg_hyper, ham, 6; tol=1e-8)
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
    for ik = 1:length(basis.kpoints)
        @test res.λ[ik][1:5] ≈ ref[basis.krange_thisproc_allspin[ik]] atol=5e-7
    end
end

@testitem "Diagonalization of a core Hamiltonian" setup=[TestCases] begin
    using DFTK
    silicon = TestCases.silicon

    Si = ElementPsp(silicon.atnum; psp=load_psp("hgh/lda/si-q4"))
    model = Model(silicon.lattice, silicon.atoms, silicon.positions;
                  terms=[Kinetic(), AtomicLocal(), AtomicNonlocal()])

    basis = PlaneWaveBasis(model; Ecut=10, silicon.kgrid, fft_size=[21, 21, 21])
    ham = Hamiltonian(basis)

    res = diagonalize_all_kblocks(lobpcg_hyper, ham, 5; tol=1e-8, interpolate_kpoints=false)
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
    for ik = 1:length(basis.kpoints)
        @test res.λ[ik] ≈ ref[basis.krange_thisproc_allspin[ik]] atol=0.02
    end
end

@testitem "Full diagonalization of a LDA Hamiltonian" setup=[TestCases] begin
    using DFTK
    silicon = TestCases.silicon

    Si = ElementPsp(silicon.atnum; psp=load_psp("hgh/lda/si-q4"))
    model = model_DFT(silicon.lattice, silicon.atoms, silicon.positions, :lda_xc_teter93)
    basis = PlaneWaveBasis(model; Ecut=2, silicon.kgrid)
    ham = Hamiltonian(basis; ρ=guess_density(basis))

    res1 = diagonalize_all_kblocks(lobpcg_hyper, ham, 5; tol=1e-8, interpolate_kpoints=false)
    res2 = diagonalize_all_kblocks(diag_full, ham, 5; tol=1e-8, interpolate_kpoints=false)
    for ik = 1:length(basis.kpoints)
        @test res1.λ[ik] ≈ res2.λ[ik] atol=1e-6
    end
end

@testitem "LOBPCG Internal data structures" setup=[TestCases] begin
    using DFTK
    using LinearAlgebra

    a1 = rand(10, 5)
    a2 = rand(10, 2)
    a3 = rand(10, 7)
    b1 = rand(10, 6)
    b2 = rand(10, 2)
    A = hcat(a1,a2,a3)
    B = hcat(b1,b2)
    Ablock = DFTK.LazyHcat(a1, a2, a3)
    Bblock = DFTK.LazyHcat(b1, b2)
    @test Ablock'*Bblock ≈ A'*B
    @test Ablock'*B ≈ A'*B

    C = rand(14, 4)
    @test Ablock*C ≈ A*C

    D = rand(10, 4)
    @test mul!(D,Ablock, C, 1, 0) ≈ A*C
end
