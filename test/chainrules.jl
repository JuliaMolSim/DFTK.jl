using Test
using DFTK
using FiniteDiff
using Zygote
include("testcases.jl")

# Testing rrules needed for reverse Hellmann-Feynman stress

@testset "ChainRules" begin

    function has_consistent_derivative(f, a)
        d1 = Zygote.gradient(f, a)[1]
        d2 = FiniteDiff.finite_difference_derivative(f, a)
        isapprox(d1, d2, atol=1e-5)
    end

    function has_consistent_gradient(f, x)
        g1 = Zygote.gradient(f, x)[1]
        g2 = FiniteDiff.finite_difference_gradient(f, x)
        isapprox(g1, g2, atol=1e-4)
    end


    a = 10.26
    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
    atoms = [Si]
    positions = silicon.positions
    function make_model(a)
        lattice = a / 2 * [[0. 1. 1.];
                           [1. 0. 1.];
                           [1. 1. 0.]]
        terms = [Kinetic(), AtomicLocal()]
        Model(lattice, atoms, positions; terms, temperature=1e-3)
    end
    kgrid = [1, 1, 1]
    Ecut = 7
    make_basis(model::Model) = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
    make_basis(a::Real) = make_basis(make_model(a))
    basis = make_basis(a)

    @testset "r_to_G, G_to_r" begin
        kpt = basis.kpoints[1]
        x = rand(ComplexF64,259)
        y = rand(20,20,20)
        w = rand(ComplexF64,20,20,20)

        # r_to_G w.r.t. lattice
        @test has_consistent_derivative(a -> abs2(sum(r_to_G(make_basis(a), y) .* y)), a)

        # r_to_G kpt w.r.t. lattice
        @test has_consistent_derivative(a -> abs2(sum(r_to_G(make_basis(a), kpt, w) .* x)), a)

        # G_to_r w.r.t. lattice
        @test has_consistent_derivative(a -> abs2(sum(G_to_r(make_basis(a), w) .* w)), a)

        # G_to_r kpt w.r.t. lattice
        @test has_consistent_derivative(a -> abs2(sum(G_to_r(make_basis(a), kpt, x) .* y)), a)

        # r_to_G w.r.t. f_real
        @test has_consistent_gradient(y -> abs2(sum(r_to_G(basis, y) .* w)), y)

        # r_to_G kpt w.r.t. f_real
        @test has_consistent_gradient(w -> abs2(sum(r_to_G(basis, kpt, w) .* x)), w)

        # G_to_r w.r.t. f_fourier
        @test has_consistent_gradient(w -> abs2(sum(G_to_r(basis, w) .* y)), w)

        # G_to_r kpt w.r.t. f_fourier
        @test has_consistent_gradient(x -> abs2(sum(G_to_r(basis, kpt, x) .* y)), x)
    end

    @testset "PlaneWaveBasis w.r.t. lattice" begin
        @test has_consistent_derivative(a -> make_model(a).recip_cell_volume, a)
        @test has_consistent_derivative(a -> make_basis(a).model.recip_cell_volume, a)
        @test has_consistent_derivative(a -> make_basis(a).r_to_G_normalization, a)
        @test has_consistent_derivative(a -> make_basis(a).G_to_r_normalization, a)
        @test has_consistent_derivative(a -> make_basis(a).dvol, a)
    end

    @testset "term precomputations w.r.t. lattice" begin
        # Kinetic
        @test has_consistent_derivative(a -> sum(make_basis(a).terms[1].kinetic_energies[1]), a)

        # AtomicLocal
        @test has_consistent_derivative(a -> make_basis(a).terms[2].potential[1], a)
    end

    @testset "compute_density w.r.t. lattice" begin
        scfres = self_consistent_field(basis, is_converged=DFTK.ScfConvergenceDensity(1e-4))
        ψ = scfres.ψ
        occupation = scfres.occupation

        @test compute_density(basis, ψ, occupation) == DFTK._autodiff_compute_density(basis, ψ, occupation)
        @test has_consistent_derivative(a -> sum(compute_density(make_basis(a), ψ, occupation)), a)
    end
end
