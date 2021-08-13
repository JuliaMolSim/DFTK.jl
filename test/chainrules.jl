using Test
using DFTK
using FiniteDiff
using Zygote
include("testcases.jl")

# Testing rrules needed for reverse Hellmann-Feynman stress

@testset "ChainRules" begin
    a = 10.26
    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
    atoms = [Si => silicon.positions]
    function make_model(a)
        lattice = a / 2 * [[0. 1. 1.];
                           [1. 0. 1.];
                           [1. 1. 0.]]
        terms = [Kinetic(), AtomicLocal()]
        Model(lattice; atoms=atoms, terms=terms, temperature=1e-3)
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

        function has_consistent_derivative(f, a)
            d1 = Zygote.gradient(f, a)[1]
            d2 = FiniteDiff.finite_difference_derivative(f, a)
            isapprox(d1, d2, atol=1e-5)
        end

        # r_to_G w.r.t. lattice
        @test has_consistent_derivative(a -> abs2(sum(r_to_G(make_basis(a), y) .* y)), a)

        # r_to_G kpt w.r.t. lattice
        @test has_consistent_derivative(a -> abs2(sum(r_to_G(make_basis(a), kpt, w) .* x)), a)

        # G_to_r w.r.t. lattice
        @test has_consistent_derivative(a -> abs2(sum(G_to_r(make_basis(a), w) .* w)), a)

        # G_to_r kpt w.r.t. lattice
        @test has_consistent_derivative(a -> abs2(sum(G_to_r(make_basis(a), kpt, x) .* y)), a)

        function has_consistent_gradient(f, x)
            g1 = Zygote.gradient(f, x)[1]
            g2 = FiniteDiff.finite_difference_gradient(f, x)
            isapprox(g1, g2, atol=1e-4)
        end

        # r_to_G w.r.t. f_real
        @test has_consistent_gradient(y -> abs2(sum(r_to_G(basis, y) .* w)), y)

        # r_to_G kpt w.r.t. f_real
        @test has_consistent_gradient(w -> abs2(sum(r_to_G(basis, kpt, w) .* x)), w)

        # G_to_r w.r.t. f_fourier
        @test has_consistent_gradient(w -> abs2(sum(G_to_r(basis, w) .* y)), w)

        # G_to_r kpt w.r.t. f_fourier
        @test has_consistent_gradient(x -> abs2(sum(G_to_r(basis, kpt, x) .* y)), x)
    end
    # TODO more coverage of rrules
end
