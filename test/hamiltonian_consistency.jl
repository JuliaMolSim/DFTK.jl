using Test
using DFTK
using Logging
import DFTK: mpi_sum

include("testcases.jl")
using Random
Random.seed!(0)

function test_matrix_repr_operator(hamk, ψk; atol=1e-8)
    for operator in hamk.operators
        try
            operator_matrix = Matrix(operator)
            @test norm(operator_matrix*ψk - operator*ψk) < atol
        catch e
            allowed_missing_operators = Union{DFTK.DivAgradOperator,
                                              DFTK.MagneticFieldOperator}
            @assert operator isa allowed_missing_operators
            @info "Matrix of operator $(nameof(typeof(operator))) not yet supported" maxlog=1
        end
    end
end

function test_consistency_term(term; rtol=1e-4, atol=1e-8, ε=1e-6, kgrid=[1, 2, 3],
                               kshift=[0, 1, 0]/2, lattice=silicon.lattice,
                               Ecut=10, spin_polarization=:none)
    sspol = spin_polarization != :none ? " ($spin_polarization)" : ""
    xc    = term isa Xc ? "($(first(term.functionals)))" : ""
    @testset "$(typeof(term))$xc $sspol" begin
        n_dim = 3 - count(iszero, eachcol(lattice))
        Si = n_dim == 3 ? ElementPsp(14, psp=load_psp(silicon.psp)) : ElementCoulomb(:Si)
        atoms = [Si, Si]
        model = Model(lattice, atoms, silicon.positions;
                      terms=[term], spin_polarization, symmetries=true)
        basis = PlaneWaveBasis(model; Ecut, kgrid, kshift)

        n_electrons = silicon.n_electrons
        n_bands = div(n_electrons, 2, RoundUp)
        filled_occ = DFTK.filled_occupation(model)

        ψ = [Matrix(qr(randn(ComplexF64, length(G_vectors(basis, kpt)), n_bands)).Q)
             for kpt in basis.kpoints]
        occupation = [filled_occ * rand(n_bands) for _ in 1:length(basis.kpoints)]
        occ_scaling = n_electrons / sum(sum(occupation))
        occupation = [occ * occ_scaling for occ in occupation]
        ρ = with_logger(NullLogger()) do
            compute_density(basis, ψ, occupation)
        end
        E0, ham = energy_hamiltonian(basis, ψ, occupation; ρ=ρ)

        @assert length(basis.terms) == 1

        δψ = [randn(ComplexF64, size(ψ[ik])) for ik = 1:length(basis.kpoints)]
        function compute_E(ε)
            ψ_trial = ψ .+ ε .* δψ
            ρ_trial = with_logger(NullLogger()) do
                compute_density(basis, ψ_trial, occupation)
            end
            E, _ = energy_hamiltonian(basis, ψ_trial, occupation; ρ=ρ_trial)
            E.total
        end

        diff = (compute_E(ε) - compute_E(-ε)) / (2ε)

        diff_predicted = 0.0
        for ik in 1:length(basis.kpoints)
            Hψk = ham.blocks[ik]*ψ[ik]
            test_matrix_repr_operator(ham.blocks[ik], ψ[ik]; atol=atol)
            δψkHψk = sum(occupation[ik][iband] * real(dot(δψ[ik][:, iband], Hψk[:, iband]))
                       for iband=1:n_bands)
            diff_predicted += 2 * basis.kweights[ik] * δψkHψk
        end
        diff_predicted = mpi_sum(diff_predicted, basis.comm_kpts)

        err = abs(diff - diff_predicted)
        @test err < rtol * abs(E0.total) || err < atol
    end
end

@testset "Hamiltonian consistency" begin
    test_consistency_term(Kinetic())
    test_consistency_term(AtomicLocal())
    test_consistency_term(AtomicNonlocal())
    test_consistency_term(ExternalFromReal(X -> cos(X[1])))
    test_consistency_term(ExternalFromFourier(X -> abs(norm(X))))
    test_consistency_term(LocalNonlinearity(ρ -> ρ^2))
    test_consistency_term(Hartree())
    test_consistency_term(Ewald())
    test_consistency_term(PspCorrection())
    test_consistency_term(Xc(:lda_xc_teter93))
    test_consistency_term(Xc(:lda_xc_teter93), spin_polarization=:collinear)
    test_consistency_term(Xc(:gga_x_pbe), spin_polarization=:collinear)
    test_consistency_term(Xc(:mgga_x_tpss))
    test_consistency_term(Xc(:mgga_x_scan))
    test_consistency_term(Xc(:mgga_c_scan), spin_polarization=:collinear)
    test_consistency_term(Xc(:mgga_x_b00))
    test_consistency_term(Xc(:mgga_c_b94), spin_polarization=:collinear)

    let
        a = 6
        pot(x, y, z) = (x - a/2)^2 + (y - a/2)^2
        Apot(x, y, z) = .2 * [y - a/2, -(x - a/2), 0]
        Apot(X) = Apot(X...)
        test_consistency_term(Magnetic(Apot); kgrid=[1, 1, 1], kshift=[0, 0, 0],
                              lattice=[a 0 0; 0 a 0; 0 0 0], Ecut=20)
        test_consistency_term(DFTK.Anyonic(2, 3.2); kgrid=[1, 1, 1], kshift=[0, 0, 0],
                              lattice=[a 0 0; 0 a 0; 0 0 0], Ecut=20)
    end
end
