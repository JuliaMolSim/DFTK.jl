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
                               Ecut=10, spin_polarization=:none, architecture=DFTK.CPU())
    sspol = spin_polarization != :none ? " ($spin_polarization)" : ""
    xc    = term isa Xc ? "($(first(term.functionals)))" : ""
    @testset "$(typeof(term))$xc $sspol" begin
        n_dim = 3 - count(iszero, eachcol(lattice))
        Si = n_dim == 3 ? ElementPsp(14, psp=load_psp(silicon.psp_hgh)) : ElementCoulomb(:Si)
        atoms = [Si, Si]
        model = Model(lattice, atoms, silicon.positions;
                      terms=[term], spin_polarization, symmetries=true)
        basis = PlaneWaveBasis(model; Ecut, kgrid, kshift, architecture)

        to_arch(x) = DFTK.to_device(basis.architecture, x)

        n_electrons = silicon.n_electrons
        n_bands = div(n_electrons, 2, RoundUp)
        filled_occ = DFTK.filled_occupation(model)

        ψ = [
            to_arch(Matrix(qr(randn(ComplexF64, length(G_vectors(basis, kpt)), n_bands)).Q))
             for kpt in basis.kpoints]
        occupation = [ to_arch(filled_occ * rand(n_bands)) for _ in 1:length(basis.kpoints)]
        occ_scaling = n_electrons / sum(sum(occupation))
        occupation = [occ * occ_scaling for occ in occupation]
        ρ = with_logger(NullLogger()) do
            compute_density(basis, ψ, occupation)
        end
        E0, ham = energy_hamiltonian(basis, ψ, occupation; ρ)

        @assert length(basis.terms) == 1

        δψ = [to_arch(randn(ComplexF64, size(ψ[ik]))) for ik = 1:length(basis.kpoints)]
        function compute_E(ε)
            ψ_trial = ψ .+ ε .* δψ
            ρ_trial = with_logger(NullLogger()) do
                compute_density(basis, ψ_trial, occupation)
            end
            E = energy_hamiltonian(basis, ψ_trial, occupation; ρ=ρ_trial).energies
            E.total
        end

        diff = (compute_E(ε) - compute_E(-ε)) / (2ε)

        diff_predicted = 0.0
        for ik in 1:length(basis.kpoints)
            Hψk = ham.blocks[ik]*ψ[ik]
            # NOTE: Operators as matrices do not yet work for GPU
            isa(architecture, DFTK.CPU) && test_matrix_repr_operator(ham.blocks[ik], ψ[ik]; atol)
            #δψkHψk = sum(occupation[ik][iband] * real(dot(δψ[ik][:, iband], Hψk[:, iband]))
            #           for iband=1:n_bands)
            # Vectorised for GPU compatibility 
            δψkHψk = sum(Array(occupation[ik]) .* real.(dot.(eachcol(δψ[ik]), eachcol(Hψk))))
            diff_predicted += 2 * basis.kweights[ik] * δψkHψk
        end
        diff_predicted = mpi_sum(diff_predicted, basis.comm_kpts)

        err = abs(diff - diff_predicted)
        @test err < rtol * abs(E0.total) || err < atol
    end
end