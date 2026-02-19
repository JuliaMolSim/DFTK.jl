@testmodule HamConsistency begin
using Test
using DFTK
using Logging
using DFTK: mpi_sum
using LinearAlgebra
using ..TestCases: silicon
testcase = silicon

function test_consistency_term(term; rtol=1e-4, atol=1e-8, test_for_constant=false,
                               ε=1e-6, kgrid=[1, 2, 3], kshift=[0, 1, 0]/2, Ecut=10,
                               lattice=testcase.lattice, atom=nothing, spin_polarization=:none)
    sspol = spin_polarization != :none ? " ($spin_polarization)" : ""
    xc    = term isa Xc ? "($(first(term.functionals)))" : ""
    @testset "$(typeof(term))$xc $sspol" begin
        n_dim = 3 - count(iszero, eachcol(lattice))
        if isnothing(atom)
            atom = n_dim == 3 ? ElementPsp(14, load_psp(testcase.psp_gth)) : ElementCoulomb(:Si)
        end
        atoms = [atom, atom]
        model = Model(lattice, atoms, testcase.positions; terms=[term], spin_polarization,
                      symmetries=true)
        basis = PlaneWaveBasis(model; Ecut, kgrid=MonkhorstPack(kgrid; kshift))
        @assert length(basis.terms) == 1

        n_electrons = testcase.n_electrons
        n_bands = div(n_electrons, 2, RoundUp)
        filled_occ = DFTK.filled_occupation(model)

        ψ = [Matrix(qr(randn(ComplexF64, length(G_vectors(basis, kpt)), n_bands)).Q)
             for kpt in basis.kpoints]
        occupation = [filled_occ * rand(n_bands) for _ = 1:length(basis.kpoints)]
        occ_scaling = n_electrons / sum(sum(occupation))
        occupation = [occ * occ_scaling for occ in occupation]
        ρ = with_logger(NullLogger()) do
            compute_density(basis, ψ, occupation)
        end
        τ = compute_kinetic_energy_density(basis, ψ, occupation)
        hubbard_n = nothing
        if term isa Hubbard
            hubbard_n = DFTK.compute_hubbard_n(only(basis.terms), basis, ψ, occupation)
        end
        E0, ham = energy_hamiltonian(basis, ψ, occupation; ρ, τ, hubbard_n)

        # Test operator agrees with matrix form
        for ik = 1:length(basis.kpoints)
            for operator in ham.blocks[ik].operators
                @test norm(Matrix(operator) * ψ[ik] - operator * ψ[ik]) < atol
            end
        end

        # Test operator is derivative of the energy
        δψ = [randn(ComplexF64, size(ψ[ik])) for ik = 1:length(basis.kpoints)]
        function compute_E(ε)
            ψ_trial = ψ .+ ε .* δψ
            ρ_trial = with_logger(NullLogger()) do
                compute_density(basis, ψ_trial, occupation)
            end
            τ_trial = compute_kinetic_energy_density(basis, ψ_trial, occupation)
            hubbard_n_trial = nothing
            if term isa Hubbard
                thub = only(basis.terms)
                hubbard_n_trial = DFTK.compute_hubbard_n(thub, basis, ψ_trial, occupation)
            end
            (; energies) = energy_hamiltonian(basis, ψ_trial, occupation;
                                              ρ=ρ_trial, τ=τ_trial,
                                              hubbard_n=hubbard_n_trial)
            energies.total
        end
        diff = (compute_E(ε) - compute_E(-ε)) / (2ε)

        diff_predicted = 0.0
        for ik = 1:length(basis.kpoints)
            Hψk = ham.blocks[ik] * ψ[ik]
            δψkHψk = sum(occupation[ik][iband] * real(dot(δψ[ik][:, iband], Hψk[:, iband]))
                         for iband = 1:n_bands)
            diff_predicted += 2 * basis.kweights[ik] * δψkHψk
        end
        diff_predicted = mpi_sum(diff_predicted, basis.comm_kpts)

        if test_for_constant
            @test abs(diff) < atol
            @test abs(diff_predicted) < atol
        else
            # Make sure that we don't accidentally test 0 == 0
            @test abs(diff) > atol

            err = abs(diff - diff_predicted)
            @test err < rtol * abs(E0.total) || err < atol
        end
    end
end
end


@testitem "Hamiltonian consistency" setup=[TestCases, HamConsistency] tags=[:dont_test_mpi] begin
    using DFTK
    using LinearAlgebra
    using .HamConsistency: test_consistency_term, testcase

    test_consistency_term(Ewald(); test_for_constant=true)
    test_consistency_term(PspCorrection(); test_for_constant=true)

    test_consistency_term(Kinetic())
    test_consistency_term(ExternalFromReal(X -> cos(X[1])))
    test_consistency_term(ExternalFromFourier(X -> abs(norm(X))))
    test_consistency_term(LocalNonlinearity(ρ -> ρ^2))
    test_consistency_term(Hartree())

    let
        Si = ElementPsp(14, load_psp(testcase.psp_upf))
        test_consistency_term(Hubbard(OrbitalManifold([1, 2], "3P") => 0.01), atom=Si)
        test_consistency_term(Hubbard(OrbitalManifold([1, 2], "3P") => 0.01), atom=Si,
                              spin_polarization=:collinear)
        test_consistency_term(Hubbard(OrbitalManifold([1, 2], "3S") => 0.01,
                                      OrbitalManifold([1, 2], "3P") => 0.02), atom=Si)
        test_consistency_term(Hubbard([OrbitalManifold(Si, "3S"), OrbitalManifold(Si, "3P")],
                                      [0.01, 0.02]), atom=Si)
    end

    for psp in [testcase.psp_gth, testcase.psp_upf]
        Si = ElementPsp(14, load_psp(psp))
        test_consistency_term(AtomicLocal(), atom=Si)
        test_consistency_term(AtomicNonlocal(), atom=Si)
        test_consistency_term(Xc([:lda_xc_teter93]), atom=Si)
        test_consistency_term(Xc([:lda_xc_teter93]), atom=Si, spin_polarization=:collinear)
        test_consistency_term(Xc([:gga_x_pbe]), atom=Si, spin_polarization=:collinear)

        # TODO: for use_nlcc=true need to fix consistency for meta-GGA with NLCC
        #       (see JuliaMolSim/DFTK.jl#1180)
        test_consistency_term(Xc([:mgga_x_tpss]; use_nlcc=false), atom=Si)
        test_consistency_term(Xc([:mgga_x_scan]; use_nlcc=false), atom=Si)
        test_consistency_term(Xc([:mgga_c_scan]; use_nlcc=false), atom=Si,
                              spin_polarization=:collinear)
        test_consistency_term(Xc([:mgga_x_b00]; use_nlcc=false), atom=Si)
        test_consistency_term(Xc([:mgga_c_b94]; use_nlcc=false), atom=Si,
                              spin_polarization=:collinear)
    end

    for exx_algorithm in (VanillaExx(), AceExx())
        test_consistency_term(ExactExchange(; coulomb_kernel_model=ProbeCharge(), exx_algorithm);
                              kgrid=(1, 1, 1), kshift=(0, 0, 0))
    end
    test_consistency_term(ExactExchange(); spin_polarization=:collinear,
                          kgrid=(1, 1, 1), kshift=(0, 0, 0))

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
