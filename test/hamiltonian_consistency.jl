@testmodule HamConsistency begin
using Test
using DFTK
using ForwardDiff
using DftFunctionals: needs_τ
using Logging
using DFTK: mpi_sum
using LinearAlgebra
using ..TestCases: silicon

function test_consistency_term(term; rtol=1e-4, atol=1e-8, ε=1e-6,
                               test_for_constant=false, test_energy_ad=true,
                               n_empty=3, kgrid=[1, 2, 3], kshift=[0, 1, 0]/2, Ecut=10,
                               lattice=silicon.lattice, atom=nothing, spin_polarization=:none,
                               exxalg=AceExx())
    sspol = spin_polarization != :none ? " ($spin_polarization)" : ""
    xc    = term isa Xc ? "($(first(term.functionals)))" : ""
    @testset "$(typeof(term))$xc $sspol" begin
        n_dim = 3 - count(iszero, eachcol(lattice))
        if isnothing(atom)
            atom = n_dim == 3 ? ElementPsp(14, load_psp(silicon.psp_gth)) : ElementCoulomb(:Si)
        end
        atoms = [atom, atom]
        model = Model(lattice, atoms, silicon.positions; terms=[term], spin_polarization,
                      symmetries=true)
        basis = PlaneWaveBasis(model; Ecut, kgrid=MonkhorstPack(kgrid; kshift))
        @assert length(basis.terms) == 1

        n_electrons = silicon.n_electrons
        n_bands = div(n_electrons, 2, RoundUp)
        filled_occ = DFTK.filled_occupation(model)

        ψ = [Matrix(qr(randn(ComplexF64, length(G_vectors(basis, kpt)), n_bands + n_empty)).Q)
             for kpt in basis.kpoints]
        occupation  = [filled_occ * append!(rand(n_bands), zeros(n_empty))
                       for _ = 1:length(basis.kpoints)]
        occ_scaling = length(basis.kpoints) * n_electrons / sum(sum(occupation))
        occupation  = [occ * occ_scaling for occ in occupation]
        ρ = with_logger(NullLogger()) do
            compute_density(basis, ψ, occupation)
        end
        τ = compute_kinetic_energy_density(basis, ψ, occupation)
        hubbard_n = nothing
        if term isa Hubbard
            hubbard_n = DFTK.compute_hubbard_n(only(basis.terms), basis, ψ, occupation)
        end
        E0, ham = energy_hamiltonian(basis, ψ, occupation; exxalg, ρ, τ, hubbard_n)

        # Function to compute energy only
        E0_ene = DFTK.energy(basis, ψ, occupation; exxalg, ρ, τ, hubbard_n).energies
        @test abs(E0.total - E0_ene.total) < 1e-14

        # Test operator agrees with matrix form
        for ik = 1:length(basis.kpoints)
            for operator in ham.blocks[ik].operators
                @test norm(Matrix(operator) * ψ[ik] - operator * ψ[ik]) < atol
            end
        end

        # Test operator is derivative of the energy
        δψ = [randn(ComplexF64, size(ψ[ik])) for ik = 1:length(basis.kpoints)]
        function compute_E(ε::T) where {T}
            modelE = Model(model; lattice=Matrix{T}(model.lattice))
            basisE = PlaneWaveBasis(modelE; Ecut, kgrid=MonkhorstPack(kgrid; kshift))

            ψ_trial = ψ .+ ε .* δψ
            ρ_trial = with_logger(NullLogger()) do
                compute_density(basisE, ψ_trial, occupation)
            end
            τ_trial = nothing
            if needs_τ(only(basisE.terms))
                τ_trial = compute_kinetic_energy_density(basisE, ψ_trial, occupation)
            end
            hubbard_n_trial = nothing
            if term isa Hubbard
                thub = only(basisE.terms)
                hubbard_n_trial = DFTK.compute_hubbard_n(thub, basisE, ψ_trial, occupation)
            end
            (; energies) = DFTK.energy(basisE, ψ_trial, occupation;
                                       exxalg, ρ=ρ_trial, τ=τ_trial, hubbard_n=hubbard_n_trial)
            energies.total
        end
        diff = (compute_E(ε) - compute_E(-ε)) / (2ε)

        # Copy diff for simplicity of testing code in case no AD available.
        # TODO: Better use tagged duals like in kernels.jl as that avoids precompilation
        diff_ad = test_energy_ad ? ForwardDiff.derivative(compute_E, 0.0) : diff

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
            @test abs(diff_ad) < atol
        else
            @test abs(diff) > atol  # Make sure that we don't accidentally test 0 == 0

            err = abs(diff - diff_predicted)
            @test err < rtol * abs(E0.total) || err < atol

            err_ad = abs(diff_ad - diff_predicted)
            test_energy_ad && @show err_ad
            @test err_ad < rtol * abs(E0.total) || err_ad < atol
        end
    end
end
end


@testitem "Hamiltonian consistency" setup=[TestCases, HamConsistency] tags=[:dont_test_mpi] begin
    using DFTK
    using LinearAlgebra
    using .HamConsistency: test_consistency_term, silicon

    test_consistency_term(Ewald(); test_for_constant=true)
    test_consistency_term(PspCorrection(); test_for_constant=true)

    test_consistency_term(Kinetic())
    test_consistency_term(ExternalFromReal(X -> cos(X[1])))
    test_consistency_term(ExternalFromFourier(X -> abs(norm(X))))
    test_consistency_term(LocalNonlinearity(ρ -> ρ^2))
    test_consistency_term(Hartree())

    let
        Si = ElementPsp(14, load_psp(silicon.psp_upf))
        hubbards = [
            Hubbard(OrbitalManifold([1, 2], "3P") => 0.01),
            Hubbard(OrbitalManifold([1, 2], "3S") => 0.01,
                    OrbitalManifold([1, 2], "3P") => 0.02),
            Hubbard([OrbitalManifold(Si, "3S"), OrbitalManifold(Si, "3P")], [0.01, 0.02])
        ]
        for hubbard in hubbards
            # TODO: AD of energy( ) function not yet supported for Hubbard
            test_consistency_term(hubbard; atom=Si, test_energy_ad=false)
        end
        test_consistency_term(hubbards[1];
                              atom=Si, spin_polarization=:collinear, test_energy_ad=false)
    end

    for psp in [silicon.psp_gth, silicon.psp_upf]
        tauad = (; test_energy_ad=false)  # TODO: AD involving tau not yet available

        Si = ElementPsp(14, load_psp(psp))
        test_consistency_term(AtomicLocal(); atom=Si)
        test_consistency_term(AtomicNonlocal(); atom=Si)
        test_consistency_term(Xc([:lda_xc_teter93]); atom=Si)
        test_consistency_term(Xc([:lda_xc_teter93]); atom=Si, spin_polarization=:collinear)
        test_consistency_term(Xc([:gga_x_pbe]); atom=Si, spin_polarization=:collinear)

        # TODO: for use_nlcc=true need to fix consistency for meta-GGA with NLCC
        #       (see JuliaMolSim/DFTK.jl#1180)
        test_consistency_term(Xc([:mgga_x_tpss]; use_nlcc=false); atom=Si, tauad...)
        test_consistency_term(Xc([:mgga_x_scan]; use_nlcc=false); atom=Si, tauad...)
        test_consistency_term(Xc([:mgga_c_scan]; use_nlcc=false); atom=Si, tauad...,
                              spin_polarization=:collinear)
        test_consistency_term(Xc([:mgga_x_b00]; use_nlcc=false); atom=Si, tauad...)
        test_consistency_term(Xc([:mgga_c_b94]; use_nlcc=false); atom=Si, tauad...,
                              spin_polarization=:collinear)
        test_consistency_term(Xc([:mgga_x_scanl]); atom=Si)
    end

    @testset "Exact exchange" begin
        # TODO: AD of energy( ) function not yet supported for ExactExchange
        exxad = (; test_energy_ad=false)
        for exxalg in (VanillaExx(), AceExx())
            test_consistency_term(ExactExchange(; kernel=Coulomb(ProbeCharge()));
                                  exxad..., exxalg, kgrid=(1, 1, 1), kshift=(0, 0, 0))
        end
        test_consistency_term(ExactExchange(); spin_polarization=:collinear,
                              exxad..., kgrid=(1, 1, 1), kshift=(0, 0, 0))
    end

    let
        a = 6
        pot(x, y, z) = (x - a/2)^2 + (y - a/2)^2
        Apot(x, y, z) = .2 * [y - a/2, -(x - a/2), 0]
        Apot(X) = Apot(X...)
        test_consistency_term(Magnetic(Apot); test_energy_ad=false,
                              lattice=[a 0 0; 0 a 0; 0 0 0], 
                              Ecut=20, kgrid=[1, 1, 1], kshift=[0, 0, 0])
        test_consistency_term(DFTK.Anyonic(2, 3.2); test_energy_ad=false, 
                              lattice=[a 0 0; 0 a 0; 0 0 0], 
                              Ecut=20, kgrid=[1, 1, 1], kshift=[0, 0, 0])
    end
end
