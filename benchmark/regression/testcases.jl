@testsetup module Regression
using DFTK
using Unitful
using UnitfulAtomic
using AtomsBase
using ..TestCases: magnesium

high_symmetry = let
    a = 4.474
    lattice = [[0, a, a], [a, 0, a], [a, a, 0]]u"bohr"
    x = 6.711
    y = 2.237
    atoms = [
        Atom(:Cu, [0, 0, 0]u"bohr", magnetic_moment=0),
        Atom(:O,  [x, y, x]u"bohr", magnetic_moment=0),
        Atom(:O,  [x, y, y]u"bohr", magnetic_moment=0),
    ]
    system = periodic_system(atoms, lattice)
    merge(DFTK.parse_system(system), (; temperature=0.03, Ecut=20, kgrid=[4,4,4],
          n_electrons=45, description="high_sym"))
end
high_kpoints = merge(magnesium, (; kgrid=[13,13,13], Ecut=20, description="high_kpoint"))
high_Ecut = merge(magnesium, (; kgrid=[4,4,4], Ecut=60, description="high_Ecut"))

testcases = (; high_symmetry, high_kpoints, high_Ecut)
end


@testitem "Hamiltonian application" tags=[:regression] setup=[TestCases, Regression] begin
    using DFTK
    using LinearAlgebra
    using BenchmarkTools
    using .Main: SUITE

    for testcase in Regression.testcases
        model = Model(testcase.lattice, testcase.atoms, testcase.positions;
                      testcase.temperature, terms=[Kinetic()])
        basis = PlaneWaveBasis(model; testcase.Ecut, testcase.kgrid)

        n_electrons = testcase.n_electrons
        n_bands = div(n_electrons, 2, RoundUp)
        ψ = [Matrix(qr(randn(ComplexF64, length(G_vectors(basis, kpt)), n_bands)).Q)
             for kpt in basis.kpoints]
        filled_occ = DFTK.filled_occupation(model)
        occupation = [filled_occ * rand(n_bands) for _ = 1:length(basis.kpoints)]
        occ_scaling = n_electrons / sum(sum(occupation))
        occupation = [occ * occ_scaling for occ in occupation]

        (; ham) = energy_hamiltonian(basis, ψ, occupation)

        SUITE["ham"][testcase.description] =
            @benchmarkable for ik = 1:length($(basis.kpoints))
                $(ham.blocks)[ik]*$ψ[ik]
            end
    end
end

@testitem "Single SCF step" tags=[:regression] setup=[TestCases, Regression] begin
    using DFTK
    using BenchmarkTools
    using .Main: SUITE

    for testcase in Regression.testcases
        model = model_LDA(testcase.lattice, testcase.atoms, testcase.positions;
                          testcase.temperature)
        basis = PlaneWaveBasis(model; testcase.Ecut, testcase.kgrid)
        SUITE["scf"][testcase.description] =
            @benchmarkable self_consistent_field($basis; tol=1e5)
    end
end

@testitem "Density + symmetrization" tags=[:regression] setup=[TestCases, Regression] begin
    using DFTK
    using BenchmarkTools
    using .Main: SUITE

    for testcase in Regression.testcases
        model = model_LDA(testcase.lattice, testcase.atoms, testcase.positions;
                          testcase.temperature)
        basis = PlaneWaveBasis(model; testcase.Ecut, testcase.kgrid)
        scfres = self_consistent_field(basis; tol=10)

        ψ, occupation = DFTK.select_occupied_orbitals(basis, scfres.ψ, scfres.occupation;
                                                      threshold=1e-6)

        SUITE["density"]["ρ"][testcase.description] =
            @benchmarkable compute_density($basis, $ψ, $occupation)
        SUITE["density"]["sym"][testcase.description] =
            @benchmarkable DFTK.symmetrize_ρ($basis, $(scfres.ρ))
    end
end

@testitem "Basis construction" tags=[:regression] setup=[TestCases, Regression] begin
    using DFTK
    using BenchmarkTools
    using .Main: SUITE

    for testcase in Regression.testcases
        model = model_LDA(testcase.lattice, testcase.atoms, testcase.positions;
                          testcase.temperature)
        SUITE["basis"][testcase.description] =
            @benchmarkable PlaneWaveBasis($model;
                                          Ecut=$(testcase.Ecut), kgrid=$(testcase.kgrid))
    end
end

@testitem "Sternheimer" tags=[:regression] setup=[TestCases, Regression] begin
    using DFTK
    using BenchmarkTools
    using .Main: SUITE

    for testcase in Regression.testcases
        model = model_LDA(testcase.lattice, testcase.atoms, testcase.positions;
                          testcase.temperature)
        basis = PlaneWaveBasis(model; testcase.Ecut, testcase.kgrid)
        scfres = self_consistent_field(basis; tol=10)

        rhs = DFTK.compute_projected_gradient(basis, scfres.ψ, scfres.occupation)
        SUITE["response"]["sternheimer"][testcase.description] =
            @benchmarkable DFTK.solve_ΩplusK_split($scfres, $rhs; tol=1e-1)
    end
end

@testitem "Response with AD" tags=[:regression] setup=[TestCases, Regression] begin
    using DFTK
    using BenchmarkTools
    using LinearAlgebra
    using ForwardDiff
    using .Main: SUITE

    function make_basis(ε::T; a=10., Ecut=30) where {T}
        lattice=T(a) * I(3)  # lattice is a cube of ``a`` Bohrs
        # Helium at the center of the box
        atoms     = [ElementPsp(:He; psp=load_psp("hgh/lda/He-q2"))]
        positions = [[1/2, 1/2, 1/2]]

        model = model_DFT(lattice, atoms, positions, [:lda_x, :lda_c_vwn];
                          extra_terms=[ExternalFromReal(r -> -ε * (r[1] - a/2))],
                          symmetries=false)
        PlaneWaveBasis(model; Ecut, kgrid=[1, 1, 1])  # No k-point sampling on isolated system
    end

    # dipole moment of a given density (assuming the current geometry)
    function dipole(basis, ρ)
        @assert isdiag(basis.model.lattice)
        a  = basis.model.lattice[1, 1]
        rr = [a * (r[1] - 1/2) for r in r_vectors(basis)]
        sum(rr .* ρ) * basis.dvol
    end

    # Function to compute the dipole for a given field strength
    function compute_dipole(ε; tol=1e-2, kwargs...)
        scfres = self_consistent_field(make_basis(ε; kwargs...); tol)
        dipole(scfres.basis, scfres.ρ)
    end

    SUITE["response"]["ad"] = @benchmarkable ForwardDiff.derivative($compute_dipole, 0.0)
end
