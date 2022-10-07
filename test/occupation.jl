using Test
using DFTK
using SpecialFunctions
using Logging

include("testcases.jl")

smearing_methods = (
        DFTK.Smearing.None(),
        DFTK.Smearing.FermiDirac(),
        DFTK.Smearing.Gaussian(),
        DFTK.Smearing.MarzariVanderbilt(),
        (DFTK.Smearing.MethfesselPaxton(i) for i in 1:4)...
    )

@testset "Smearing functions" begin
    for m in smearing_methods
        @test DFTK.Smearing.occupation(m, -Inf) == 1
        @test DFTK.Smearing.occupation(m, Inf) == 0
        x = .04
        ε = 1e-8
        @test abs((DFTK.Smearing.occupation(m, x+ε) - DFTK.Smearing.occupation(m, x))/ε -
                  DFTK.Smearing.occupation_derivative(m, x)) < 1e-4

        # entropy functions should satisfy s' = x f'
        sprime = (DFTK.Smearing.entropy(m, x+ε) - DFTK.Smearing.entropy(m, x))/ε
        fprime = (DFTK.Smearing.occupation(m, x+ε) - DFTK.Smearing.occupation(m, x))/ε
        @test abs(sprime - x*fprime) < 1e-4
    end
end

if mpi_nprocs() == 1 # can't be bothered to convert the tests
@testset "Smearing for insulators" begin
    Ecut = 5
    n_bands = 10
    fft_size = [15, 15, 15]

    # Emulate an insulator ... prepare energy levels
    energies = [zeros(n_bands) for k in silicon.kcoords]
    n_occ = div(silicon.n_electrons, 2, RoundUp)
    n_k = length(silicon.kcoords)
    for ik in 1:n_k
        energies[ik] = sort(rand(n_bands))
        energies[ik][n_occ+1:end] .+= 2
    end
    εHOMO = maximum(energies[ik][n_occ] for ik in 1:n_k)
    εLUMO = minimum(energies[ik][n_occ + 1] for ik in 1:n_k)

    # Occupation for zero temperature
    model = Model(silicon.lattice, silicon.atoms, silicon.positions; temperature=0.0,
                  terms=[Kinetic()])
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.kweights; fft_size)
    occupation0, εF0 = DFTK.compute_occupation(basis, energies)
    @test εHOMO < εF0 < εLUMO
    @test DFTK.weighted_ksum(basis, sum.(occupation0)) ≈ model.n_electrons

    # See that the electron count still works if we add temperature
    Ts = (0, 1e-6, .1, 1.0)
    for temperature in Ts, smearing in smearing_methods
        model = Model(silicon.lattice, silicon.atoms, silicon.positions;
                      temperature, smearing, terms=[Kinetic()])
        basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.kweights; fft_size)
        occs, _ = with_logger(NullLogger()) do
            DFTK.compute_occupation(basis, energies)
        end
        @test sum(basis.kweights .* sum.(occs)) ≈ model.n_electrons
    end

    # See that the occupation is largely uneffected with only a bit of temperature
    Ts = (0, 1e-6, 1e-4)
    for temperature in Ts, smearing in smearing_methods
        model = Model(silicon.lattice, silicon.atoms, silicon.positions;
                      temperature, smearing, terms=[Kinetic()])
        basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.kweights; fft_size)
        occupation, _ = DFTK.compute_occupation(basis, energies)

        for ik in 1:n_k
            @test all(isapprox.(occupation[ik], occupation0[ik], atol=1e-2))
        end
    end
end
end

if mpi_nprocs() == 1 # can't be bothered to convert the tests
@testset "Smearing for metals" begin
    testcase = magnesium
    Ecut = 5
    fft_size = [15, 15, 15]
    kgrid  = [2, 3, 4]

    # Emulate a metal ...
    energies = [[-0.08063210585291,  0.11227915155236, 0.13057816014162, 0.57672256037074],
                [ 0.09509047528102,  0.09538152469111, 0.27197836572013, 0.28750689088845],
                [-0.00144586520885,  0.18640677556553, 0.19603060374450, 0.24422060327989],
                [ 0.05693643182609,  0.16919740718547, 0.24190245274401, 0.25674283154835],
                [-0.06756541677784,  0.03381889875058, 0.23162853469956, 0.50981867707851],
                [ 0.10685980948954,  0.10728887405642, 0.20784971952147, 0.20786603845828],
                [ 0.01122399002894,  0.11011069317735, 0.24016826005369, 0.30770620467001],
                [ 0.06925846412968,  0.16087157153058, 0.19146746736359, 0.27463770659603],
                [-0.02937886574534, -0.02937886574483, 0.36206906745747, 0.36206906745749],
                [ 0.13314087354890,  0.13314087354890, 0.15834732772541, 0.15834732772541],
                [ 0.04869672986772,  0.04869672986772, 0.27749728805752, 0.27749728805768],
                [ 0.10585630776222,  0.10585630776223, 0.22191839818805, 0.22191839818822]]

    symmetries = DFTK.symmetry_operations(testcase.lattice, testcase.atoms, testcase.positions)
    kcoords, _ = bzmesh_ir_wedge(kgrid, symmetries)

    n_k = length(kcoords)
    @assert n_k == length(energies)

    parameters = (
        (DFTK.Smearing.FermiDirac(),        0.01, 0.16163115311626172),
        (DFTK.Smearing.FermiDirac(),        0.02, 0.1624111568340279),
        (DFTK.Smearing.FermiDirac(),        0.03, 0.1630075080960013),
        (DFTK.Smearing.MethfesselPaxton(1), 0.01, 0.16120395021955866),
        (DFTK.Smearing.MethfesselPaxton(1), 0.02, 0.16153528960704408),
        (DFTK.Smearing.MethfesselPaxton(1), 0.03, 0.16131173898225953),
    )

    for (smearing, temperature, εF_ref) in parameters
        model = Model(silicon.lattice, testcase.atoms, testcase.positions;
                      n_electrons=testcase.n_electrons,
                      temperature, smearing, terms=[Kinetic()])
        basis = PlaneWaveBasis(model; Ecut, kgrid, fft_size, kshift=[1, 0, 1]/2)
        occupation, εF = with_logger(NullLogger()) do
            DFTK.compute_occupation(basis, energies)
        end

        @test DFTK.weighted_ksum(basis, sum.(occupation)) ≈ model.n_electrons
        @test εF ≈ εF_ref
    end
end
end

if mpi_nprocs() == 1 # can't be bothered to convert the tests
@testset "Fixed Fermi level" begin
    testcase = magnesium
    atoms = fill(ElementGaussian(1.0, 0.5), length(testcase.positions))
    temperature = 0.01

    compute_scfres(εF=nothing) = begin
        comput = isnothing(εF) ? Dict(:n_electrons=>testcase.n_electrons) : Dict(:εF=>εF)
        model = Model(silicon.lattice, atoms, testcase.positions; temperature, comput...,
                      check_electrostatics=false)
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2])
        self_consistent_field(basis; nbandsalg=FixedBands(; n_bands_converge=8))
    end
    scfres_ref = compute_scfres()
    εF_ref = scfres_ref.εF
    n_electrons_ref = scfres_ref.basis.model.n_electrons
    @test n_electrons_ref == testcase.n_electrons

    δεF = εF_ref / 4
    for εF in [εF_ref - δεF, εF_ref + δεF]
        scfres = compute_scfres(εF)
        @test εF ≈ scfres.εF
        n_electrons = DFTK.weighted_ksum(scfres.basis, sum.(scfres.occupation))
        εF > εF_ref && @test n_electrons > n_electrons_ref
        εF < εF_ref && @test n_electrons < n_electrons_ref
    end

    # Violates charge neutrality:
    @test_throws ErrorException model_atomic(silicon.lattice, testcase.atoms,
                                             testcase.positions; temperature, εF=0.1)
end
end
