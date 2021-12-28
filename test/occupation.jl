using Test
using DFTK
using SpecialFunctions

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
    model = Model(silicon.lattice; n_electrons=silicon.n_electrons, temperature=0.0,
                  smearing=nothing, terms=[Kinetic()])
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)
    occupation0, εF0 = DFTK.compute_occupation_bandgap(basis, energies)
    @test εHOMO < εF0 < εLUMO
    @test DFTK.weighted_ksum(basis, sum.(occupation0)) ≈ model.n_electrons

    # See that the electron count still works if we add temperature
    Ts = (0, 1e-6, .1, 1.0)
    for temperature in Ts, meth in smearing_methods
        model = Model(silicon.lattice; n_electrons=silicon.n_electrons, temperature,
                      smearing=meth, terms=[Kinetic()])
        basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)
        occs, _ = DFTK.compute_occupation(basis, energies)
        @test sum(basis.kweights .* sum.(occs)) ≈ model.n_electrons
    end

    # See that the occupation is largely uneffected with only a bit of temperature
    Ts = (0, 1e-6, 1e-4)
    for T in Ts, meth in smearing_methods
        model = Model(silicon.lattice; n_electrons=silicon.n_electrons, temperature=T,
                      smearing=meth, terms=[Kinetic()])
        basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)
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
    kgrid_size = [2, 3, 4]

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

    spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
    symmetries = DFTK.symmetry_operations(testcase.lattice, [spec => testcase.positions])
    kcoords, ksymops = bzmesh_ir_wedge(kgrid_size, symmetries)

    n_bands = length(energies[1])
    n_k = length(kcoords)
    @assert n_k == length(energies)

    parameters = (
        (DFTK.Smearing.FermiDirac(),        0.01, 0.17251898225370),
        (DFTK.Smearing.FermiDirac(),        0.02, 0.17020763046058),
        (DFTK.Smearing.FermiDirac(),        0.03, 0.16865552281082),
        (DFTK.Smearing.MethfesselPaxton(1), 0.01, 0.16917895217084),
        (DFTK.Smearing.MethfesselPaxton(1), 0.02, 0.17350869020891),
        (DFTK.Smearing.MethfesselPaxton(1), 0.03, 0.17395190342809),
    )

    for (meth, temperature, εF_ref) in parameters
        model = Model(silicon.lattice, n_electrons=testcase.n_electrons;
                      temperature=temperature, smearing=meth, terms=[Kinetic()])
        basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops; fft_size=fft_size)
        occupation, εF = DFTK.compute_occupation(basis, energies)

        @test DFTK.weighted_ksum(basis, sum.(occupation)) ≈ model.n_electrons
        @test εF ≈ εF_ref
    end
end
end
