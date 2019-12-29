using Test
using DFTK: smearing_functions, find_fermi_level, Model, PlaneWaveBasis
using DFTK: find_occupation, find_occupation_bandgap
using DFTK: smearing_fermi_dirac, load_psp, Species, bzmesh_ir_wedge
using DFTK: smearing_methfessel_paxton_1

include("testcases.jl")


@testset "Smearing for insulators" begin
    Ecut = 5
    n_bands = 10
    fft_size = [15, 15, 15]

    # Emulate an insulator ... prepare energy levels
    energies = [zeros(n_bands) for k in silicon.kcoords]
    n_occ = div(silicon.n_electrons, 2)
    n_k = length(silicon.kcoords)
    for ik in 1:n_k
        energies[ik] = sort(rand(n_bands))
        energies[ik][n_occ+1:end] .+= 2
    end
    εHOMO = maximum(energies[ik][n_occ] for ik in 1:n_k)
    εLUMO = minimum(energies[ik][n_occ + 1] for ik in 1:n_k)

    # Occupation for zero temperature
    model = Model(silicon.lattice, silicon.n_electrons; temperature=0.0, smearing=nothing)
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)
    occupation0, εF0 = find_occupation_bandgap(basis, energies)
    @test εHOMO < εF0 < εLUMO
    @test sum(basis.kweights .* sum.(occupation0)) ≈ model.n_electrons

    # See that the electron count still works if we add temperature
    Ts = (0, 1e-6, .1, 1.0)
    for T in Ts, fun in smearing_functions
        model = Model(silicon.lattice, silicon.n_electrons; temperature=T, smearing=fun)
        basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)
        occs, _ = find_occupation(basis, energies)
        @test sum(basis.kweights .* sum.(occs)) ≈ model.n_electrons
    end

    # See that the occupation is largely uneffected with only a bit of temperature
    Ts = (0, 1e-6, 1e-4)
    for T in Ts, fun in smearing_functions
        model = Model(silicon.lattice, silicon.n_electrons; temperature=T, smearing=fun)
        basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)
        occupation, _= find_occupation(basis, energies)

        for ik in 1:n_k
            @test all(isapprox.(occupation[ik], occupation0[ik], atol=1e-2))
        end
    end
end

@testset "Smearing for metals" begin
    testcase = manganese
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

    spec = Species(testcase.atnum, psp=load_psp(testcase.psp))
    kcoords, ksymops = bzmesh_ir_wedge(kgrid_size, testcase.lattice,
                                       spec => testcase.positions)

    n_bands = length(energies[1])
    n_k = length(kcoords)
    @assert n_k == length(energies)

    parameters = (
        (smearing_fermi_dirac,         0.01, 0.17251898225370),
        (smearing_fermi_dirac,         0.02, 0.17020763046058),
        (smearing_fermi_dirac,         0.03, 0.16865552281082),
        (smearing_methfessel_paxton_1, 0.01, 0.16917895217084),
        (smearing_methfessel_paxton_1, 0.02, 0.17350869020891),
        (smearing_methfessel_paxton_1, 0.03, 0.17395190342809),
    )

    for (smearing, Tsmear, εF_ref) in parameters
        model = Model(silicon.lattice, testcase.n_electrons;
                      temperature=Tsmear, smearing=smearing)
        basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops; fft_size=fft_size)
        occupation, εF = find_occupation(basis, energies)

        @test sum(basis.kweights .* sum.(occupation)) ≈ model.n_electrons
        @test εF ≈ εF_ref
    end
end
