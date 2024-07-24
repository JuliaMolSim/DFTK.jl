@testmodule Occupation begin
using DFTK

smearing_methods = (
    DFTK.Smearing.None(),
    DFTK.Smearing.FermiDirac(),
    DFTK.Smearing.Gaussian(),
    DFTK.Smearing.MarzariVanderbilt(),
    DFTK.Smearing.MethfesselPaxton.(1:4)...
)
fermialgs = (
    FermiBisection(),
    FermiTwoStage(),
)
end


@testitem "Smearing functions" setup=[Occupation] begin
    using DFTK

    for m in Occupation.smearing_methods
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

@testitem "Smearing for insulators" tags=[:dont_test_mpi] setup=[Occupation, TestCases] begin
    using DFTK: FermiZeroTemperature
    using Logging
    silicon = TestCases.silicon

    Ecut = 5
    n_bands = 10
    fft_size = [15, 15, 15]

    # Emulate an insulator ... prepare energy levels
    n_k = length(silicon.kgrid)
    eigenvalues = [zeros(n_bands) for _ = 1:n_k]
    n_occ = div(silicon.n_electrons, 2, RoundUp)
    for ik = 1:n_k
        eigenvalues[ik] = sort(rand(n_bands))
        eigenvalues[ik][n_occ+1:end] .+= 2
    end
    εHOMO = maximum(eigenvalues[ik][n_occ]     for ik = 1:n_k)
    εLUMO = minimum(eigenvalues[ik][n_occ + 1] for ik = 1:n_k)

    # Occupation for zero temperature
    occupation0 = let
        model = Model(silicon.lattice, silicon.atoms, silicon.positions;
                      temperature=0.0, terms=[Kinetic()])
        basis = PlaneWaveBasis(model; Ecut, silicon.kgrid, fft_size)
        occupation, εF = DFTK.compute_occupation(basis, eigenvalues, FermiZeroTemperature())
        @test εHOMO < εF < εLUMO
        @test DFTK.weighted_ksum(basis, sum.(occupation)) ≈ model.n_electrons
        occupation
    end

    # See that the electron count still works if we add temperature
    for temperature in (0, 1e-6, .1, 1.0), smearing in Occupation.smearing_methods,
                                           alg in Occupation.fermialgs
        model = Model(silicon.lattice, silicon.atoms, silicon.positions;
                      temperature, smearing, terms=[Kinetic()])
        basis = PlaneWaveBasis(model; Ecut, silicon.kgrid, fft_size)
        occs = with_logger(NullLogger()) do
            DFTK.compute_occupation(basis, eigenvalues, alg; tol_n_elec=1e-12).occupation
        end
        @test sum(basis.kweights .* sum.(occs)) ≈ model.n_electrons
    end

    # See that the occupation is largely uneffected with only a bit of temperature
    for temperature in (0, 1e-6, 1e-4), smearing in Occupation.smearing_methods,
                                        alg in Occupation.fermialgs
        model = Model(silicon.lattice, silicon.atoms, silicon.positions;
                      temperature, smearing, terms=[Kinetic()])
        basis = PlaneWaveBasis(model; Ecut, silicon.kgrid, fft_size)
        (; occupation) = DFTK.compute_occupation(basis, eigenvalues, alg; tol_n_elec=1e-6)

        for ik = 1:n_k
            @test all(isapprox.(occupation[ik], occupation0[ik]; atol=1e-2))
        end
    end
end

@testitem "Smearing for a simple metal" #=
    =#    tags=[:dont_test_mpi] setup=[Occupation, TestCases] begin
    using DFTK
    using Logging
    (; silicon, magnesium) = TestCases.all_testcases

    # Note: Mixture of silicon and magnesium is on purpose
    model = Model(silicon.lattice, magnesium.atoms, magnesium.positions;
                  n_electrons=magnesium.n_electrons, temperature=1e-2, terms=[Kinetic()])
    basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 3, 4], kshift=[1, 0, 1]/2)

    # Emulate a metal ...
    eigenvalues = [[-0.08063210585291,  0.11227915155236, 0.13057816014162, 0.57672256037074],
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
    @assert length(basis.kpoints) == length(eigenvalues)

    parameters = (
        (DFTK.Smearing.FermiDirac(),        0.01, 0.16163115311626172),
        (DFTK.Smearing.FermiDirac(),        0.02, 0.1624111568340279),
        (DFTK.Smearing.FermiDirac(),        0.03, 0.1630075080960013),
        (DFTK.Smearing.MethfesselPaxton(1), 0.01, 0.16120395021955866),
        (DFTK.Smearing.MethfesselPaxton(1), 0.02, 0.16153528960704408),
        (DFTK.Smearing.MethfesselPaxton(1), 0.03, 0.16131173898225953),
    )
    for (smearing, temperature, εF_ref) in parameters, alg in Occupation.fermialgs
        occupation, εF = with_logger(NullLogger()) do
            DFTK.compute_occupation(basis, eigenvalues, alg; smearing, temperature,
                                    tol_n_elec=1e-10)
        end
        @test DFTK.weighted_ksum(basis, sum.(occupation)) ≈ model.n_electrons
        @test εF ≈ εF_ref
    end
end

@testitem "Fermi level finding for smearing multiple εF" #=
    =#    tags=[:dont_test_mpi] setup=[Occupation, TestCases] begin
    using DFTK
    using Logging
    iron_bcc = TestCases.iron_bcc

    # This is an iron setup, which caused trouble in the past
    #
    model = Model(iron_bcc.lattice, iron_bcc.atoms, iron_bcc.positions;
                  n_electrons=iron_bcc.n_electrons, temperature=1e-2, terms=[Kinetic()],
                  magnetic_moments=[4])
    basis = PlaneWaveBasis(model; Ecut=5, kgrid=(4, 4, 4))

    eigenvalues = [
         [-0.09317171, 0.05662733, 0.05662733, 0.05662733, 0.1027973, 0.1027973, 1.133822],
         [-0.01970275, 0.04049757, 0.06672252, 0.09504781, 0.09571417, 0.1045477, 0.777691],
         [-0.008317428, 0.02549644, 0.1018706, 0.1075814, 0.140451, 0.2306614, 0.4425744],
         [0.01779316, 0.03559934, 0.06175814, 0.09824229, 0.1133369, 0.2534185, 0.598366],
         [0.03831527, 0.03831527, 0.03831527, 0.1085794, 0.1085794, 0.5062097, 0.5062097],
         [0.02249843, 0.04778311, 0.07322733, 0.07322733, 0.09343694, 0.1691936, 0.8155277],
         [0.01298261, 0.03133082, 0.07131262, 0.0928459, 0.1361777, 0.3808009, 0.4713903],
         [-0.006884875, -0.006884875, 0.1317392, 0.1317392, 0.1317392, 0.560895, 0.560895],
         [-0.0686855, 0.3042072, 0.3042072, 0.3042072, 0.3824597, 0.3824597, 1.141987],
         [0.01739212, 0.2749508, 0.2790246, 0.3584238, 0.3687848, 0.3864797, 0.8680247],
         [0.1453898, 0.2491354, 0.2495129, 0.3776472, 0.3951613, 0.4310432, 0.6072722],
         [0.1478936, 0.2322534, 0.2519767, 0.3713906, 0.3948955, 0.4582024, 0.707126],
         [0.2196432, 0.2196432, 0.2196432, 0.3973532, 0.3973532, 0.632154, 0.632154],
         [0.09041911, 0.2750986, 0.312412, 0.312412, 0.3584034, 0.4077667, 0.9012088],
         [0.1944933, 0.2061272, 0.3099407, 0.3365089, 0.4243378, 0.4929272, 0.599024],
         [0.1830541, 0.1830541, 0.417559, 0.417559, 0.417559, 0.5923824, 0.5923824],
    ]
    @assert length(basis.kpoints) == length(eigenvalues)

    parameters = (  #                                                # other εF with +ve DOS
        (DFTK.Smearing.Gaussian(),          1e-2, 0.26725860386964656),
        (DFTK.Smearing.MarzariVanderbilt(), 1e-2, 0.2624352644962286),
        (DFTK.Smearing.MethfesselPaxton(1), 1e-2, 0.2685411900311375),
        (DFTK.Smearing.MethfesselPaxton(2), 1e-2, 0.2627127425669326),  # 0.2713096607939751
        (DFTK.Smearing.MethfesselPaxton(5), 1e-2, 0.2614680485215832),  # 0.2724701412295064
        (DFTK.Smearing.Gaussian(),          1e-3, 0.27413279592006573),
        (DFTK.Smearing.MarzariVanderbilt(), 1e-3, 0.2744172412944558),
        (DFTK.Smearing.MethfesselPaxton(1), 1e-3, 0.27445994971377974),
        (DFTK.Smearing.MethfesselPaxton(2), 1e-3, 0.2745745124212358),
        (DFTK.Smearing.MethfesselPaxton(5), 1e-3, 0.2747069248135472),
        (DFTK.Smearing.Gaussian(),          1e-4, 0.27488223611466617),
        (DFTK.Smearing.MarzariVanderbilt(), 1e-4, 0.27490853712608),
        (DFTK.Smearing.MethfesselPaxton(1), 1e-4, 0.274908008227603),
        (DFTK.Smearing.MethfesselPaxton(2), 1e-4, 0.27491607692918685),
        (DFTK.Smearing.MethfesselPaxton(5), 1e-4, 0.2749270541035508),
    )
    for (smearing, temperature, εF_ref) in parameters
        fermialg = DFTK.default_fermialg(smearing)  # TODO Test others
        occupation, εF = with_logger(NullLogger()) do
            DFTK.compute_occupation(basis, eigenvalues, fermialg; smearing, temperature)
        end

        @test DFTK.weighted_ksum(basis, sum.(occupation)) ≈ model.n_electrons
        @test εF ≈ εF_ref
    end
end

@testitem "Density for smearing with multiple εF" setup=[Occupation, TestCases] begin
    using DFTK
    testcase = TestCases.iron_bcc

    magnetic_moments = [4.0]
    model = model_PBE(testcase.lattice, testcase.atoms, testcase.positions;
                      temperature=1e-2, smearing=Smearing.Gaussian(), magnetic_moments)
    basis = PlaneWaveBasis(model; Ecut=10, kgrid=[4, 4, 4])
    scfres = self_consistent_field(basis; ρ=guess_density(basis, magnetic_moments), tol=1e-4)

    for temperature in (1e-4, 1e-3, 1e-2), smearing in Occupation.smearing_methods,
                                           alg in Occupation.fermialgs
        smearing isa Smearing.None && continue
        occupation, εF = DFTK.compute_occupation(scfres.basis, scfres.eigenvalues, alg;
                                                 smearing, temperature)
        ρ = DFTK.compute_density(scfres.basis, scfres.ψ, scfres.occupation;
                                 scfres.occupation_threshold)
        atol = scfres.occupation_threshold
        @test DFTK.weighted_ksum(basis, sum.(occupation)) ≈ model.n_electrons atol=atol
        @test sum(ρ) * scfres.basis.dvol ≈ model.n_electrons atol=atol
    end
end

@testitem "Fixed Fermi level" tags=[:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    testcase = TestCases.magnesium

    function run_scf(; kwargs...)
        atoms = fill(ElementGaussian(1.0, 0.5), length(testcase.positions))
        model = Model(testcase.lattice, atoms, testcase.positions;
                      temperature=0.01, disable_electrostatics_check=true, kwargs...)
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=(2, 2, 2))
        self_consistent_field(basis; nbandsalg=FixedBands(; n_bands_converge=8))
    end
    scfres_ref = run_scf(; testcase.n_electrons)
    εF_ref = scfres_ref.εF
    n_electrons_ref = scfres_ref.basis.model.n_electrons
    @test n_electrons_ref == testcase.n_electrons

    δεF = εF_ref / 4
    for εF in [εF_ref - δεF, εF_ref + δεF]
        scfres = run_scf(; εF)
        @test εF ≈ scfres.εF
        n_electrons = DFTK.weighted_ksum(scfres.basis, sum.(scfres.occupation))
        εF > εF_ref && @test n_electrons > n_electrons_ref
        εF < εF_ref && @test n_electrons < n_electrons_ref
    end
end
