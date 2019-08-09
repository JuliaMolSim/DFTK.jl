using Test
using DFTK

include("silicon_testcases.jl")
@testset "Smearing" begin
    Ecut = 5
    grid_size=15
    basis = PlaneWaveBasis(lattice, grid_size * ones(3), Ecut, kpoints, kweights, ksymops)
    nband = 10
    n_elec = 8
    energies = [zeros(nband) for k in kpoints]
    for ik in 1:length(kpoints)
        energies[ik] = sort(rand(nband))
        energies[ik][div(n_elec,2)+1:end] .+= 2 # emulate an insulator
    end
    Psi = [fill(NaN,1,nband) for k in kpoints]

    occs_zero_temp = DFTK.occupation_step(basis, energies, Psi, n_elec)
    @test sum(kweights .* sum.(occs_zero_temp)) ≈ n_elec
    Ts = (0, 1e-6, .1, 1.0)
    for T in Ts, fun in DFTK.smearing_functions
        occs = DFTK.occupation_temperature(basis, energies, Psi, n_elec, T, smearing=fun)
        @test sum(kweights .* sum.(occs)) ≈ n_elec
        if T < 1e-4
            for ik in 1:length(kpoints)
                for band = 1:nband
                    @test all(isapprox.(occs[ik], occs_zero_temp[ik], atol=1e-2))
                end
            end
        end
    end
end
