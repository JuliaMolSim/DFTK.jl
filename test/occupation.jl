using Test
using DFTK: smearing_functions, find_fermi_level, Model, PlaneWaveModel
using DFTK: find_occupation_gap_zero_temperature, find_occupation_fermi_metal

include("testcases.jl")


@testset "Smearing for insulators" begin
    Ecut = 5
    n_bands = 10
    fft_size = [15, 15, 15]
    n_electrons = silicon.n_electrons

    # Emulate an insulator ... prepare energy levels
    energies = [zeros(n_bands) for k in silicon.kcoords]
    n_occ = div(n_electrons, 2)
    n_k = length(silicon.kcoords)
    for ik in 1:n_k
        energies[ik] = sort(rand(n_bands))
        energies[ik][n_occ+1:end] .+= 2
    end
    εHOMO = maximum(energies[ik][n_occ] for ik in 1:n_k)
    εLUMO = minimum(energies[ik][n_occ + 1] for ik in 1:n_k)
    Psi = [fill(NaN, 1, n_bands) for i in 1:n_k]

    # Occupation for zero temperature
    model = Model(silicon.lattice, silicon.n_electrons; temperature=0.0, smearing=nothing)
    basis = PlaneWaveModel(model, fft_size, Ecut, silicon.kcoords, silicon.kweights, silicon.ksymops)
    εF0, occupation0 = find_occupation_gap_zero_temperature(basis, energies, Psi)
    @test εHOMO < εF0 < εLUMO
    @test sum(basis.kweights .* sum.(occupation0)) ≈ model.n_electrons

    # See that the electron count still works if we add temperature
    Ts = (0, 1e-6, .1, 1.0)
    for T in Ts, fun in smearing_functions
        model = Model(silicon.lattice, silicon.n_electrons; temperature=T, smearing=fun)
        basis = PlaneWaveModel(model, fft_size, Ecut, silicon.kcoords, silicon.kweights, silicon.ksymops)
        _, occs = find_occupation_fermi_metal(basis, energies, Psi)
        @test sum(basis.kweights .* sum.(occs)) ≈ model.n_electrons
    end

    # See that the occupation is largely uneffected with only a bit of temperature
    Ts = (0, 1e-6, 1e-4)
    for T in Ts, fun in smearing_functions
        model = Model(silicon.lattice, silicon.n_electrons; temperature=T, smearing=fun)
        basis = PlaneWaveModel(model, fft_size, Ecut, silicon.kcoords, silicon.kweights, silicon.ksymops)
        _, occupation = find_occupation_fermi_metal(basis, energies, Psi)

        for ik in 1:n_k
            @test all(isapprox.(occupation[ik], occupation0[ik], atol=1e-2))
        end
    end
end

# TODO εF determination and smearing for metals
