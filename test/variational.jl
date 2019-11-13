using DFTK
using Test
using LinearAlgebra: norm

include("testcases.jl")

function get_scf_energies(testcase, supersampling, functionals)
    Ecut=3
    grid_size=15
    scf_tol=1e-10
    n_bands = 10
    kcoords = [[.2, .3, .4]]

    fft_size = determine_grid_size(testcase.lattice, Ecut, supersampling=supersampling,
                                   ensure_smallprimes=false)
    spec = Species(testcase.atnum, psp=load_psp(testcase.psp))
    if length(functionals) > 0
        model = model_dft(testcase.lattice, functionals, spec => testcase.positions)
    else
        model = model_reduced_hf(testcase.lattice, spec => testcase.positions)
    end

    ksymops = nothing
    basis = PlaneWaveBasis(model, fft_size, Ecut, kcoords, ksymops)
    ham = Hamiltonian(basis, guess_density(basis, spec => testcase.positions))
    scfres = self_consistent_field!(ham, n_bands, tol=scf_tol)
    values(scfres.energies)
end


@testset "Energy is exact for supersampling>2 without XC" begin
    energies = [get_scf_energies(silicon, supersampling, []) for supersampling in (1, 2, 3)]

    @test norm(energies[1] .- energies[2]) > 1e-10
    # supersampling == 2 is exact and going beyond has no effect on energies
    @test norm(energies[2] .- energies[3]) < 1e-10
end

@testset "Energy is not exact for supersampling>2 with XC" begin
    energies = [get_scf_energies(silicon, supersampling, [:lda_x, :lda_c_vwn])
                for supersampling in (1, 2, 3)]

    @test norm(energies[1] .- energies[2]) > 1e-10
    # supersampling == 2 is not exact for XC
    @test norm(energies[2] .- energies[3]) > 1e-10
end
