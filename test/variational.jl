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
    spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
    model = model_DFT(testcase.lattice, [spec => testcase.positions], functionals)

    ksymops = [[(Mat3{Int}(I), Vec3(zeros(3)))] for _ in 1:length(kcoords)]
    basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops; fft_size=fft_size)
    scfres = self_consistent_field(basis; tol=scf_tol)
    values(scfres.energies.energies)
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
