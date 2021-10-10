using DFTK
using Test
using LinearAlgebra: norm

include("testcases.jl")

function get_scf_energies(testcase, supersampling, functionals)
    Ecut=3
    grid_size=15
    scf_tol=1e-10  # Tolerance in total enengy
    n_bands = 10
    kcoords = [[.2, .3, .4]]

    spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
    model = model_DFT(testcase.lattice, [spec => testcase.positions], functionals)
    fft_size = compute_fft_size(model, Ecut, kcoords;
                                supersampling, ensure_smallprimes=false, algorithm=:precise)

    ksymops = [[DFTK.identity_symop()] for _ in 1:length(kcoords)]
    basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops, [DFTK.identity_symop()]; fft_size)
    scfres = self_consistent_field(basis; tol=scf_tol)
    values(scfres.energies)
end


@testset "Energy is exact for supersampling>2 without XC" begin
    energies = [get_scf_energies(silicon, supersampling, []) for supersampling in (1, 2, 3)]

    @test abs(sum(energies[1]) - sum(energies[2])) > 1e-10

    # supersampling == 2 is exact and going beyond has no effect on the total energy
    @test abs(sum(energies[2]) - sum(energies[3])) < 1e-10

    # Individual terms are not variational and only converged to the amount
    # the density and the orbitals are converged, which is about sqrt(energy)
    @test norm(energies[2] .- energies[3]) < 1e-5
end

@testset "Energy is not exact for supersampling>2 with XC" begin
    energies = [get_scf_energies(silicon, supersampling, [:lda_x, :lda_c_vwn])
                for supersampling in (1, 2, 3)]

    @test abs(sum(energies[1]) - sum(energies[2])) > 1e-10

    # supersampling == 2 is not exact in total energy for XC
    @test abs(sum(energies[2]) - sum(energies[3])) > 1e-10
end
