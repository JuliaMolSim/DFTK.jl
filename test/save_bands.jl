using Test
using DFTK

include("testcases.jl")

@testset "save_bands" begin
    model = model_LDA(silicon.lattice, silicon.atoms, silicon.positions)
    basis = PlaneWaveBasis(model; Ecut=7, kgrid=(1, 1, 1))

    ρ = guess_density(basis)
    band_data = compute_bands(basis; ρ, n_bands=8, kline_density=3)

    # TODO Test at least collinear and non-collinear spin

    @testset "JSON" begin
        mktempdir() do tmpdir
            @test false  # TODO
        end
    end

    # TODO Definitely test this with MPI !
    #
    # also test that all keys required by aiida are actually around
end
