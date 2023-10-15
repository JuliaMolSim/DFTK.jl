using Test
using DFTK
using MPI
using JSON3

include("testcases.jl")

function test_save_bands(label; spin_polarization=:none, Ecut=7, temperature=0.0)
    n_bands = 8
    εF = 0.28

    if spin_polarization == :collinear
        magnetic_moments = [1.0, 1.0]
    else
        magnetic_moments = []
    end
    model = model_LDA(silicon.lattice, silicon.atoms, silicon.positions;
                      spin_polarization, temperature, magnetic_moments)
    basis = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1))
    n_spin = model.n_spin_components
    ρ = guess_density(basis, magnetic_moments)

    band_data = compute_bands(basis; εF, magnetic_moments, ρ, n_bands, kline_density=3)
    n_kpoints = length(band_data.basis.kcoords_global)

    @testset "JSON ($label)" begin
        mktempdir() do tmpdir
            dumpfile = joinpath(tmpdir, "bands.json")
            dumpfile = MPI.bcast(dumpfile, 0, MPI.COMM_WORLD)  # master -> everyone
            save_bands(dumpfile, band_data)

            all_eigenvalues = DFTK.gather_kpts(band_data.eigenvalues, band_data.basis)
            all_occupation  = DFTK.gather_kpts(band_data.occupation,  band_data.basis)
            all_n_iter      = DFTK.gather_kpts(band_data.diagonalization.n_iter, band_data.basis)

            if mpi_master()
                data = open(JSON3.read, dumpfile)  # Get data back as dict
                @test data["n_bands"]   == n_bands
                @test data["n_kpoints"] == n_kpoints
                @test data["n_spin"]    == n_spin
                @test data["εF"]        == εF
                @test data["kcoords"]   ≈  band_data.basis.kcoords_global atol=1e-12

                eigenvalues_json = reshape(data["eigenvalues"], (n_spin, n_kpoints, n_bands))
                occupation_json  = reshape(data["occupation"],  (n_spin, n_kpoints, n_bands))
                n_iter_json      = reshape(data["n_iter"],      (n_spin, n_kpoints))

                for σ in 1:n_spin
                    for (i, ik) in enumerate(DFTK.krange_spin(band_data.basis, σ))
                        @test all_eigenvalues[ik] ≈  eigenvalues_json[σ, i, :] atol=1e-12
                        @test all_occupation[ik]  ≈  occupation_json[σ, i, :]  atol=1e-12
                        @test all_n_iter[ik]      == n_iter_json[σ, i]
                    end
                end
            end  # master
        end  # tmpdir
    end  # json test
end


@testset "save_bands" begin
    test_save_bands("nospin notemp";  spin_polarization=:none)
    test_save_bands("collinear temp"; spin_polarization=:collinear)
end
