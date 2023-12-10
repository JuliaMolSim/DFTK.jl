@testitem "Save_bands" setup=[TestCases, DictAgreement] begin
using Test
using DFTK
using MPI
using JSON3
testcase = TestCases.silicon

function test_save_bands(label; spin_polarization=:none, Ecut=7, temperature=0.0)
    n_bands = 8
    εF = 0.28

    if spin_polarization == :collinear
        magnetic_moments = [1.0, 1.0]
    else
        magnetic_moments = []
    end
    model = model_LDA(testcase.lattice, testcase.atoms, testcase.positions;
                      spin_polarization, temperature, magnetic_moments)
    basis = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1))
    ρ = guess_density(basis, magnetic_moments)
    band_data = compute_bands(basis; εF, magnetic_moments, ρ, n_bands, kline_density=3)

    @testset "JSON ($label)" begin
        # Tests that data required downstream (e.g. in Aiida) is present in the dict
        # and behaves as expected.

        mktempdir() do tmpdir
            dumpfile = joinpath(tmpdir, "bands.json")
            dumpfile = MPI.bcast(dumpfile, 0, MPI.COMM_WORLD)  # master -> everyone
            save_bands(dumpfile, band_data)

            if mpi_master()
                data = open(JSON3.read, dumpfile)  # Get data back as dict
            else
                data = nothing
            end  # master

            DictAgreement.test_dict_agreement(band_data, data; explicit_reshape=true)
        end  # tmpdir
    end  # json test
end


test_save_bands("nospin notemp";  spin_polarization=:none)
test_save_bands("collinear temp"; spin_polarization=:collinear)
end
