@testsetup module DictAgreement
using Test
using DFTK

function test_dict_agreement(band_data, dict; explicit_reshape=false)
    # NOTE: For MPI-parallel tests, this needs to be called on each processor,
    #       but testing only happens on master

    model = band_data.basis.model
    all_eigenvalues = DFTK.gather_kpts(band_data.eigenvalues, band_data.basis)
    all_occupation  = DFTK.gather_kpts(band_data.occupation,  band_data.basis)
    all_n_iter      = DFTK.gather_kpts(last(band_data.diagonalization).n_iter, band_data.basis)
    rotations       = [symop.W for symop in band_data.basis.symmetries]
    translations    = [symop.w for symop in band_data.basis.symmetries]
    n_kpoints = length(band_data.basis.kcoords_global)
    n_spin    = model.n_spin_components
    n_bands   = length(band_data.eigenvalues[1])

    function condreshape(data, shape...)
        if explicit_reshape
            reshape(data, shape...)
        else
            data
        end
    end

    if mpi_master()
        # Tests that data required downstream (e.g. in Aiida) is present in the dict
        # and behaves as expected.

        @test dict["n_bands"]   == n_bands
        @test dict["n_kpoints"] == n_kpoints
        @test dict["n_spin_components"] == n_spin
        @test dict["model_name"]  == model.model_name
        @test dict["temperature"] ≈  model.temperature  atol=1e-12
        @test dict["smearing"]    == "$(model.smearing)"
        @test dict["atomic_symbols"] == map(e -> string(atomic_symbol(e)), model.atoms)
        @test dict["atomic_positions"] ≈ model.positions atol=1e-12
        @test dict["εF"]        ≈  band_data.εF  atol=1e-12
        @test dict["kcoords"]   ≈  band_data.basis.kcoords_global  atol=1e-12
        @test dict["kweights"]  ≈  band_data.basis.kweights_global atol=1e-12
        @test dict["Ecut"]      ≈  band_data.basis.Ecut
        @test [dict["fft_size"]...]  == [band_data.basis.fft_size...]
        @test dict["symmetries_translations"] ≈ translations atol=1e-12

        lattice_resh = condreshape(dict["lattice"], 3, 3)
        rotations_resh = [condreshape(rot, 3, 3) for rot in dict["symmetries_rotations"]]
        @test lattice_resh ≈ model.lattice atol=1e-12
        @test rotations_resh ≈ rotations atol=1e-12

        eigenvalues_resh = condreshape(dict["eigenvalues"], (n_spin, n_kpoints, n_bands))
        occupation_resh  = condreshape(dict["occupation"],  (n_spin, n_kpoints, n_bands))
        n_iter_resh      = condreshape(dict["n_iter"],      (n_spin, n_kpoints))
        for σ in 1:n_spin
            for (i, ik) in enumerate(DFTK.krange_spin(band_data.basis, σ))
                @test all_eigenvalues[ik] ≈  eigenvalues_resh[σ, i, :] atol=1e-12
                @test all_occupation[ik]  ≈  occupation_resh[σ, i, :]  atol=1e-12
                @test all_n_iter[ik]      == n_iter_resh[σ, i]
            end
        end
    end  # master
end  # function
end  # module

@testitem "todict" setup=[TestCases, DictAgreement] begin
using Test
using DFTK
testcase = TestCases.silicon

function test_todict(label; spin_polarization=:none, Ecut=7, temperature=0.0,
                     kgrid=MonkhorstPack(1, 1, 1))
    if spin_polarization == :collinear
        magnetic_moments = [1.0, 1.0]
    else
        magnetic_moments = []
    end
    model = model_LDA(testcase.lattice, testcase.atoms, testcase.positions;
                      spin_polarization, temperature, magnetic_moments)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    nbandsalg = FixedBands(; n_bands_converge=8)
    scfres = self_consistent_field(basis; tol=1e-1, nbandsalg)
    bands  = compute_bands(scfres, kgrid; n_bands=8, tol=1e-1)

    function randomize_intarray!(data)
        data .= rand(1:10, size(data))
    end
    randomize_intarray!(last(bands.diagonalization).n_iter)
    randomize_intarray!(last(scfres.diagonalization).n_iter)

    # This also tests Model.todict, Basis.todict
    dict_scfres = DFTK.band_data_to_dict(scfres)
    dict_bands  = DFTK.band_data_to_dict(bands)


    @testset "$label" begin
        DictAgreement.test_dict_agreement(scfres, dict_scfres; explicit_reshape=false)
        DictAgreement.test_dict_agreement(bands,  dict_bands;  explicit_reshape=false)
    end
end

test_todict("nospin notemp";  spin_polarization=:none)
test_todict("collinear temp"; spin_polarization=:collinear, kgrid=MonkhorstPack(2, 1, 3))
end
