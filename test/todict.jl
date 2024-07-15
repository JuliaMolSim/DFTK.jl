@testmodule DictAgreement begin
using Test
using DFTK
using MPI

function test_agreement_bands(band_data, dict; explicit_reshape=false, test_ψ=true)
    # NOTE: For MPI-parallel tests, this needs to be called on each processor

    basis = band_data.basis
    model = basis.model
    n_kpoints = length(basis.kcoords_global)
    n_spin    = model.n_spin_components
    n_bands   = length(band_data.eigenvalues[1])
    max_n_G   = DFTK.mpi_max(maximum(kpt -> length(G_vectors(basis, kpt)), basis.kpoints),
                             basis.comm_kpts)
    rotations       = [symop.W for symop in basis.symmetries]
    translations    = [symop.w for symop in basis.symmetries]

    function condreshape(data, shape...)
        if explicit_reshape
            reshape(data, shape...)
        else
            data
        end
    end

    n_iter_resh      = nothing
    eigenvalues_resh = nothing
    occupation_resh  = nothing
    resid_resh       = nothing
    n_G_resh         = nothing
    G_vecs_resh      = nothing
    ψ_resh           = nothing
    if mpi_master()
        # Tests that data required downstream (e.g. in Aiida) is present in the dict
        # and behaves as expected.

        @test dict["n_bands"]           == n_bands
        @test dict["n_kpoints"]         == n_kpoints
        @test dict["n_atoms"]           == length(model.atoms)
        @test dict["n_spin_components"] == n_spin
        @test dict["model_name"]        == model.model_name
        @test dict["temperature"]       ≈  model.temperature  atol=1e-12
        @test dict["smearing"]          == "$(model.smearing)"
        @test dict["atomic_symbols"]    == map(e -> string(atomic_symbol(e)), model.atoms)
        @test dict["atomic_positions"] ≈ model.positions atol=1e-12
        @test dict["εF"]        ≈  band_data.εF  atol=1e-12
        @test dict["kcoords"]   ≈  basis.kcoords_global  atol=1e-12
        @test dict["kweights"]  ≈  basis.kweights_global atol=1e-12
        @test dict["Ecut"]      ≈  basis.Ecut
        @test dict["dvol"]      ≈  basis.dvol atol=1e-12
        @test [dict["fft_size"]...]  == [basis.fft_size...]
        @test dict["symmetries_translations"] ≈ translations atol=1e-12
        @test dict["use_symmetries_for_kpoint_reduction"] == basis.use_symmetries_for_kpoint_reduction
        @test dict["symmetries_respect_rgrid"] == basis.symmetries_respect_rgrid

        lattice_resh = condreshape(dict["lattice"], 3, 3)
        rotations_resh = [condreshape(rot, 3, 3) for rot in dict["symmetries_rotations"]]
        @test lattice_resh   ≈ model.lattice atol=1e-12
        @test rotations_resh ≈ rotations     atol=1e-12

        diagon = dict["diagonalization"]
        n_iter_resh      = condreshape(diagon["n_iter"],         (n_kpoints, n_spin))
        resid_resh       = condreshape(diagon["residual_norms"], (n_bands, n_kpoints, n_spin))
        eigenvalues_resh = condreshape(dict["eigenvalues"],      (n_bands, n_kpoints, n_spin))
        occupation_resh  = condreshape(dict["occupation"],       (n_bands, n_kpoints, n_spin))

        if test_ψ
            n_G_resh    = condreshape(dict["kpt_n_G_vectors"],  (n_kpoints, n_spin))
            G_vecs_resh = condreshape(dict["kpt_G_vectors"],    (3, max_n_G, n_kpoints, n_spin))
            ψ_resh      = condreshape(dict["ψ"], (max_n_G, n_bands, n_kpoints, n_spin))
        end
    end
    n_iter_resh      = MPI.bcast(n_iter_resh,      0, MPI.COMM_WORLD)
    resid_resh       = MPI.bcast(resid_resh,       0, MPI.COMM_WORLD)
    eigenvalues_resh = MPI.bcast(eigenvalues_resh, 0, MPI.COMM_WORLD)
    occupation_resh  = MPI.bcast(occupation_resh,  0, MPI.COMM_WORLD)
    n_G_resh         = MPI.bcast(n_G_resh,         0, MPI.COMM_WORLD)
    G_vecs_resh      = MPI.bcast(G_vecs_resh,      0, MPI.COMM_WORLD)
    ψ_resh           = MPI.bcast(ψ_resh,           0, MPI.COMM_WORLD)

    for σ = 1:n_spin, ik = DFTK.krange_spin(basis, σ)
        ikgl = mod1(basis.krange_thisproc_allspin[ik], n_kpoints)  # global k-point index

        ldiag = last(band_data.diagonalization)
        @test n_iter_resh[ikgl, σ]         == ldiag.n_iter[ik]
        @test resid_resh[:, ikgl, σ]       ≈  ldiag.residual_norms[ik]  atol=1e-12
        @test eigenvalues_resh[:, ikgl, σ] ≈  band_data.eigenvalues[ik] atol=1e-12
        @test occupation_resh[:, ikgl, σ]  ≈  band_data.occupation[ik]  atol=1e-12

        if test_ψ
            @test n_G_resh[ikgl, σ] == length(basis.kpoints[ik].G_vectors)
            @test all(G_vecs_resh[:, iG, ikgl, σ] == G
                      for (iG, G) in enumerate(basis.kpoints[ik].G_vectors))
            @test ψ_resh[1:n_G_resh[ikgl, σ], :, ikgl, σ] ≈ band_data.ψ[ik] atol=1e-12
        end
    end
end  # function

function test_agreement_scfres(scfres, dict; explicit_reshape=false, test_ψ=true)
    test_agreement_bands(scfres, dict; explicit_reshape, test_ψ)

    function condreshape(data, shape...)
        if explicit_reshape
            reshape(data, shape...)
        else
            data
        end
    end

    if mpi_master()
        ρ_resh = condreshape(dict["ρ"], scfres.basis.fft_size...,
                             scfres.basis.model.n_spin_components)
        @test ρ_resh                ≈ scfres.ρ atol=1e-12
        @test dict["damping_value"] ≈ scfres.α atol=1e-12

        for key in keys(scfres.energies)
            @test dict["energies"][key] ≈ scfres.energies[key] atol=1e-12
        end
        @test dict["energies"]["total"] ≈ scfres.energies.total atol=1e-12

        # Note: For MPI runs, each processor may not gather the data exactly the same
        #       way, which can induce machine-epsilons inconsistencies. That is why we
        #       do not use strict equalities.
        for key in dict["scfres_extra_keys"]
            key == "damping_value" && continue
            if dict[key] isa Number
                @test dict[key] ≈  getproperty(scfres, Symbol(key))  atol=1e-12
            else
                @test dict[key] == getproperty(scfres, Symbol(key))
            end
        end

        # Check some keys that are relied upon downstream
        @test "converged"            in dict["scfres_extra_keys"]
        @test "occupation_threshold" in dict["scfres_extra_keys"]
        @test "n_bands_converge"     in dict["scfres_extra_keys"]
        @test "n_iter"               in dict["scfres_extra_keys"]
    end  # master
end  # function
end  # module

@testitem "todict" setup=[TestCases, DictAgreement] tags=[:serialisation] begin
using Test
using DFTK
testcase = TestCases.silicon

function test_todict(label; spin_polarization=:none, Ecut=7, temperature=0.0, kgrid)
    if spin_polarization == :collinear
        magnetic_moments = [1.0, 1.0]
    else
        magnetic_moments = []
    end
    model = model_LDA(testcase.lattice, testcase.atoms, testcase.positions;
                      spin_polarization, temperature, magnetic_moments)
    basis = PlaneWaveBasis(model; Ecut, kgrid, use_symmetries_for_kpoint_reduction=false)
    nbandsalg = FixedBands(; n_bands_converge=8)
    scfres = self_consistent_field(basis; tol=1e-1, nbandsalg)
    bands  = compute_bands(scfres, kgrid; n_bands=8, tol=1e-1)

    function randomize_intarray!(data)
        data .= rand(1:10, size(data))
    end
    randomize_intarray!(last(bands.diagonalization).n_iter)
    randomize_intarray!(last(scfres.diagonalization).n_iter)

    # This also tests Model.todict, Basis.todict
    dict_scfres = DFTK.scfres_to_dict(scfres;    save_ψ=true)
    dict_bands  = DFTK.band_data_to_dict(bands;  save_ψ=true)

    @testset "$label" begin
        DictAgreement.test_agreement_bands(bands,   dict_bands)
        DictAgreement.test_agreement_bands(scfres,  dict_scfres)
        DictAgreement.test_agreement_scfres(scfres, dict_scfres)
    end
end

test_todict("nospin notemp";  spin_polarization=:none,
            kgrid=MonkhorstPack(1, max(2, mpi_nprocs()), 1))  # At least one k-point per MPI proc.
test_todict("collinear temp"; spin_polarization=:collinear,
            kgrid=MonkhorstPack(2, 1, 3), temperature=1e-3)
end
