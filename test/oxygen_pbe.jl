include("run_scf_and_compare.jl")
include("testcases.jl")

function run_oxygen_pbe(T; kwargs...)
    # These values were computed using ABINIT at Ecut=13. They are not yet converged and
    # thus require the same discretization parameters to be obtained.
    ref_evals = [
        [ -1.214034187039492, -0.617266880915023, -0.370358278719988, -0.370358278712184,
          -0.352666293942291,  0.008939230950524,  0.008939230958503,  0.098061783836035],
        [ -1.177293790864051, -0.551559952680091, -0.318494822087423, -0.308762934378086,
          -0.308762934370473,  0.090371516203330,  0.090371516211105,  0.115096531941182],
    ]
    ref_etot = -29.505154342215235
    ref_magn = 1.99231275

    # Produce reference data and guess for this configuration
    Ecut = 13
    O = ElementPsp(o2molecule.atnum, psp=load_psp("hgh/pbe/O-q6.hgh"))
    magnetic_moments = [O => [1., 1.]]
    model = model_PBE(Array{T}(o2molecule.lattice), [O => o2molecule.positions],
                      temperature=0.02, smearing=smearing=Smearing.Gaussian(),
                      magnetic_moments=magnetic_moments)
    basis = PlaneWaveBasis(model, Ecut; fft_size=[24, 24, 30], kgrid=[1, 1, 1])

    scfres = run_scf_and_compare(T, basis, ref_evals, ref_etot;
                                 ρ=guess_density(basis, magnetic_moments),
                                 test_etot=false,
                                 kwargs...)
    @test scfres.energies.total ≈ ref_etot atol=1e-4  # A little large a difference ...

    magnetization = sum(spin_density(scfres.ρ)) * basis.dvol
    @test magnetization ≈ ref_magn atol=1e-4
end

@testset "Oxygen PBE (Float64)" begin
    run_oxygen_pbe(Float64, test_tol=5e-5, scf_tol=1e-8, n_ignored=1)
end
