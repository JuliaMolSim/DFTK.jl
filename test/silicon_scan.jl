include("run_scf_and_compare.jl")
include("testcases.jl")

@testset "Silicon SCAN (small)" begin
    # These values were computed using ABINIT with the same setup
    ref_scan = [
        [-0.205978425740779,  0.25380659461563, 0.25380659461831, 0.254732985691879,
          0.359893487572120,  0.35989348757842, 0.36073308297652, 0.393192520642558],
        [-0.152952427803195,  0.05011058966751, 0.21579740516477, 0.216562129047254,
          0.329569264011892,  0.39586039396581, 0.39684227507671, 0.562830753144349],
        [-0.132594117409957,  0.06238473081469, 0.16156601427580, 0.162158286220009,
          0.288731746483282,  0.33738156594328, 0.54223142980221, 0.542824715864193],
        [-0.080135324991746, -0.00598988555554, 0.08632002434991, 0.173824020722616,
          0.321122229469835,  0.48210676303452, 0.51353080396070, 0.534196032300466],
    ]
    ref_etot = -7.856498623457256

    T = Float64
    Si = ElementPsp(silicon.atnum, psp=load_psp("hgh/pbe/Si-q4"))
    model = model_SCAN(Array{T}(silicon.lattice), [Si => silicon.positions])
    basis = PlaneWaveBasis(model; Ecut=15, fft_size=(27, 27, 27), kgrid=(3, 3, 3))
    run_scf_and_compare(T, basis, ref_scan, ref_etot; scf_tol=1e-9, test_tol=5e-5, n_ignored=1)
end


if !isdefined(Main, :FAST_TESTS)  # Only runs if this file is manually included
    @testset "Silicon SCAN (large)" begin
        # These values were computed using ABINIT with the same setup
        ref_scan = [
            [-0.205379663642861,  0.25371569916511, 0.25371569916800, 0.253715699170890,
              0.359500439613377,  0.35950043961729, 0.35950043962121, 0.394335128961021],
            [-0.152408007282470,  0.05049206064896, 0.21577864285782, 0.215778642861017,
              0.329623574443729,  0.39547375598451, 0.39547375598783, 0.562404353912307],
            [-0.132037173004760,  0.06278078694045, 0.16166915233730, 0.161669152342039,
              0.288276826894090,  0.33677227169080, 0.54118762648736, 0.541187626491229],
            [-0.079750743918380, -0.00519854971828, 0.08620156861818, 0.173319561343748,
              0.320653142337978,  0.48083236420357, 0.51209055187998, 0.533626303812142],
        ]
        ref_etot = -7.857384389260792

        T = Float64
        Si = ElementPsp(silicon.atnum, psp=load_psp("hgh/pbe/Si-q4"))
        model = model_SCAN(Array{T}(silicon.lattice), [Si => silicon.positions])
        basis = PlaneWaveBasis(model; Ecut=50, fft_size=(48, 48, 48), kgrid=(3, 3, 3))
        run_scf_and_compare(T, basis, ref_scan, ref_etot; test_tol=1e-8, n_ignored=2,
                            is_converged=DFTK.ScfConvergenceDensity(1e-9))
    end
end
