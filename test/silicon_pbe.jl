@testmodule SiliconPBE begin
using DFTK
using ..RunSCF: run_scf_and_compare
using ..TestCases: silicon

function run_silicon_pbe(T; Ecut=5, grid_size=15, spin_polarization=:none, kwargs...)
    # These values were computed using ABINIT with the same kpoints as testcases.jl
    # and Ecut = 25
    ref_pbe = [
        [-0.181210259413818, 0.258840553222639, 0.258840553225549, 0.258840553228459,
          0.351692348652324, 0.351692348656259, 0.351692348660193, 0.380606400669216,
          0.540705881744348, 0.540705883460555],
        [-0.130553299114991, 0.062256443775155, 0.221871391287580, 0.221871391290802,
          0.322398722411882, 0.386194327436667, 0.386194327439986, 0.546859898649217,
          0.550571701390781, 0.550571701394327],
        [-0.111170738096744, 0.074494899973125, 0.169461730083372, 0.169461730088140,
          0.284305392082236, 0.330468937070505, 0.524509288492752, 0.524509288496625,
          0.616964090764029, 0.619623658242765],
        [-0.061054203629684, 0.009700769243041, 0.095769985640881, 0.180784778430457,
          0.315000287382235, 0.471042322838057, 0.495281775946584, 0.517469860611792,
          0.530124341745161, 0.539044739392045],
    ]
    ref_etot = -7.854477356672080

    Si = ElementPsp(silicon.atnum; psp=load_psp("hgh/pbe/si-q4"))
    atoms = [Si, Si]

    if spin_polarization == :collinear
        magnetic_moments = zero.(silicon.positions)
    else
        magnetic_moments = []
    end
    model = model_PBE(silicon.lattice, atoms, silicon.positions; spin_polarization,
                      magnetic_moments)
    model = convert(Model{T}, model)
    basis = PlaneWaveBasis(model; Ecut, kgrid=(3, 3, 3), fft_size=fill(grid_size, 3))

    spin_polarization == :collinear && (ref_pbe = vcat(ref_pbe, ref_pbe))
    run_scf_and_compare(T, basis, ref_pbe, ref_etot; ρ=guess_density(basis), kwargs...)
end
end


@testitem "Silicon PBE (small, Float64)" #=
    =#    tags=[:core] setup=[RunSCF, TestCases, SiliconPBE] begin
    SiliconPBE.run_silicon_pbe(Float64; Ecut=7, test_tol=0.03, n_ignored=0, grid_size=17)
end

@testitem "Silicon PBE (small, Float32)" #=
    =#    tags=[:core] setup=[RunSCF, TestCases, SiliconPBE] begin
    SiliconPBE.run_silicon_pbe(Float32; Ecut=7, test_tol=0.03, n_ignored=1, grid_size=17,
                               scf_ene_tol=1e-4)
end

@testitem "Silicon PBE (large, Float64)" #=
    =#    tags=[:slow, :core] setup=[RunSCF, TestCases, SiliconPBE] begin
    SiliconPBE.run_silicon_pbe(Float64; Ecut=25, test_tol=1e-5, n_ignored=0, grid_size=33,
                               scf_ene_tol=1e-8)
end

@testitem "Silicon PBE (small, collinear spin)" #=
    =#    tags=[:core] setup=[RunSCF, SiliconPBE, TestCases] begin
    SiliconPBE.run_silicon_pbe(Float64; Ecut=7, test_tol=0.03, n_ignored=0, grid_size=17,
                               spin_polarization=:collinear)
end

@testitem "Silicon PBE (large, collinear spin)" #=
    =#    tags=[:slow, :core] setup=[RunSCF, SiliconPBE, TestCases] begin
    SiliconPBE.run_silicon_pbe(Float64; Ecut=25, test_tol=1e-5, n_ignored=0, grid_size=33,
                               scf_ene_tol=1e-8, spin_polarization=:collinear)
end
