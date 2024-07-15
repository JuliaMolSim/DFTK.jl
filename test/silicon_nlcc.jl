@testmodule SiliconNLCC begin
using DFTK
using LazyArtifacts
using ..RunSCF: run_scf_and_compare
using ..TestCases: silicon

function run_silicon_nlcc(T; Ecut=5, grid_size=15, spin_polarization=:none, kwargs...)
    # see testcases_ABINIT/silicon_NLCC
    ref_nlcc = [[-0.2642374089889072 ,  0.17633199048837778 ,  0.17633199049124157 ,
                  0.17633199049410528,  0.2676329232248921  ,  0.26763292322886734 ,
                  0.26763292323284316,  0.29254078044940535],
                [-0.21347758888465515, -0.02137765808090306 ,  0.1392708247046037  ,
                  0.13927082470777663,  0.23562749032329677 ,  0.302055464935735   ,
                  0.3020554649390603 ,  0.4562988597226669 ],
                [-0.19413530811649513, -0.008560806549330271,  0.0865935110974515  ,
                  0.08659351110215216,  0.19750492417788898 ,  0.24357603461949845 ,
                  0.4398784398950577 ,  0.43987843989908193],
                [-0.14378274415091313, -0.07334337538226694 ,  0.011438007879041803,
                  0.09799935803577159,  0.2283382740083082  ,  0.3849959774887962  ,
                  0.4107924302468714 ,  0.4317852399909192]]

    ref_etot = -8.50167205710043

    Si = ElementPsp(silicon.atnum;
                    psp=load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf", "Si.upf"))
    atoms = [Si, Si]

    if spin_polarization == :collinear
        magnetic_moments = zero.(silicon.positions)
    else
        magnetic_moments = []
    end
    model = model_DFT(Array{T}(silicon.lattice), atoms, silicon.positions,
                      [:lda_x, :lda_c_pw]; spin_polarization, magnetic_moments)
    basis = PlaneWaveBasis(model; Ecut, kgrid=(3, 3, 3), fft_size=fill(grid_size, 3))

    spin_polarization == :collinear && (ref_nlcc = vcat(ref_nlcc, ref_nlcc))
    run_scf_and_compare(T, basis, ref_nlcc, ref_etot; ρ=guess_density(basis), kwargs...)
end
end

@testitem "Silicon NLCC (small, Float64)" setup=[RunSCF, TestCases, SiliconNLCC] begin
    SiliconNLCC.run_silicon_nlcc(Float64; Ecut=7, test_tol=0.03, n_ignored=0, grid_size=17)
end

@testitem "Silicon NLCC (large, Float64)" #=
    =#    tags=[:slow] setup=[RunSCF, TestCases, SiliconNLCC] begin
    SiliconNLCC.run_silicon_nlcc(Float64; Ecut=25, test_tol=6e-5, n_ignored=0, grid_size=36,
                                 scf_ene_tol=1e-11)
end

@testitem "Silicon NLCC (small, collinear spin)" setup=[RunSCF, TestCases, SiliconNLCC] begin
    SiliconNLCC.run_silicon_nlcc(Float64; Ecut=7, test_tol=0.03, n_ignored=0, grid_size=17,
                                 spin_polarization=:collinear)
end

@testitem "Silicon NLCC (large, collinear spin)" #=
    =#    tags=[:slow] setup=[RunSCF, TestCases, SiliconNLCC] begin
    SiliconNLCC.run_silicon_nlcc(Float64; Ecut=25, test_tol=6e-5, n_ignored=0, grid_size=36,
                                 scf_ene_tol=1e-11, spin_polarization=:collinear)
end
