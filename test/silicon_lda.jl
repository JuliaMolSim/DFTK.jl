@testmodule SiliconLDA begin
using DFTK
using ..RunSCF: run_scf_and_compare
using ..TestCases: silicon

function run_silicon_lda(T; Ecut=5, grid_size=15, spin_polarization=:none, kwargs...)
    # These values were computed using ABINIT with the same kpoints as testcases.jl
    # and Ecut = 25
    ref_lda = [
        [-0.178566465714968, 0.261882541175914, 0.261882541178847, 0.261882541181782,
          0.354070367072414, 0.354070367076363, 0.354070367080310, 0.376871160884678],
        [-0.127794342370963, 0.064395861472044, 0.224958824747686, 0.224958824750934,
          0.321313617512188, 0.388442495007398, 0.388442495010722, 0.542078732298094],
        [-0.108449612789883, 0.077125812982728, 0.172380374761464, 0.172380374766260,
          0.283802499666810, 0.329872296009131, 0.525606867582028, 0.525606867585921],
        [-0.058089253154566, 0.012364292440522, 0.097350168867990, 0.183765652148129,
          0.314593174568090, 0.470869435132365, 0.496966579772700, 0.517009645871194],
    ]
    ref_etot = -7.911817522631488

    Si = ElementPsp(silicon.atnum; psp=load_psp("hgh/lda/si-q4"))
    atoms = [Si, Si]

    if spin_polarization == :collinear
        magnetic_moments = zero.(silicon.positions)
    else
        magnetic_moments = []
    end
    model = model_DFT(silicon.lattice, atoms, silicon.positions, [:lda_x, :lda_c_vwn];
                      spin_polarization, magnetic_moments)
    model = convert(Model{T}, model)
    basis = PlaneWaveBasis(model; Ecut, silicon.kgrid, fft_size=fill(grid_size, 3) )

    spin_polarization == :collinear && (ref_lda = vcat(ref_lda, ref_lda))
    run_scf_and_compare(T, basis, ref_lda, ref_etot; ρ=guess_density(basis), kwargs...)
end
end


@testitem "Silicon LDA (small, Float64)" #=
    =#    tags=[:minimal] setup=[RunSCF, TestCases, SiliconLDA] begin
    SiliconLDA.run_silicon_lda(Float64; Ecut=7, test_tol=0.03, n_ignored=0, grid_size=17,
                               scf_ene_tol=1e-5)
end

@testitem "Silicon LDA (large, Float64)" #=
    =#    tags=[:slow] setup=[RunSCF, TestCases, SiliconLDA] begin
    SiliconLDA.run_silicon_lda(Float64; Ecut=25, test_tol=1e-5, n_ignored=0, grid_size=33,
                               scf_ene_tol=1e-7)
end

@testitem "Silicon LDA (small, Float32)" #=
    =#    tags=[:minimal] setup=[RunSCF, TestCases, SiliconLDA] begin
    SiliconLDA.run_silicon_lda(Float32; Ecut=7, test_tol=0.03, n_ignored=1, grid_size=19,
                               scf_ene_tol=1e-4)
end

@testitem "Silicon LDA (small, collinear spin)" #=
    =#    tags=[:minimal] setup=[RunSCF, TestCases, SiliconLDA] begin
    SiliconLDA.run_silicon_lda(Float64; Ecut=7, test_tol=0.03, n_ignored=0, grid_size=17,
                               scf_ene_tol=1e-5, spin_polarization=:collinear)
end

@testitem "Silicon LDA (large, collinear spin)" #=
    =#    tags=[:slow] setup=[RunSCF, TestCases, SiliconLDA] begin
    SiliconLDA.run_silicon_lda(Float64; Ecut=25, test_tol=1e-5, n_ignored=0, grid_size=33,
                               scf_ene_tol=1e-7, spin_polarization=:collinear)
end
