include("run_scf_and_compare.jl")
include("testcases.jl")

function run_silicon_lda(T ;Ecut=5, grid_size=15, spin_polarization=:none, kwargs...)
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

    fft_size = fill(grid_size, 3)
    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.atnum, functional="lda", family="hgh"))
    model = model_DFT(Array{T}(silicon.lattice), [Si => silicon.positions], [:lda_x, :lda_c_vwn],
                      spin_polarization=spin_polarization)
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)

    spin_polarization == :collinear && (ref_lda = vcat(ref_lda, ref_lda))
    run_scf_and_compare(T, basis, ref_lda, ref_etot;
                        ρ=guess_density(basis),
                        kwargs...)
end


@testset "Silicon LDA (small, Float64)" begin
    run_silicon_lda(Float64, Ecut=7, test_tol=0.03, n_ignored=0, grid_size=17, scf_tol=1e-5,
                    n_ep_extra=0)
end

if !isdefined(Main, :FAST_TESTS) || !FAST_TESTS
    @testset "Silicon LDA (large, Float64)" begin
        run_silicon_lda(Float64, Ecut=25, test_tol=1e-5, n_ignored=0,
                        grid_size=33, scf_tol=1e-7, n_ep_extra=0)
    end
end

@testset "Silicon LDA (small, Float32)" begin
    run_silicon_lda(Float32, Ecut=7, test_tol=0.03, n_ignored=1, grid_size=19, scf_tol=1e-4,
                    n_ep_extra=1)
end

@testset "Silicon LDA (small, collinear spin)" begin
    run_silicon_lda(Float64, Ecut=7, test_tol=0.03, n_ignored=0, grid_size=17,
                    scf_tol=1e-5, n_ep_extra=0, spin_polarization=:collinear)
end

if !isdefined(Main, :FAST_TESTS) || !FAST_TESTS
    @testset "Silicon LDA (large, collinear spin)" begin
        run_silicon_lda(Float64, Ecut=25, test_tol=1e-5, n_ignored=0,
                        grid_size=33, scf_tol=1e-7, n_ep_extra=0, spin_polarization=:collinear)
    end
end
