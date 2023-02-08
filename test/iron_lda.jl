include("run_scf_and_compare.jl")
include("testcases.jl")

function run_iron_lda(T; kwargs...)
    # These values were computed using ABINIT with the same kpoints as testcases.jl
    # and Ecut = 15. They are not yet converged and thus require the same discretisation
    # parameters to be obtained.
    ref_lda = [
        [ 0.055335160026957, 0.318268719950663, 0.318268719983204, 0.453844901754021,
          0.465456022940131, 0.465456022955040, 0.796938936853792, 0.9378278608989292],
        [ 0.202562784343803, 0.257484383305897, 0.298153168492320, 0.484002264268755,
          0.486738682667850, 0.586413954242265, 0.606054175235276, 0.7921984533391616],
        [-0.025828679734167, 0.379219384359043, 0.379219384380489, 0.412266299189737,
          0.421421148410838, 0.469007916884015, 1.014229786618585, 1.082646799659422],
        [ 0.122870253157049, 0.284288157581980, 0.344201118979085, 0.418278709282607,
          0.470914403053476, 0.473315810505416, 0.712342007821782, 0.8817040989078889],
        [ 0.249396221271434, 0.249396221271986, 0.283803391450257, 0.464807045105138,
          0.464807045129581, 0.600140396214226, 0.641661514945205, 0.6416615149484401],
        [ 0.215115530345620, 0.230559459189795, 0.413101385500933, 0.413101385526945,
          0.443282733300631, 0.476867020334080, 0.702492626996283, 0.7024926270118694],
        [ 0.099871712572642, 0.424238848817503, 0.424238848844880, 0.596529745402267,
          0.628841951719554, 0.628841951734441, 0.891993361117189, 1.0316903612443131],
        [ 0.273624284119989, 0.308366088258763, 0.394136428473583, 0.637445234492337,
          0.660367125811219, 0.695536341395981, 0.759456147985534, 0.9049047470326816],
        [ 0.016725290665167, 0.501424956361337, 0.501424956379435, 0.543811076656092,
          0.565010802783482, 0.63953000442497 , 1.097171561169634, 1.1851677714770334],
        [ 0.172968957815038, 0.369175928007706, 0.455665586569160, 0.556763288068905,
          0.626381596590897, 0.636563034574486, 0.834717579200534, 0.9711639337454779],
        [ 0.323081016250375, 0.323081016251097, 0.370943969450466, 0.609089414206416,
          0.609089414224977, 0.713615164563541, 0.776414016931158, 0.7764140169376185],
        [ 0.283999916316325, 0.339176502497316, 0.523562806877426, 0.523562806894741,
          0.576405064140235, 0.604381023304363, 0.808531656348124, 0.8085316563716097],
    ]
    ref_etot = -16.670871429685356

    magnetic_moments = [4.0]
    model = model_DFT(iron_bcc.lattice, iron_bcc.atoms, iron_bcc.positions,
                      [:lda_xc_teter93]; temperature=0.01, magnetic_moments,
                      smearing=Smearing.FermiDirac())
    model = convert(Model{T}, model)
    basis = PlaneWaveBasis(model; Ecut=15, fft_size=[20, 20, 20],
                           kgrid=[4, 4, 4], kshift=[1/2, 1/2, 1/2])
    run_scf_and_compare(T, basis, ref_lda, ref_etot;
                        ρ=guess_density(basis, magnetic_moments), kwargs...)
end

@testset "Iron LDA (Float64)" begin
    run_iron_lda(Float64, test_tol=5e-6, scf_tol=1e-11)
end
