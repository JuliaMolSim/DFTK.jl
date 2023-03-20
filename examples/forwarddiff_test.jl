using DFTK
using PseudoPotentialIO
using ForwardDiff
using OffsetArrays
using LinearAlgebra

hgh_lda_family = "hgh_lda_hgh";
pd_lda_family = "pd_nc_sr_lda_standard_0.4.1_upf";
aluminium = (
    lattice = Matrix(Diagonal([4 * 7.6324708938577865, 7.6324708938577865,
                               7.6324708938577865])),
    atnum = 13,
    n_electrons = 12,
    psp_hgh = PseudoPotentialIO.load_psp(hgh_lda_family, "al-q3.hgh"),
    psp_upf = PseudoPotentialIO.load_psp(pd_lda_family, "Al.upf"),
    positions = [[0, 0, 0], [0, 1/2, 1/2], [1/8, 0, 1/2], [1/8, 1/2, 0]],
    temperature = 0.0009500431544769484,
);
aluminium = merge(aluminium,
                  (; atoms=fill(ElementPsp(aluminium.atnum, psp=aluminium.psp_hgh), 4)));

function compute_band_energies(ε::T) where {T}
    psp  = PseudoPotentialIO.load_psp("hgh_lda_hgh", "al-q3.hgh")
    rloc = convert(T, psp.rloc)
    cloc = convert.(T, psp.cloc)
    rnl = psp.rnl .+ OffsetVector([0, ε], angular_momenta(psp))
    D = map(i -> convert.(T, i), psp.D)

    pspmod = PseudoPotentialIO.HghPsP{T}(psp.checksum, psp.Zatom, psp.Zval, psp.lmax,
                                         rloc, cloc, rnl, D)

    atoms = fill(ElementPsp(aluminium.atnum, psp=pspmod), length(aluminium.positions))
    model = model_LDA(Matrix{T}(aluminium.lattice), atoms, aluminium.positions,
                        temperature=1e-2, smearing=Smearing.Gaussian())
    basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2], kshift=[0, 0, 0])

    println(pspmod)
    println(eltype(model))
    println(eltype(basis))

    is_converged = DFTK.ScfConvergenceDensity(1e-10)
    scfres = self_consistent_field(basis; is_converged, mixing=KerkerMixing(),
                                    nbandsalg=FixedBands(; n_bands_converge=10),
                                    damping=0.6, response=ResponseOptions(verbose=true))

    ComponentArray(
        eigenvalues=hcat([ev[1:10] for ev in scfres.eigenvalues]...),
        ρ=scfres.ρ,
        energies=collect(values(scfres.energies)),
        εF=scfres.εF,
        occupation=vcat(scfres.occupation...),
    )
end

derivative_ε = let ε = 1e-4
    (compute_band_energies(ε) - compute_band_energies(-ε)) / 2ε
end
derivative_fd = ForwardDiff.derivative(compute_band_energies, 0.0)