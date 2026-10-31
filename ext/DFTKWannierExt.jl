module DFTKWannierExt

using DFTK
using Unitful
using UnitfulAtomic
import Wannier

"""
Build a Wannier.jl model that can be used for Wannierization.

The frozen window `dis_froz_min` to `dis_froz_max` for disentanglement is given in
Hartree, or as a `Unitful` energy. It is only used if `n_bands > n_wannier`.
`kmesh_tol` is the tolerance used by Wannier.jl to determine the b-vector shells.
"""
@DFTK.timing function Wannier.Model(scfres;
                                    n_bands=scfres.n_bands_converge,
                                    n_wannier=n_bands,
                                    projections=DFTK.default_wannier_centers(n_wannier),
                                    dis_froz_max=nothing,
                                    dis_froz_min=-Inf,
                                    kmesh_tol=1e-6)
    # TODO None of the routines consider spin at the moment
    @assert scfres.basis.model.spin_polarization in (:none, :spinless)
    @assert length(projections) == n_wannier
    scfres.basis.kgrid isa MonkhorstPack || error("The basis must be constructed from a MP grid.")

    # TODO Use band_data_to_dict to get this easily MPI compatible.

    # Undo symmetry operations to get full k-point list
    scfres_unfold = DFTK.unfold_bz(scfres)
    basis = scfres_unfold.basis
    ψ = scfres_unfold.ψ
    model = basis.model

    lattice = Wannier.Mat3(ustrip.(auconvert.(u"Å", model.lattice)))  # Wannier.jl uses Ångström
    recip_lattice  = Wannier.get_recip_lattice(lattice)
    atom_positions = reduce(hcat, model.positions)
    atom_labels    = [string(DFTK.element_symbol(element)) for element in model.atoms]
    kgrid          = Wannier.Vec3{Int}(basis.kgrid.kgrid_size)
    kpoints        = reduce(hcat, [Vector(kpt.coordinate) for kpt in basis.kpoints])

    # Finite-difference stencil (b-vectors) connecting each k to its neighbours k+b
    bvectors = Wannier.get_bvectors(kpoints, recip_lattice; kmesh_tol)
    n_kpts   = length(basis.kpoints)
    n_bvecs  = bvectors.n_bvecs

    # Overlaps M[:, :, ib, ik] = ⟨u_{m,k} | u_{n,k+b}⟩
    M = zeros(ComplexF64, n_bands, n_bands, n_bvecs, n_kpts)
    for ik = 1:n_kpts, ib = 1:n_bvecs
        ik_plus_b = bvectors.kpb_k[ib, ik]
        G_shift   = bvectors.kpb_b[:, ib, ik]
        M[:, :, ib, ik] = DFTK.overlap_Mmn_k_kpb(basis, ψ, ik, ik_plus_b, G_shift, n_bands)
    end

    # Initial gauge from projections, Löwdin-orthonormalised
    A = zeros(ComplexF64, n_bands, n_wannier, n_kpts)
    for (ik, (ψk, kpt)) in enumerate(zip(ψ, basis.kpoints))
        A[:, :, ik] = DFTK.compute_amn_kpoint(basis, kpt, ψk, projections, n_bands)
    end
    U = Wannier.orthonorm_lowdin(A)

    E    = reduce(hcat, [εk[1:n_bands] for εk in scfres_unfold.eigenvalues])
    E_eV = ustrip.(auconvert.(u"eV", E))  # Wannier.jl uses eV

    if n_bands != n_wannier && !isnothing(dis_froz_max) && !isnothing(dis_froz_min)
        dis_froz_min_eV::Float64 = ustrip(auconvert(u"eV", austrip(dis_froz_min)))
        dis_froz_max_eV::Float64 = ustrip(auconvert(u"eV", austrip(dis_froz_max)))
        frozen_bands = Wannier.get_frozen_bands(E_eV, dis_froz_max_eV, dis_froz_min_eV)
    else
        frozen_bands = falses(n_bands, n_kpts)
    end

    Wannier.Model(lattice, atom_positions, atom_labels, kgrid, kpoints, bvectors,
                  frozen_bands, M, U, E_eV)
end

end
