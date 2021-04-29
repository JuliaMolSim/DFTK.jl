# Computation of error estimate and corrections for the forces for the linear
# silicon eigenvalue problem, using schur complement to enhance the residual on
# LF
#
# Very basic setup, useful for testing
using DFTK
using PyPlot
using KrylovKit

include("aposteriori_forces.jl")
include("aposteriori_tools.jl")
include("aposteriori_callback.jl")

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8 + [0.42, 0.35, 0.24] ./ 10, -ones(3)/8]]

model = Model(lattice; atoms=atoms, n_electrons=2,
              terms=[Kinetic(), AtomicLocal()])
nl = false
#  model = model_LDA(lattice, atoms; n_electrons=2)
kgrid = [1,1,1]  # k-point grid (Regular Monkhorst-Pack grid)
Ecut_ref = 100   # kinetic energy cutoff in Hartree
tol = 1e-10
tol_krylov = 1e-12
basis_ref = PlaneWaveBasis(model, Ecut_ref; kgrid=kgrid)

filled_occ = DFTK.filled_occupation(model)
N = div(model.n_electrons, filled_occ)
Nk = length(basis_ref.kpoints)
T = eltype(basis_ref)
occupation = [filled_occ * ones(T, N) for ik = 1:Nk]

scfres_ref = self_consistent_field(basis_ref, tol=tol,
                                   determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-10),
                                   is_converged=DFTK.ScfConvergenceDensity(tol))


φ_ref = similar(scfres_ref.ψ)
for ik = 1:Nk
    φ_ref[ik] = scfres_ref.ψ[ik][:,1:N]
end

λ_ref = scfres_ref.eigenvalues[1][1]
f_ref = forces(scfres_ref)

Ecut_ref = 40   # kinetic energy cutoff in Hartree
basis_ref = PlaneWaveBasis(model, Ecut_ref; kgrid=kgrid)

filled_occ = DFTK.filled_occupation(model)
N = div(model.n_electrons, filled_occ)
Nk = length(basis_ref.kpoints)
T = eltype(basis_ref)
occupation = [filled_occ * ones(T, N) for ik = 1:Nk]

scfres_ref = self_consistent_field(basis_ref, tol=tol,
                                   determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-10),
                                   is_converged=DFTK.ScfConvergenceDensity(tol))

ham_ref = scfres_ref.ham

Ecut_list = 5:5:40
diff_list = []
diff_list_N1 = []
diff_list_N2 = []
diff_list_schur = []
diff_list_N1schur = []
forces_list = []

v = nothing
Mv = nothing

for Ecut in Ecut_list

    println(Ecut)
    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

    ## solve eigenvalue system
    scfres = self_consistent_field(basis, tol=tol,
                                   determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-10),
                                   is_converged=DFTK.ScfConvergenceDensity(tol))
    ## quantities
    φ = similar(scfres.ψ)
    for ik = 1:Nk
        φ[ik] = scfres.ψ[ik][:,1:N]
    end
    λ = scfres.eigenvalues[1][1]
    f = forces(scfres)
    append!(forces_list, abs(f[1][2][1]-f_ref[1][2][1]))

    ## f_est ~ |f-f*|
    # compute residual
    φr = DFTK.interpolate_blochwave(φ, basis, basis_ref)
    res = compute_residual(basis_ref, φr, occupation)

    pack, unpack, packed_proj = packing(basis_ref, φr)

    function apply_ham_proj(x)
        x = unpack(x)
        x = keep_LF(x, basis_ref, Ecut)
        x = proj(x, φr)
        x = pack(x)
        x = ham_ref.blocks[1] * x .- λ .* x
        x = unpack(x)
        x = proj(x, φr)
        x = keep_LF(x, basis_ref, Ecut)
        pack(x)
    end

    ## prepare Pks
    kpt = basis_ref.kpoints[1]
    Pks = [PreconditionerTPA(basis_ref, kpt) for kpt in basis_ref.kpoints]
    t = nothing
    for ik = 1:length(Pks)
        DFTK.precondprep!(Pks[ik], φr[ik])
        t = Pks[ik].mean_kin[1]
    end

    ## compute error on LF with Schur
    V = DFTK.total_local_potential(ham_ref)
    Mres = apply_inv_M(basis_ref, φr, Pks, res)
    Mres_real = G_to_r(basis_ref, kpt, Mres[1][:,1])
    VMres_real = (V .- λ .- t) .* Mres_real
    VMres = r_to_G(basis_ref, kpt, VMres_real)
    VMres_LF = keep_LF([VMres], basis_ref, Ecut)
    eLF, info = linsolve(apply_ham_proj, pack(proj(-VMres_LF, φr)), tol=1e-14;
                         orth=OrthogonalizeAndProject(packed_proj, pack(φr)))

    # Apply M^+-1/2
    Mres = apply_inv_sqrt_M(basis_ref, φr, Pks, res)
    MeLF = apply_sqrt_M(φr, Pks, [eLF])
    Mschur = [Mres[1] + MeLF[1]]

    # approximate forces
    f_res, cs = compute_forces_estimate(basis_ref, Mres, φr, Pks)
    f_schur, cs_schur = compute_forces_estimate(basis_ref, Mschur, φr, Pks)

    append!(diff_list, abs(f[1][2][1]-f_ref[1][2][1]-f_res[1][2][1]))
    append!(diff_list_schur, abs(f[1][2][1]-f_ref[1][2][1]-f_schur[1][2][1]))
end

figure()
rc("font", size=16)
title("Ecutref=$(Ecut_ref)")
semilogy(Ecut_list, forces_list, "r", label = "|F - F*|")
semilogy(Ecut_list, diff_list, "b-", label = "res")
#  semilogy(Ecut_list, diff_list_N2, "g-", label = "res N2")
semilogy(Ecut_list, diff_list_schur, "b:", label = "res schur")
#  semilogy(Ecut_list, diff_list_N1schur, "b--", label = "res schur LF + N1 HF")
xlabel("Ecut")
legend()

