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
Ecut_ref = 80   # kinetic energy cutoff in Hartree
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

ham = scfres_ref.ham

φ_ref = similar(scfres_ref.ψ)
for ik = 1:Nk
    φ_ref[ik] = scfres_ref.ψ[ik][:,1:N]
end

λ_ref = scfres_ref.eigenvalues[1][1]

b = φ_ref[1][:,1]

f_ref = forces(scfres_ref)

Ecut_list = 50:5:Ecut_ref-30
diff_list = []
diff_list_N1 = []
diff_list_N2 = []
diff_list_schur = []
diff_list_N1schur = []
forces_list = []

v = nothing
Mv = nothing

for Ecut in Ecut_list

    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

    ## solve linear system
    bn = keep_LF([b], basis_ref, Ecut)[1]
    function apply_ham_proj(x)
        x = keep_LF([x], basis_ref, Ecut)[1]
        x = ham.blocks[1] * x + (1 - λ_ref) * x
        keep_LF([x], basis_ref, Ecut)[1]
    end

    function apply_ham(x)
        ham.blocks[1] * x + (1 - λ_ref) * x
    end

    φ, info = linsolve(apply_ham_proj, bn, tol=1e-14)
    φr = [φ/norm(φ)]

    ## explicit computation of f-f*
    ρ = compute_density(basis_ref, φr, occupation)
    f = forces(basis_ref.terms[2], φr, occupation; ρ=ρ[1])
    append!(forces_list, abs(f[1][2][1]-f_ref[1][2][1]))

    ## f_est ~ |f-f*|
    # compute residual
    res = [apply_ham(φr[1]) - b]
    err = compute_error(basis_ref, φr, φ_ref)

    ## prepare Pks
    kpt = basis_ref.kpoints[1]
    Pks = [PreconditionerTPA(basis_ref, kpt) for kpt in basis_ref.kpoints]
    t = nothing
    for ik = 1:length(Pks)
        DFTK.precondprep!(Pks[ik], φr[ik])
        t = Pks[ik].mean_kin[1]
    end

    ## compute error on LF with Schur
    V = DFTK.total_local_potential(scfres_ref.ham)
    Mres = apply_inv_M(basis_ref, φr, Pks, res)
    Mres_real = G_to_r(basis_ref, kpt, Mres[1][:,1])
    VMres_real = (V .+  (1 - λ_ref) .- t) .* Mres_real
    VMres = r_to_G(basis_ref, kpt, VMres_real)
    VMres_LF = keep_LF([VMres], basis_ref, Ecut)
    eLF, info = linsolve(apply_ham_proj, -VMres_LF[1], tol=1e-14)

    ## compute residual with one Neumann step
    V = DFTK.total_local_potential(scfres_ref.ham)
    Mres = apply_inv_M(basis_ref, φr, Pks, res)
    Mres_real = G_to_r(basis_ref, kpt, Mres[1][:,1])
    VMres_real = (V .+  (1 - λ_ref) .- t) .* Mres_real
    VMres = r_to_G(basis_ref, kpt, VMres_real)
    res_N1 = [res[1] .- VMres]

    ## compute residual with two Neumann steps
    MVMres = apply_inv_M(basis_ref, φr, Pks, [VMres])
    MVMres_real = G_to_r(basis_ref, kpt, MVMres[1][:,1])
    VMVMres_real = (V .+ (1 - λ_ref) .- t) .* MVMres_real
    VMVMres = r_to_G(basis_ref, kpt, VMVMres_real)
    res_N2 = [res_N1[1] .+ VMVMres]

    # Apply M^+-1/2
    Mres = apply_inv_sqrt_M(basis_ref, φr, Pks, res)
    Mres_N1 = apply_inv_sqrt_M(basis_ref, φr, Pks, res_N1)
    Mres_N2 = apply_inv_sqrt_M(basis_ref, φr, Pks, res_N2)
    MeLF = apply_sqrt_M(φr, Pks, [eLF])
    Mschur = [Mres[1] + MeLF[1]]
    M_N1schur = keep_HF(Mres_N1, basis_ref, Ecut)
    M_N1schur = [M_N1schur[1] + MeLF[1]]
    Merr = apply_sqrt_M(φr, Pks, err)

    # approximate forces
    f_res, cs = compute_forces_estimate(basis_ref, Mres, φr, Pks)
    f_res_N2, cs_N2 = compute_forces_estimate(basis_ref, Mres_N2, φr, Pks)
    f_schur, cs_schur = compute_forces_estimate(basis_ref, Mschur, φr, Pks)
    f_N1schur, cs_N1schur = compute_forces_estimate(basis_ref, M_N1schur, φr, Pks)
    f_err, cs_err = compute_forces_estimate(basis_ref, Merr, φr, Pks)

    append!(diff_list, abs(f_res[1][2][1]))
    append!(diff_list_N2, abs(f_res_N2[1][2][1]))
    append!(diff_list_schur, abs(f_schur[1][2][1]))
    append!(diff_list_N1schur, abs(f_N1schur[1][2][1]))

    #  plot for test
    global Mv
    G_energies = DFTK.G_vectors_cart(basis_ref.kpoints[1])
    normG = norm.(G_energies)

    figure()
    plot(abs.(Mv[sortperm(normG)]), label="potentiel")
    xlabel("index of G by increasing norm")
    legend()

    figure()
    plot(abs.(Merr[1][sortperm(normG)]), label="err")
    plot(abs.(Mres[1][sortperm(normG)]), label="res")
    xlabel("index of G by increasing norm")
    legend()

    figure()
    plot(abs.(Merr[1][sortperm(normG)]), label="err")
    plot(abs.(Mschur[1][sortperm(normG)]), label="res schur")
    xlabel("index of G by increasing norm")
    legend()

    figure()
    plot(abs.(Merr[1][sortperm(normG)]), label="err")
    plot(abs.(M_N1schur[1][sortperm(normG)]), label="res schur LF + N1 HF")
    xlabel("index of G by increasing norm")
    legend()

    figure()
    plot(abs.(Merr[1][sortperm(normG)]), label="err")
    plot(abs.(Mres_N2[1][sortperm(normG)]), label="res N2")
    plot(abs.(Mres_N1[1][sortperm(normG)]), label="res N1")
    xlabel("index of G by increasing norm")
    legend()

    figure()
    title("produits conj(potential) * vec, où vec est dans la légende")
    plot(real.(conj.(Mv[sortperm(normG)]) .* Merr[1][sortperm(normG)]), label="err")
    plot(real.(conj.(Mv[sortperm(normG)]) .* Mschur[1][sortperm(normG)]), label="schur")
    plot(real.(conj.(Mv[sortperm(normG)]) .* Mres_N2[1][sortperm(normG)]), label="res N2")
    plot(real.(conj.(Mv[sortperm(normG)]) .* Mres[1][sortperm(normG)]), label="res")
    xlabel("index of G by increasing norm")
    legend()

    figure()
    plot([f_ref[1][2][1] - f[1][2][1] for G in G_energies], "k-")

    plot([f_res[1][2][1] for G in G_energies], "b--")
    plot(cumsum(cs), "b-", label="res")

    plot([f_res_N2[1][2][1] for G in G_energies], "y--")
    plot(cumsum(cs_N2), "y-", label="res N2")

    plot([f_N1schur[1][2][1] for G in G_energies], "g--")
    plot(cumsum(cs_N1schur), "g-", label="res schur LF + N1 HF")

    plot([f_err[1][2][1] for G in G_energies], "r--")
    plot(cumsum(cs_err), "r-", label="err")
    xlabel("index of G by increasing norm")
    title("cumsum")
    legend()
    STOP
end

figure()
rc("font", size=16)
title("Ecutref=$(Ecut_ref)")
semilogy(Ecut_list, forces_list, "r", label = "|F - F*|")
semilogy(Ecut_list, diff_list, "b-", label = "res")
semilogy(Ecut_list, diff_list_N2, "g-", label = "res N2")
semilogy(Ecut_list, diff_list_schur, "b:", label = "res schur")
semilogy(Ecut_list, diff_list_N1schur, "b--", label = "res schur LF + N1 HF")
xlabel("Ecut")
legend()

