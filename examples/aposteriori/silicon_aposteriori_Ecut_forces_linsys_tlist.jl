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
Ecut_ref = 100           # kinetic energy cutoff in Hartree
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
f = nothing

Ecut_list = 5:5:Ecut_ref-30
forces_list = []
estimator_forces_list_err = []
estimator_forces_list_res = []
estimator_forces_list_res_N1 = []
estimator_forces_list_res_N2 = []

estimator_forces_list_err_HF = []
estimator_forces_list_res_HF = []

estimator_forces_list_err_φp = []
estimator_forces_list_err_φ  = []
estimator_forces_list_res_φp = []
estimator_forces_list_res_φ = []

ratio_list = []
ratio_list_N1 = []
ratio_list_N2 = []

v = nothing
CS_list_err = []
CS_list_res = []
CS_list_HF_err = []
CS_list_HF_res = []
CS_list_φp_err = []
CS_list_φp_res = []

diff_list_N1 = []
diff_list_N2 = []
t_list = vcat(0.1:0.1:1.5, 2:0.5:4)

for t in t_list

    Ecut = 50
    println("--------------------------")
    println("t=$(t)")

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
    println(norm(φ))
    φr = [φ/norm(φ)]
    #  φr = [φ]

    ## explicit computation of f-f*
    global f
    ρ = compute_density(basis_ref, φr, occupation)
    f = forces(basis_ref.terms[2], φr, occupation; ρ=ρ[1])
    append!(forces_list, norm(f[1][2][1]-f_ref[1][2][1]))

    ## f_est ~ |f-f*|
    # compute residual
    res = [apply_ham(φr[1]) - b]
    err = compute_error(basis_ref, φr, φ_ref)

    ## prepare Pks
    kpt = basis_ref.kpoints[1]
    Pks = [PreconditionerTPA(basis_ref, kpt) for kpt in basis_ref.kpoints]
    for ik = 1:length(Pks)
        DFTK.precondprep!(Pks[ik], φr[ik])
        Pks[ik].mean_kin[1] = t
        println(Pks[ik].mean_kin[1])
    end

    ## compute residual with one Neumann step
    V = DFTK.total_local_potential(scfres_ref.ham)
    Mres = apply_inv_M(basis_ref, φr, Pks, res)
    Mres_real = G_to_r(basis_ref, kpt, Mres[1][:,1])
    VMres_real = (V .+  (1 - λ_ref) .- t) .* Mres_real
    VMres = r_to_G(basis_ref, kpt, VMres_real)
    res_N1 = [res[1] .- VMres]

    #  ## compute residual with two Neumann steps
    MVMres = apply_inv_M(basis_ref, φr, Pks, [VMres])
    MVMres_real = G_to_r(basis_ref, kpt, MVMres[1][:,1])
    VMVMres_real = (V .+ (1 - λ_ref) .- t) .* MVMres_real
    VMVMres = r_to_G(basis_ref, kpt, VMVMres_real)
    res_N2 = [res_N1[1] .+ VMVMres]

    # Apply M^+-1/2
    Mres = apply_inv_sqrt_M(basis_ref, φr, Pks, res)
    Mres_N1 = apply_inv_sqrt_M(basis_ref, φr, Pks, res_N1)
    Mres_N2 = apply_inv_sqrt_M(basis_ref, φr, Pks, res_N2)
    Merr = apply_sqrt_M(φr, Pks, err)

    f_err, cs_err = compute_forces_estimate(basis_ref, Merr, φr, Pks)
    f_res, cs_res = compute_forces_estimate(basis_ref, Mres, φr, Pks)
    f_res_N1, cs_res_N1 = compute_forces_estimate(basis_ref, Mres_N1, φr, Pks)
    f_res_N2, cs_res_N2 = compute_forces_estimate(basis_ref, Mres_N2, φr, Pks)

    append!(diff_list_N1, f_res_N1[1][2][1])
    append!(diff_list_N2, f_res_N2[1][2][1])

    #  figure()
    #  title("t=$(t)")
    #  plot([f_ref[1][2][1] - f[1][2][1] for i in cs_res], "k-", label="f_ref")
    #  plot(cumsum(cs_err), "r-", label="cs err")
    #  plot([f_err[1][2][1] for i in cs_err], "r--", label="f_err")
    #  plot(cumsum(cs_res), "g-", label="cs res")
    #  plot([f_res[1][2][1] for i in cs_res], "g--", label="f_res")
    #  plot(cumsum(cs_res_N1), "b-", label="cs res_N1")
    #  plot([f_res_N1[1][2][1] for i in cs_res], "b--", label="f_res_N1")
    #  plot(cumsum(cs_res_N2), "y-", label="cs res_n2")
    #  plot([f_res_N2[1][2][1] for i in cs_res], "y--", label="f_res_N2")
    #  legend()
end
figure()
plot(t_list, diff_list_N1, label="N2")
plot(t_list, diff_list_N2, label="N1")
plot(t_list, [f_ref[1][2][1] - f[1][2][1] for t in t_list], "r--", label="f_err")
xlabel("t")
legend()



