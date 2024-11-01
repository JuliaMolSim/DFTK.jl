using DFTK
using ForwardDiff
using LazyArtifacts
using LinearAlgebra

psp_hgh = load_psp("hgh/lda/Si-q4")
psp_upf = load_psp(artifact"hgh_lda_upf", "Si.pz-hgh.UPF")
psp_upf_pd = load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf/Si.upf")

lattice = [0.0  5.131570667152971 5.131570667152971;
            5.131570667152971 0.0 5.131570667152971;
            5.131570667152971 5.131570667152971  0.0]

function compute_force(ε::T; psp=psp_hgh, tol=1e-12, Ecut=5, kgrid=(2,2,2), return_scfres=false) where {T}
    positions = [[1.01, 1.02, 1.03] / 8, -ones(3) / 8 + ε * [1., 0, 0]]
    Si = ElementPsp(:Si; psp)
    atoms = [Si, Si]
    model = model_DFT(T.(lattice), atoms, positions; functionals=LDA(), symmetries=false)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    response     = ResponseOptions(; verbose=true)
    is_converged = DFTK.ScfConvergenceForce(tol)
    scfres = self_consistent_field(basis; is_converged, response)
    forces = compute_forces_cart(scfres)
    if return_scfres
        return (; scfres, forces)
    end
    forces
end

ε = 1e-5
results = Dict()
force_derivatives = Dict()
for psp in (psp_hgh, psp_upf)
    results[psp, -ε] = compute_force(-ε; psp, return_scfres=true)
    results[psp, +ε] = compute_force(+ε; psp, return_scfres=true)
    results[psp, 0.] = compute_force(0.; psp, return_scfres=true)

    force_derivatives[psp, "forward_difference"] = (results[psp,+ε].forces - results[psp,0.].forces) / ε
    force_derivatives[psp, "central_difference"] = (results[psp,+ε].forces - results[psp,-ε].forces) / 2ε
end

force_derivatives[psp_hgh, "forward_difference"]
force_derivatives[psp_upf, "forward_difference"]
force_derivatives[psp_hgh, "central_difference"]
force_derivatives[psp_upf, "central_difference"]

maximum.(abs, force_derivatives[psp_hgh, "central_difference"] - force_derivatives[psp_hgh, "forward_difference"])
maximum.(abs, force_derivatives[psp_hgh, "central_difference"] - force_derivatives[psp_upf, "central_difference"])


for psp in (psp_hgh, psp_upf)
    force_derivatives[psp, "forwarddiff"] = ForwardDiff.derivative(ε -> compute_force(ε; psp), 0.0)
end

force_derivatives[psp_upf, "forwarddiff"]
force_derivatives[psp_upf, "central_difference"]
maximum.(abs, force_derivatives[psp_hgh, "forwarddiff"] - force_derivatives[psp_upf, "forwarddiff"])
maximum.(abs, force_derivatives[psp_upf, "forwarddiff"] - force_derivatives[psp_upf, "central_difference"])


# Summary so far: ForwardDiff with HGH and UPF-HGH is consistent and agrees with finite differences

# PseudoDojo
for psp in (psp_upf_pd,)
    results[psp, -ε] = compute_force(-ε; psp, return_scfres=true)
    results[psp, +ε] = compute_force(+ε; psp, return_scfres=true)
    results[psp, 0.] = compute_force(0.; psp, return_scfres=true)

    force_derivatives[psp, "forward_difference"] = (results[psp,+ε].forces - results[psp,0.].forces) / ε
    force_derivatives[psp, "central_difference"] = (results[psp,+ε].forces - results[psp,-ε].forces) / 2ε

    force_derivatives[psp, "forwarddiff"] = ForwardDiff.derivative(ε -> compute_force(ε; psp), 0.0)
end

force_derivatives[psp_upf,    "central_difference"]
force_derivatives[psp_upf_pd, "central_difference"]
force_derivatives[psp_upf_pd, "forward_difference"]
force_derivatives[psp_upf_pd, "forwarddiff"]
maximum.(abs, force_derivatives[psp_upf_pd, "central_difference"] - force_derivatives[psp_upf_pd, "forwarddiff"]) # Large error ~0.05 !

# Problem: ForwardDiff on UPF-PD does not agree with finite difference.
#          Error is componentwise on the order 0.1.
#          But for UPF-HGH, the agreement with finite difference is ~1e-9.
# What are differences in UPF-HGH and UPF-PD?
# - UPF-PD includes core charge densities, UPF-HGH does not

