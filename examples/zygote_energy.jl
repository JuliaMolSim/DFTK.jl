using DFTK
using Zygote
import FiniteDiff

a = 10.26
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp(:Si, functional="lda"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]
# model = model_DFT(lattice, atoms, [:lda_x, :lda_c_vwn]) # xc not working yet (mutating)
terms = [
    Kinetic(),
    AtomicLocal(),
    # AtomicNonlocal(),
    # Ewald(),
    # PspCorrection(),
    # Entropy(),
    # Hartree()
]
model = Model(lattice; atoms=atoms, terms=terms, symmetries=false)
# model = model_DFT(lattice, atoms, [], symmetries=false)
kgrid = [2, 2, 2]
Ecut = 7
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

# scfres = self_consistent_field(basis, is_converged=DFTK.ScfConvergenceDensity(1e-13))
scfres = self_consistent_field(basis, is_converged=DFTK.ScfConvergenceDensity(1e6))
ψ = scfres.ψ
occupation = scfres.occupation

function total_energy(ρ)
    energies, H = energy_hamiltonian(basis, ψ, occupation; ρ=ρ)
    sum(values(getfield(energies, :energies)))
end

total_energy(scfres.ρ)
# Zygote.gradient(total_energy, scfres.ρ) # ERROR

# Zygote.gradient(x -> sum(values(Dict(1=>x))), 1.0) # ERROR
# using OrderedCollections
# Zygote.gradient(x -> sum(values(OrderedDict{Int,Float64}(1=>x))), 1.0) ERROR
#
# # Zygote has issues with Dicts?
# https://github.com/FluxML/Zygote.jl/issues/760

# Approach 2: Try direct access to term energies

function HF_energy(basis, ψ, occupation, ρ)
    # TODO ρ = compute_density(basis, ψ, occupation)
    energies = [DFTK.ene_ops(term, ψ, occupation; ρ=ρ).E for term in basis.terms]
    sum(energies)
end
HF_energy(ρ) = HF_energy(basis, ψ, occupation, ρ) # only for debug purposes, TODO delete
HF_energy(scfres.ρ)
g1 = Zygote.gradient(HF_energy, scfres.ρ)[1] # works

# check against finite differences
g2 = FiniteDiff.finite_difference_gradient(HF_energy, scfres.ρ)
sum(abs, g1 - g2)  # 3.7460628693848023e-7

using BenchmarkTools
@btime HF_energy(scfres.ρ)  # 215.649 μs (278 allocations: 772.42 KiB)
@btime Zygote.gradient(HF_energy, scfres.ρ)[1];  # 2.044 ms (3413 allocations: 3.03 MiB)
@btime FiniteDiff.finite_difference_gradient(HF_energy, scfres.ρ);  # 4.559 s (4463509 allocations: 11.78 GiB)

# also try E w.r.t. ψ
HF_energy_psi(ψ) = HF_energy(basis, ψ, occupation, scfres.ρ)
Zygote.gradient(HF_energy_psi, ψ)
@btime HF_energy_psi(ψ);  # 192.225 μs (291 allocations: 776.19 KiB)
@btime Zygote.gradient(HF_energy_psi, ψ);  # 1.823 ms (3464 allocations: 3.11 MiB)


# E w.r.t. basis 

typeof.(basis.terms)
Zygote.gradient(basis -> DFTK.ene_ops(basis.terms[1], ψ, occupation).E, basis)
Zygote.gradient(basis -> DFTK.ene_ops(basis.terms[2], ψ, occupation; ρ=scfres.ρ).E, basis)

HF_energy(basis::PlaneWaveBasis) = HF_energy(basis, ψ, occupation, scfres.ρ)
HF_energy(basis) # -4.807121625456233
g = Zygote.gradient(HF_energy, basis)[1];
dump(g; maxdepth=2)
# TODO verify result
# look at forces


# model w.r.t. lattice parameter
Si = ElementPsp(:Si, psp=load_psp(:Si, functional="lda"))
function make_model(a)
    lattice = a / 2 * [[0. 1. 1.];
                       [1. 0. 1.];
                       [1. 1. 0.]]
    atoms = [Si => [ones(3)/8, -ones(3)/8]]
    # model = model_DFT(lattice, atoms, [], symmetries=false)
    terms = [
        Kinetic(),
        AtomicLocal(), # TODO
        # AtomicNonlocal(),
        # Ewald(),
        # PspCorrection(),
        # Entropy(),
        # Hartree()
    ]
    Model(lattice; atoms=atoms, terms=terms)
end
make_model(a)
Zygote.gradient(a -> make_model(a).recip_cell_volume, a) # (-0.2686157095138732,)
FiniteDiff.finite_difference_derivative(a -> make_model(a).recip_cell_volume, a) # -0.2686157095506202

function make_basis(model::Model)
    PlaneWaveBasis(model, Ecut; kgrid=kgrid)
end
make_basis(a::Real) = make_basis(make_model(a))
make_basis(a).G_to_r_normalization # 0.06085677788055191
Zygote.gradient(a -> make_basis(a).G_to_r_normalization, a)  # (-0.0088971897486187,)
FiniteDiff.finite_difference_derivative(a -> make_basis(a).G_to_r_normalization, a)  # -0.008897189749284017

Zygote.gradient(a -> make_basis(a).dvol, a)
FiniteDiff.finite_difference_derivative(a -> make_basis(a).dvol, a)

Zygote.gradient(a -> make_basis(a).model.recip_cell_volume, a)
FiniteDiff.finite_difference_derivative(a -> make_basis(a).model.recip_cell_volume, a)

Zygote.gradient(a -> make_basis(a).r_to_G_normalization, a)
FiniteDiff.finite_difference_derivative(a -> make_basis(a).r_to_G_normalization, a)

# TODO diff through term construction (pre-computations)

Zygote.gradient(a -> sum(make_basis(a).terms[1].kinetic_energies[1]), a)
FiniteDiff.finite_difference_derivative(a -> sum(make_basis(a).terms[1].kinetic_energies[1]), a)

# TODO next steps
# - include AtomicLocal
# - HF_energy compute_density

# TODO AtomicLocal 
# failing currently below
Zygote.gradient(a -> HF_energy(make_basis(a)), a)
FiniteDiff.finite_difference_derivative(a -> HF_energy(make_basis(a)), a)

# Zygote.gradient(a -> make_basis(a).terms[2].potential[1], a)
# FiniteDiff.finite_difference_derivative(a -> make_basis(a).terms[2].potential[1], a)


# atomiclocal = AtomicLocal()
# foo(basis) = atomiclocal(basis).potential[1]
# Tangent = DFTK.ChainRulesCore.Tangent
# h = 1e-8
# Zygote.gradient(foo, basis)
# FiniteDiff.finite_difference_derivative(h -> foo(basis + Tangent{typeof(basis)}(;model=Tangent{typeof(model)}(;recip_lattice=[h 0. 0.; 0. 0. 0.; 0. 0. 0.]))), 0.0)

# TODO compute_density

function HF_energy_recompute(basis, ψ, occupation)
    ρ = compute_density(basis, ψ, occupation)
    energies = [DFTK.ene_ops(term, ψ, occupation; ρ=ρ).E for term in basis.terms]
    sum(energies)
end
HF_energy_recompute(basis, ψ, occupation)
Zygote.gradient(a -> HF_energy_recompute(make_basis(a), ψ, occupation), a)



# TODO debug r_to_G, G_to_r rrules

# ∂ wi s(Aij xj) / ∂s = wi (Aij xj) = (wi Aij) xj

x = total_density(scfres.ρ)
Zygote.gradient(a -> sum(real(r_to_G(make_basis(a), x) .* x)), a)
FiniteDiff.finite_difference_derivative(a -> sum(real(r_to_G(make_basis(a), x) .* x)), a)

# r_to_G w.r.t x works
x2 = deepcopy(x)
Zygote.gradient(x -> sum(real(r_to_G(basis, x) .* x2)), x)
FiniteDiff.finite_difference_gradient(x -> sum(real(r_to_G(basis, x) .* x2)), x)

# r_to_G wrt r_to_G_normalization works
_basis(s) = PlaneWaveBasis{Float64}(
    basis.model, basis.fft_size, basis.dvol, 
    basis.Ecut, basis.variational,
    basis.opFFT, basis.ipFFT, basis.opBFFT, basis.ipBFFT,
    s, # r_to_G_normalization
    basis.G_to_r_normalization,
    basis.kpoints, basis.kweights, basis.ksymops, basis.kgrid, basis.kshift,
    basis.kcoords_global, basis.ksymops_global, basis.comm_kpts, basis.krange_thisproc, basis.krange_allprocs,
    basis.symmetries, basis.terms
)
f(s) = sum(real(r_to_G(_basis(s), x) .* x))
f(-0.5)
Zygote.gradient(f, -0.5)
FiniteDiff.finite_difference_derivative(f, -0.5)

# G_to_r wrt G_to_r_normalization works
_basis_G_to_r(s) = PlaneWaveBasis{Float64}(
    basis.model, basis.fft_size, basis.dvol, 
    basis.Ecut, basis.variational,
    basis.opFFT, basis.ipFFT, basis.opBFFT, basis.ipBFFT,
    basis.r_to_G_normalization,
    s, #G_to_r_normalization,
    basis.kpoints, basis.kweights, basis.ksymops, basis.kgrid, basis.kshift,
    basis.kcoords_global, basis.ksymops_global, basis.comm_kpts, basis.krange_thisproc, basis.krange_allprocs,
    basis.symmetries, basis.terms
)
y = r_to_G(basis, x)
f_G_to_r(s) = sum(real(G_to_r(_basis_G_to_r(s), y) .* ones(20,20,20)))
f_G_to_r(0.1)
Zygote.gradient(f_G_to_r, 0.1)
FiniteDiff.finite_difference_derivative(f_G_to_r, 0.1)
