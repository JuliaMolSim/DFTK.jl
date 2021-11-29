# # Polarizability using Zygote

using DFTK
using LinearAlgebra
using Zygote

## Construct PlaneWaveBasis given a particular electric field strength
## Again we take the example of a Helium atom.
He = ElementPsp(:He, psp=load_psp("hgh/lda/He-q2"))
atoms = [He => [[1/2; 1/2; 1/2]]]  # Helium at the center of the box
function make_basis(ε::T; a=10., Ecut=5) where T
    lattice=T(a) * I(3)  # lattice is a cube of ``a`` Bohrs
    # model = model_DFT(lattice, atoms, [:lda_x, :lda_c_vwn];
    #                   extra_terms=[ExternalFromReal(r -> -ε * (r[1] - a/2))],
    #                   symmetries=false)
    terms = [
        Kinetic(),
        AtomicLocal(),
        # AtomicNonlocal(),
        # Ewald(),
        # PspCorrection(),
        # Entropy(),
        # Hartree(),
        ExternalFromReal(r -> -ε * (r[1] - a/2))
    ]
    model = Model(lattice; atoms, terms, temperature=1e-3)
    PlaneWaveBasis(model, Ecut; kgrid=[1, 1, 1])  # No k-point sampling on isolated system
end

## dipole moment of a given density (assuming the current geometry)
function dipole(basis, ρ)
    # @assert isdiag(basis.model.lattice)
    a  = basis.model.lattice[1, 1]
    rr = [a * (r[1] - 1/2) for r in r_vectors(basis)]
    sum(rr .* ρ) * basis.dvol
end

## Function to compute the dipole for a given field strength
function compute_dipole(ε; tol=1e-8, kwargs...)
    scfres = self_consistent_field(make_basis(ε; kwargs...), tol=tol)
    dipole(scfres.basis, scfres.ρ)
end


# With this in place we can compute the polarizability from finite differences
# (just like in the previous example):
polarizability_fd = let
    ε = 0.01
    (compute_dipole(ε) - compute_dipole(0.0)) / ε
end
# 0.6198098991595362

Zygote.gradient(compute_dipole, 0.0) # TODO

# debug output:
# (size(ψ), size.(ψ)) = ((1,), [(7809, 1)])
# (size(∂ψ), size.(∂ψ)) = ((1,), [(7809, 8)])
# (size(occupation), size.(occupation)) = ((1,), [(1,)])
# after DFTK.select_occupied_orbitals:
# (size(∂ψ), size.(∂ψ)) = ((1,), [(7809, 1)])
# (size(∂Hψ), size.(∂Hψ)) = ((1,), [(7809, 1)])
# ∂H = ChainRulesCore.NoTangent()
# δenergies = ChainRulesCore.ZeroTangent()

# TODO next steps:
# find out why mul_pullback returns NoTangent here for ∂H
1+1


###
### debug H*ψ
###
using ChainRulesCore
using DFTK: _autodiff_apply_hamiltonian
scfres = self_consistent_field(make_basis(0.0), tol=1e-8)
H = scfres.ham;
dump(H; maxdepth=2)
ψ = scfres.ψ
Hψ, mul_pullback =
    rrule(Zygote.ZygoteRuleConfig(), *, H, ψ)

H * ψ
Hψ
Hψ ≈ H*ψ # true

_, ∂H, ∂ψ = mul_pullback(Hψ)
∂H
∂ψ
∂ψ ≈ H * Hψ # true


# typeof(∂H) =
Tangent{Hamiltonian, 
    NamedTuple{
        (:basis, :blocks), 
        Tuple{
            NoTangent, Vector{
                Tangent{HamiltonianBlock, 
                    NamedTuple{
                        (:basis, :kpoint, :operators, :optimized_operators, :scratch), 
                        Tuple{
                            Tangent{PlaneWaveBasis{Float64}, 
                                NamedTuple{
                                    (:model, :fft_size, :dvol, :Ecut, :variational, :opFFT, :ipFFT, :opBFFT, :ipBFFT, :r_to_G_normalization, :G_to_r_normalization, :kpoints, :kweights, :ksymops, :kgrid, :kshift, :kcoords_global, :ksymops_global, :comm_kpts, :krange_thisproc, :krange_allprocs, :symmetries, :terms), 
                                    Tuple{NoTangent, Tangent{Tuple{Int64, Int64, Int64}, Tuple{ComplexF64, ComplexF64, ComplexF64}}, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, ComplexF64, ComplexF64, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent}
                                }
                            }, NoTangent, NoTangent, Vector{Tangent}, NoTangent
                        }
                    }
                }
            }
        }
    }
}
# observe: basis.fft_size gets a Tuple{ComplexF64, ComplexF64, ComplexF64} which is weird and should not be needed

# challenge: differentiate energy_hamiltonian w.r.t. basis
(energies, H), energy_hamiltonian_pullback = 
        rrule_via_ad(Zygote.ZygoteRuleConfig(), energy_hamiltonian, scfres.basis, scfres.ψ, scfres.occupation, scfres.ρ)
energy_hamiltonian_pullback((NoTangent(), NoTangent()))
energy_hamiltonian_pullback((energies, ∂H))
# ERROR: Need an adjoint for constructor NamedTuple{(:E, :H), Tuple{Energies{Float64}, Hamiltonian}}. 
# Gradient is of type Tuple{Energies{Float64}, NamedTuple{(:basis, :blocks), Tuple{Nothing, Vector{Tangent{HamiltonianBlock, NamedTuple{(:basis, :kpoint, :operators, :optimized_operators, :scratch), Tuple{Tangent{PlaneWaveBasis{Float64}, NamedTuple{(:model, :fft_size, :dvol, :Ecut, :variational, :opFFT, :ipFFT, :opBFFT, :ipBFFT, :r_to_G_normalization, :G_to_r_normalization, :kpoints, :kweights, :ksymops, :kgrid, :kshift, :kcoords_global, :ksymops_global, :comm_kpts, :krange_thisproc, :krange_allprocs, :symmetries, :terms), Tuple{NoTangent, Tangent{Tuple{Int64, Int64, Int64}, Tuple{ComplexF64, ComplexF64, ComplexF64}}, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, ComplexF64, ComplexF64, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, NoTangent}}}, NoTangent, NoTangent, Vector{Tangent}, NoTangent}}}}}}}

# next challenge: rrule for HamiltonianBlock constructor
hb = H.blocks[1]
hb2, back = rrule_via_ad(Zygote.ZygoteRuleConfig(), HamiltonianBlock, hb.basis, hb.kpoint, hb.operators, hb.scratch)
back(hb2)

hb2

(energies, H), energy_hamiltonian_pullback = 
        rrule_via_ad(Zygote.ZygoteRuleConfig(), DFTK._autodiff_energy_hamiltonian, scfres.basis, scfres.ψ, scfres.occupation, scfres.ρ)
energy_hamiltonian_pullback((NoTangent(), ∂H))

optops, back = rrule_via_ad(Zygote.ZygoteRuleConfig(), ops -> DFTK.optimize_operators_(ops), hb.operators)

back(optops) # ERROR: LoadError: Need an adjoint for constructor DFTK.RealSpaceMultiplication. Gradient is of type DFTK.RealSpaceMultiplication


