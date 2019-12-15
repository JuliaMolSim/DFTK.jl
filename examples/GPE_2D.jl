## 2D Gross-Pitaevskii equation, with magnetic field, and with
## multiple electrons (of course, it doesn't make physical sense, but
## why not)

## This is pretty WIP, and only serves as a very rough demo. Nothing
## has been checked properly, so do not use for any serious purposes.

using PyCall
using DFTK
using Printf
using StaticArrays

kgrid = [1, 1, 1] # No kpoints
Ecut = 300

const a = 10
lattice = a .* [[1 0 0.]; [0 1 0]; [0 0 0]] # unit cell. Having one lattice vectors as zero means a 2D system

# Potential
V(x, y) = 1*((x-a/2)^2 + (y-a/2)^2)
# Vector potential: uniform magnetic field in the z direction
A(x, y) = .2 * @SVector [y-a/2, -(x-a/2)]
const α = 500.0

function external_pot(basis::PlaneWaveBasis, energy::Union{Ref, Nothing}, potential; ρ=nothing, kwargs...)
    model = basis.model
    N = prod(basis.fft_size)
    x = a*basis.grids[1]
    y = a*basis.grids[2]

    # Compute Vext
    Vext = zeros(basis.fft_size)
    Vext[:, :, 1] = V.(x, y')
    # Fill output as needed
    (potential !== nothing) && (potential .= Vext)
    if energy !== nothing
        dVol = model.unit_cell_volume / N # integration element
        energy[] = real(sum(ρ.real .* Vext)) * dVol
    end

    energy, potential
end

function magnetic_pot(basis::PlaneWaveBasis{T}, energy::Union{Ref, Nothing}, potential;
                      ρ=nothing, Psi=nothing, occupation=nothing, kwargs...) where {T}
    model = basis.model
    N = prod(basis.fft_size)
    x = a*basis.grids[1]
    y = a*basis.grids[2]

    Apot = [zeros(basis.fft_size) for i = 1:3]
    for i = 1:2
        Apot[i][:, :, 1] = [A(xx, yy)[i] for xx in x, yy in y]
    end

    # Fill output as needed
    (potential !== nothing) && (potential .= Apot)
    if energy !== nothing
        energy[] = zero(T)
        dVol = model.unit_cell_volume / N

        for ik = 1:length(basis.kpoints)
            kpt = basis.kpoints[ik]
            Psi_real = G_to_r(basis, kpt, Psi[ik])
            for i = 1:3
                all(Apot[i] .== 0) && continue
                pi = [(G[i] + kpt.coordinate[i]) for G in kpt.basis]
                ∂iPsi_fourier = pi .* Psi[ik]
                ∂iPsi_real = G_to_r(basis, kpt, ∂iPsi_fourier)
                for n = 1:size(Psi[ik], 2)
                    energy[] += real(dVol *
                                     occupation[ik][n] *
                                     (dot(Psi_real[:, :, :, n], Apot[i] .* ∂iPsi_real[:, :, :, n])))
                end
            end
        end
    end

    energy, potential
end


function nonlinearity(basis::PlaneWaveBasis, energy::Union{Ref,Nothing}, potential;
                        ρ=nothing, kwargs...)
    @assert ρ !== nothing
    model = basis.model
    potential !== nothing && (potential .= 0)

    if energy !== nothing
        dVol = basis.model.unit_cell_volume / prod(basis.fft_size)
        energy[] = α * dVol * sum(real(ρ.real).^ 2)/2
    end
    if potential !== nothing
        potential .= α * ρ.real
    end
    energy, potential
end
     
n_electrons = 2 # increase this for fun
# We add the needed terms
model = Model(lattice, n_electrons;
              external=external_pot,
              magnetic=magnetic_pot, 
              xc=nonlinearity,
              spin_polarisation=:spinless # "spinless fermions"
              )

kpoints, ksymops = bzmesh_uniform(kgrid) # create dummy BZ mesh
basis = PlaneWaveBasis(model, Ecut, kpoints, ksymops)

# # We solve the self-consistent equation with an SCF algorithm (which
# # is a pretty bad idea; implementing direct minimization is TODO)
# n_bands_scf = model.n_electrons
# ham = Hamiltonian(basis) # zero initial guess for the density
# scfres = self_consistent_field(ham, model.n_electrons, tol=1e-6)

scfres = direct_minimization(basis, x_tol=1e-6)
ham = scfres.ham

# Print obtained energies
energies = scfres.energies
println("\nEnergy breakdown:")
for key in sort([keys(energies)...]; by=S -> string(S))
    @printf "    %-20s%-10.7f\n" string(key) energies[key]
end
@printf "\n    %-20s%-15.12f\n\n" "total" sum(values(energies))

using PyPlot
x = a*basis.grids[1]
ρ = real(scfres.ρ.real)[:, :, 1] # converged density
ψ_fourier = scfres.Psi[1][:, 1] # first kpoint, all G components, first eigenvector
ψ = G_to_r(basis, basis.kpoints[1], ψ_fourier)[:, :, 1] # IFFT back to real space

figure()
pcolor(x, x, real(ρ))
for i = 1:n_electrons
    ψ_fourier = scfres.Psi[1][:, i] # first kpoint, all G components, first eigenvector
    ψ = G_to_r(basis, basis.kpoints[1], ψ_fourier)[:, :, 1] # IFFT back to real space
    figure()
    pcolor(x, x, abs2.(ψ))
end
nothing
