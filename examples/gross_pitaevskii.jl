## Gross-Pitaevskii equation: -1/2 Δ ψ + V ψ + α |ψ|^2 ψ = λ ψ, ||ψ||_L^2 = 1
## We emulate this with custom external potential V, and a custom xc term

using PyCall
using DFTK
using Printf

kgrid = [1, 1, 1] # No kpoints
Ecut = 4000

a = 10
lattice = a .* [[1 0 0.]; [0 0 0]; [0 0 0]] # unit cell. Having two lattice vectors as zero means a 1D system

f(x) = (x-a/2)^2 # potential
const α = 2.0

function external_pot(basis::PlaneWaveBasis, energy::Union{Ref, Nothing}, potential; ρ=nothing, kwargs...)
    model = basis.model
    N = basis.fft_size[1]
    x = a*basis.grids[1]

    # Compute Vext
    Vext = zeros(basis.fft_size)
    Vext[:, 1, 1] = f.(x)
    # Fill output as needed
    (potential !== nothing) && (potential .= Vext)
    if energy !== nothing
        dVol = model.unit_cell_volume / N # integration element. In 1D, unit_cell_volume is just the length
        energy[] = real(sum(real(ρ) .* Vext)) * dVol
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
        energy[] = α * dVol * sum(real(ρ) .^ 2)/2
    end
    if potential !== nothing
        potential .= α * real(ρ)
    end
    energy, potential
end
     
n_electrons = 1 # increase this for fun
# We add the needed terms
model = Model(lattice, n_electrons;
              external=external_pot,
              xc=nonlinearity,
              spin_polarisation=:spinless # "spinless fermions"
              )

kpoints, ksymops = bzmesh_uniform(kgrid) # create dummy BZ mesh
basis = PlaneWaveBasis(model, Ecut, kpoints, ksymops)

# We solve the self-consistent equation with an SCF algorithm (which
# is a pretty bad idea; implementing direct minimization is TODO)
n_bands_scf = model.n_electrons
ham = Hamiltonian(basis, Density(basis)) # zero initial guess for the density
scfres = self_consistent_field!(ham, model.n_electrons, tol=1e-6)

# Print obtained energies
energies = scfres.energies
println("\nEnergy breakdown:")
for key in sort([keys(energies)...]; by=S -> string(S))
    @printf "    %-20s%-10.7f\n" string(key) energies[key]
end
@printf "\n    %-20s%-15.12f\n\n" "total" sum(values(energies))


using PyPlot
x = a*basis.grids[1]
ρ = real(scfres.ρ)[:, 1, 1] # converged density
ψ_fourier = scfres.Psi[1][:, 1] # first kpoint, all G components, first eigenvector
ψ = G_to_r(basis, basis.kpoints[1], ψ_fourier)[:, 1, 1] # IFFT back to real space
@assert sum(abs2.(ψ)) * (x[2]-x[1]) ≈ 1.0

# phase fix
ψ /= (ψ[div(end, 2)] / abs(ψ[div(end, 2)]))

# ψ solves -Δ/2 ψ + Vext ψ + α ψ^2 ψ = λ ψ. Check that with finite differences
N = length(x)
A = diagm(-1=>-ones(N-1), 0=>2ones(N), 1=>-ones(N-1)) / (x[2]-x[1])^2 / 2
H = A + Diagonal(f.(x) + α .* ρ)

figure()
plot(x, ψ)
plot(x, ρ)
plot(x, H*ψ - dot(ψ, H*ψ)/dot(ψ, ψ)*ψ)
legend(("ψ", "ρ", "resid"))

figure()
plot(x, ham.pot_external[:, 1, 1])
plot(x, ham.pot_xc[:, 1, 1])
plot(x, ham.pot_local[:, 1, 1])
legend(("Vext", "Vnl", "Vtot"))
