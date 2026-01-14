# # [Solving -div(a(x)∇u(x)) = f(x)](@id divAgrad-solver)
#
# This example demonstrates DFTK's flexibility by solving a PDE problem:
# -div(a(x)∇u(x)) = f(x) in 2D with periodic boundary conditions.
#
# The coefficient a(x) is a sum of a background uniform value and constant values
# in spherical inclusions. We solve this by minimizing the corresponding quadratic
# functional using the machinery from DFT calculations.

using DFTK
using LinearAlgebra
using LinearMaps
using Plots

#
# First, we define a new element type that represents a spherical inclusion.
# The inclusion modifies the coefficient a(x) by a constant value within a ball.
#

"""
Element representing a spherical inclusion that modifies the coefficient a(x)
in the div-grad problem. The inclusion has a constant value inside a ball of given radius.
"""
struct ElementSphericalInclusion{T} <: DFTK.Element
    a_value::T  # Value of the coefficient modification in the inclusion
    radius::T   # Radius of the spherical inclusion
end

# We need to implement the Fourier transform of the characteristic function of a ball
# For a ball of radius R centered at origin, the Fourier transform is:
# FT[χ_R](p) = 4π R³/3 * 3(sin(p*R) - p*R*cos(p*R))/(p*R)³
# However, this is for the characteristic function. For our coefficient a(x),
# we want the value a_value in the ball.
function DFTK.local_potential_fourier(el::ElementSphericalInclusion{T}, p) where {T}
    R = el.radius
    if p == 0
        # Integral of a_value over the ball: a_value * (4π/3) * R³
        return el.a_value * 4 * T(π) * R^3 / 3
    else
        pR = p * R
        # Fourier transform: a_value * 4π * R³ * (sin(pR) - pR*cos(pR)) / (pR)³
        return el.a_value * 4 * T(π) * R^3 * (sin(pR) - pR * cos(pR)) / (pR)^3
    end
end

function DFTK.local_potential_real(el::ElementSphericalInclusion, r::Real)
    # Real space: a_value inside the ball, 0 outside
    r <= el.radius ? el.a_value : 0.0
end

#
# Next, we define a new term for the div(a(x)∇) operator
#

"""
Term for -½∇⋅(a∇) operator where a is constructed from atomic-like contributions.
Similar to TermAtomicLocal but uses DivAgradOperator instead of RealSpaceMultiplication.
"""
struct TermAtomicDivAGrad{AT} <: DFTK.TermLinear
    A_values::AT  # The coefficient field a(x) on the real-space grid
end

"""
AtomicDivAGrad: Construct the coefficient field a(x) from atomic positions.
"""
struct AtomicDivAGrad{T}
    background_value::T  # Background uniform value of a(x)
end

function (divAgrad::AtomicDivAGrad)(basis::DFTK.PlaneWaveBasis{T}) where {T}
    # Compute the coefficient field a(x) = background + sum of inclusions
    # Note: DivAgradOperator implements -½∇⋅(A∇), but we want -∇⋅(a∇)
    # Therefore we need A = 2a
    
    # Start with contributions from each "atom" (spherical inclusion)
    local_pot = DFTK.compute_local_potential(basis)
    # These add to a(x), so we multiply by 2 to get the contribution to A(x)
    # Then add the background contribution
    A_values = 2 * divAgrad.background_value .+ 2 .* local_pot
    
    TermAtomicDivAGrad(A_values)
end

@DFTK.timing "ene_ops: divAgrad" function DFTK.ene_ops(term::TermAtomicDivAGrad,
                                                        basis::DFTK.PlaneWaveBasis{T}, 
                                                        ψ, occupation;
                                                        kwargs...) where {T}
    ops = [DFTK.DivAgradOperator(basis, kpt, term.A_values) for kpt in basis.kpoints]
    
    # For a linear problem, we don't need to compute the energy during the solve
    # The energy would be E = ½⟨ψ|H|ψ⟩ - ⟨f|ψ⟩, but we're solving Hψ = f
    E = T(Inf)
    
    (; E, ops)
end

# Custom preconditioner type that implements pseudo-inverse (zeroing DC component)
struct PseudoInversePreconditioner{T}
    diag::Vector{T}
    zero_idx::Int
end

function LinearAlgebra.ldiv!(y, P::PseudoInversePreconditioner, x)
    y .= x ./ (P.diag .+ 1)  # Add 1 to avoid division by zero
    y[P.zero_idx] = 0  # Zero out the DC component
    return y
end

#
# Solver function for the linear problem -div(a∇u) = f
#

"""
Solve the linear PDE problem -div(a(x)∇u(x)) = f(x) using CG iteration.

# Arguments
- `basis`: PlaneWaveBasis for the problem
- `f`: Right-hand side function values on the real-space grid

# Returns
- `u`: Solution on the real-space grid
- `info`: Information from the CG solver
"""
function solve_linear_problem(basis, f; tol=1e-6, maxiter=100)
    # Convert f to Fourier space and create right-hand side
    # We solve for the first k-point and first band (single equation)
    kpt = only(basis.kpoints)
    
    # Get the Hamiltonian (which represents our -div(a∇) operator)
    # We pass a dummy ψ and occupation to construct the Hamiltonian
    ψ_dummy = [DFTK.random_orbitals(basis, kpt, 1) for kpt in basis.kpoints]
    occupation_dummy = [fill(1.0, 1) for _ in basis.kpoints]
    
    eh = DFTK.energy_hamiltonian(basis, ψ_dummy, occupation_dummy)
    ham = eh.ham
    
    # Get Hamiltonian block for first k-point
    hamk = ham.blocks[1]
    
    # Setup the linear map for CG
    n_G = length(DFTK.G_vectors(basis, kpt))
    function apply_H(x)
        result = similar(x)
        LinearAlgebra.mul!(result, hamk, x)
        return result
    end
    
    T = real(eltype(basis))
    H_map = LinearMap{Complex{T}}(apply_H, n_G, n_G; ishermitian=true, isposdef=false)
    
    # Setup preconditioner - pseudo-inverse of diagonal kinetic energy
    # For the DivAGrad operator with roughly uniform a(x), the eigenvalues
    # scale like |k+G|², so we use this as a preconditioner
    kin_energies = [T(DFTK.norm2(kpt.coordinate + G) / 2) for G in DFTK.G_vectors_cart(basis, kpt)]
    
    # Find the index of G=0 for the pseudo-inverse
    G_zero_idx = findfirst(G -> all(iszero, G), DFTK.G_vectors(basis, kpt))
    @assert !isnothing(G_zero_idx)
    
    P = PseudoInversePreconditioner(kin_energies, G_zero_idx)
    
    # Initial guess (zero)
    u_fourier = zeros(Complex{T}, n_G)
    
    # Right-hand side in Fourier space
    b = DFTK.fft(basis, kpt, f)
    
    # Projection operator to enforce zero average
    function proj(x)
        # Remove the zero Fourier mode (constant component)
        x_copy = copy(x)
        x_copy[G_zero_idx] = 0
        return x_copy
    end
    
    # Also project b
    b = proj(b)
    
    # Solve using CG
    info = DFTK.cg!(u_fourier, H_map, b; precon=P, proj=proj, tol=tol, maxiter=maxiter)
    
    # Convert solution back to real space
    u = DFTK.ifft(basis, kpt, u_fourier)
    
    return u, info
end

#
# Example: Solve a sample problem
#

# Setup a 2D lattice
a = 10.0  # Box size
lattice = a .* [[1 0 0.]; [0 1 0]; [0 0 0]]  # 2D system

# Define spherical inclusions at specific positions
# These act as "atoms" that modify the coefficient a(x)
inclusion = ElementSphericalInclusion(4.0, 2.0)
positions = [[0.3, 0.2, 0.0], [0.7, 0.6, 0.0], [0.15, 0.8, 0.0]]
atoms = [inclusion, inclusion, inclusion]

# Background value of a(x)
background_a = 1.0

# Create model with our custom term
# Note: We use a minimal model without actual electrons
terms = [AtomicDivAGrad(background_a)]
n_electrons = 0  # No electrons, this is a pure PDE problem
model = DFTK.Model(lattice, atoms, positions; n_electrons, terms,
                   spin_polarization=:spinless)

# Create basis
Ecut = 50  # Energy cutoff for plane waves
basis = DFTK.PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1))

# Define the right-hand side f(x)
# Must have zero average for solvability: ∫f = 0
# Use a more interesting function
r_vectors = DFTK.r_vectors_cart(basis)
f_values = zeros(Float64, basis.fft_size...)
for (i, r) in enumerate(r_vectors)
    x, y = r[1], r[2]
    f_values[i] = sin(2π * x / a) * cos(2π * y / a) + 0.5 * sin(4π * x / a) * sin(4π * y / a)
end
# Ensure zero average
f_values .-= sum(f_values) / length(f_values)

# Solve the problem
println("Solving -div(a(x)∇u(x)) = f(x)...")
u, info = solve_linear_problem(basis, f_values; tol=1e-6, maxiter=200)
println("CG converged: $(info.converged) after $(info.n_iter) iterations")
println("Final residual: $(info.residual_norm)")

# Visualize the results
x_coords = [r[1] for r in r_vectors]
y_coords = [r[2] for r in r_vectors]

# Reshape for plotting (2D)
nx, ny, nz = basis.fft_size
X = reshape(x_coords, nx, ny, nz)[:, :, 1]
Y = reshape(y_coords, nx, ny, nz)[:, :, 1]
U = reshape(u, nx, ny, nz)[:, :, 1]
F = reshape(f_values, nx, ny, nz)[:, :, 1]

# Get coefficient a(x) for visualization
a_values = background_a .+ DFTK.compute_local_potential(basis)
A = reshape(a_values, nx, ny, nz)[:, :, 1]

# Create plots
p1 = heatmap(X[:, 1], Y[1, :], A', title="Coefficient a(x)", 
             xlabel="x", ylabel="y", c=:viridis)
p2 = heatmap(X[:, 1], Y[1, :], F', title="Right-hand side f(x)", 
             xlabel="x", ylabel="y", c=:RdBu)
p3 = heatmap(X[:, 1], Y[1, :], U', title="Solution u(x)", 
             xlabel="x", ylabel="y", c=:plasma)

p = plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))
