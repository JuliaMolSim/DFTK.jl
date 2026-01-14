# # [Solving -div(a(x)∇u(x)) = f(x)](@id divAgrad-solver)
#
# This example demonstrates DFTK's flexibility by solving a PDE problem:
# -div(a(x)∇u(x)) = f(x) in 2D with periodic boundary conditions.
#
# The coefficient a(x) is a sum of a background uniform value and constant values
# in inclusions (no longer limited to spherical symmetry). We solve this by 
# minimizing the corresponding quadratic functional using the machinery from DFT calculations.

using DFTK
using LinearAlgebra
using LinearMaps
using Plots

#
# Define the term for the div(a(x)∇) operator
#

"""
Term for -½∇⋅(A∇) operator where A is constructed from the coefficient field a(x).
Note: DivAgradOperator implements -½∇⋅(A∇), so we need A = 2a to get -∇⋅(a∇).
"""
struct TermDivAGrad{AT} <: DFTK.TermLinear
    A_values::AT  # The coefficient field A(x) = 2a(x) on the real-space grid
end

@DFTK.timing "ene_ops: divAgrad" function DFTK.ene_ops(term::TermDivAGrad,
                                                        basis::DFTK.PlaneWaveBasis{T}, 
                                                        ψ, occupation;
                                                        kwargs...) where {T}
    ops = [DFTK.DivAgradOperator(basis, kpt, term.A_values) for kpt in basis.kpoints]
    
    # For a linear problem, we don't need to compute the energy during the solve
    E = T(Inf)
    
    (; E, ops)
end

#
# Helper function to build coefficient field a(x) from inclusions
#

"""
Build the full coefficient field a(x) from elementary inclusions at given positions.
Handles periodic images (one cell away) in real space.

# Arguments
- `basis`: PlaneWaveBasis
- `inclusion_func`: Function that takes a position vector r::Vec3 and returns the inclusion value
- `background_value`: Background constant value of a(x)
- `positions`: List of inclusion center positions
- `use_periodic_images`: Whether to sum over periodic images (default: true)
"""
function build_coefficient_field_real(basis, inclusion_func, background_value, positions; 
                                      use_periodic_images=true)
    T = real(eltype(basis))
    a_values = fill(T(background_value), basis.fft_size...)
    
    # Get lattice vectors
    lattice = basis.model.lattice
    
    # Iterate over all real-space points
    for (i, r) in enumerate(DFTK.r_vectors_cart(basis))
        inclusion_value = zero(T)
        
        for pos in positions
            # Consider the main cell and neighboring cells
            if use_periodic_images
                for i1 in -1:1, i2 in -1:1, i3 in -1:1
                    offset = lattice * [i1, i2, i3]
                    r_relative = r - (pos + offset)
                    inclusion_value += inclusion_func(r_relative)
                end
            else
                r_relative = r - pos
                inclusion_value += inclusion_func(r_relative)
            end
        end
        
        a_values[i] += inclusion_value
    end
    
    a_values
end

"""
Build the full coefficient field a(x) from Fourier-space inclusion functions.

# Arguments
- `basis`: PlaneWaveBasis
- `inclusion_fourier_func`: Function that takes a G vector (Cartesian) and returns the Fourier coefficient
- `background_value`: Background constant value of a(x)
- `positions`: List of inclusion center positions
"""
function build_coefficient_field_fourier(basis, inclusion_fourier_func, background_value, positions)
    T = real(eltype(basis))
    
    # Build in Fourier space
    Gs = DFTK.G_vectors(basis)
    a_fourier = zeros(Complex{T}, basis.fft_size...)
    
    for (iG, G) in enumerate(Gs)
        G_cart = basis.model.recip_lattice * G
        p = norm(G_cart)
        
        if iG == 1  # G = 0
            # DC component: background + integral of inclusions
            value = background_value * basis.model.unit_cell_volume
            for pos in positions
                value += inclusion_fourier_func(G_cart) / sqrt(basis.model.unit_cell_volume)
            end
            a_fourier[DFTK.index_G_vectors(basis, G)] = value / sqrt(basis.model.unit_cell_volume)
        else
            # Structure factor: sum over positions
            value = zero(Complex{T})
            for pos in positions
                value += DFTK.cis2pi(-dot(G, pos)) * inclusion_fourier_func(G_cart)
            end
            a_fourier[DFTK.index_G_vectors(basis, G)] = value / sqrt(basis.model.unit_cell_volume)
        end
    end
    
    # Transform to real space
    DFTK.enforce_real!(a_fourier, basis)
    DFTK.irfft(basis, a_fourier)
end

#
# Constructors for TermDivAGrad from different representations
#

"""
Construct TermDivAGrad from a real-space inclusion function.

# Arguments
- `inclusion_real`: Function r::Vec3 -> value, defining the inclusion in real space
- `background_value`: Background constant value
- `positions`: List of inclusion center positions
"""
struct DivAGradFromReal
    inclusion_real::Function
    background_value::Any
    positions::Vector
end

function (divAgrad::DivAGradFromReal)(basis::DFTK.PlaneWaveBasis{T}) where {T}
    a_values = build_coefficient_field_real(basis, divAgrad.inclusion_real, 
                                             divAgrad.background_value, divAgrad.positions)
    # Note: DivAgradOperator implements -½∇⋅(A∇), we want -∇⋅(a∇), so A = 2a
    A_values = 2 .* a_values
    TermDivAGrad(A_values)
end

"""
Construct TermDivAGrad from a Fourier-space inclusion function.

# Arguments
- `inclusion_fourier`: Function G_cart::Vec3 -> value, defining the inclusion Fourier transform
- `background_value`: Background constant value
- `positions`: List of inclusion center positions
"""
struct DivAGradFromFourier
    inclusion_fourier::Function
    background_value::Any
    positions::Vector
end

function (divAgrad::DivAGradFromFourier)(basis::DFTK.PlaneWaveBasis{T}) where {T}
    a_values = build_coefficient_field_fourier(basis, divAgrad.inclusion_fourier,
                                                divAgrad.background_value, divAgrad.positions)
    # Note: DivAgradOperator implements -½∇⋅(A∇), we want -∇⋅(a∇), so A = 2a
    A_values = 2 .* a_values
    TermDivAGrad(A_values)
end

#
# Preconditioners
#

# Custom preconditioner type that implements pseudo-inverse (zeroing DC component)
struct PseudoInversePreconditioner{T}
    diag::Vector{T}
    zero_idx::Int
end

function LinearAlgebra.ldiv!(y, P::PseudoInversePreconditioner, x)
    # Pseudo-inverse: invert all diagonal elements except the DC component
    for i in eachindex(y)
        if i == P.zero_idx
            y[i] = 0  # Zero out the DC component (pseudo-inverse)
        else
            y[i] = x[i] / (P.diag[i] + 1)  # Add small shift for stability
        end
    end
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
    
    # Find the index of G=0 for the pseudo-inverse
    G_zero_idx = findfirst(G -> all(iszero, G), DFTK.G_vectors(basis, kpt))
    @assert !isnothing(G_zero_idx)
    
    # Simple diagonal preconditioner based on kinetic energy
    kin_energies = [T(DFTK.norm2(kpt.coordinate + G) / 2) for G in DFTK.G_vectors_cart(basis, kpt)]
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
# Example: Solve a sample problem with rectangular inclusions
#

# Setup a 2D lattice
a = 10.0  # Box size
lattice = a .* [[1 0 0.]; [0 1 0]; [0 0 0]]  # 2D system

# Define rectangular inclusion function
# Rectangle with half-widths (wx, wy, wz) and value a_inc
function rectangular_inclusion_real(wx, wy, wz, a_inc)
    return function(r)
        if abs(r[1]) <= wx && abs(r[2]) <= wy && abs(r[3]) <= wz
            return a_inc
        else
            return 0.0
        end
    end
end

# Fourier transform of rectangular inclusion
# FT[χ_rect](G) = a_inc * (2wx)(2wy)(2wz) * sinc(G_x*wx)*sinc(G_y*wy)*sinc(G_z*wz)
function rectangular_inclusion_fourier(wx, wy, wz, a_inc)
    return function(G)
        T = eltype(G)
        volume = 8 * wx * wy * wz
        
        # sinc(x) = sin(πx)/(πx), but we need sin(x)/x
        function my_sinc(x)
            return abs(x) < 1e-10 ? one(T) : sin(x) / x
        end
        
        return a_inc * volume * my_sinc(G[1] * wx) * my_sinc(G[2] * wy) * my_sinc(G[3] * wz)
    end
end

# Inclusion parameters
wx, wy, wz = 1.0, 0.8, 0.5  # Half-widths
a_inc = 3.0  # Inclusion value

# Positions of inclusions
positions = [[0.25 * a, 0.25 * a, 0.0], 
             [0.75 * a, 0.75 * a, 0.0],
             [0.25 * a, 0.75 * a, 0.0]]

# Background value of a(x)
background_a = 1.0

# Test both real and Fourier approaches
println("=== Testing with DivAGradFromReal ===")
terms_real = [DivAGradFromReal(rectangular_inclusion_real(wx, wy, wz, a_inc),
                                background_a, positions)]
model_real = DFTK.Model(lattice, Float64[], Vector{Float64}[]; n_electrons=0, terms=terms_real,
                        spin_polarization=:spinless)
Ecut = 50
basis_real = DFTK.PlaneWaveBasis(model_real; Ecut, kgrid=(1, 1, 1))

println("\n=== Testing with DivAGradFromFourier ===")
terms_fourier = [DivAGradFromFourier(rectangular_inclusion_fourier(wx, wy, wz, a_inc),
                                      background_a, positions)]
model_fourier = DFTK.Model(lattice, Float64[], Vector{Float64}[]; n_electrons=0, terms=terms_fourier,
                            spin_polarization=:spinless)
basis_fourier = DFTK.PlaneWaveBasis(model_fourier; Ecut, kgrid=(1, 1, 1))

# Get the coefficient fields for comparison
a_real = basis_real.terms[1].A_values ./ 2  # Divide by 2 since A = 2a
a_fourier = basis_fourier.terms[1].A_values ./ 2

println("Difference between real and Fourier constructions:")
println("  Max absolute difference: ", maximum(abs.(a_real .- a_fourier)))
println("  RMS difference: ", sqrt(sum(abs2.(a_real .- a_fourier)) / length(a_real)))

# Define the right-hand side f(x) = da/dx (derivative in x direction)
# Calculate this in Fourier domain for exactness
kpt = only(basis_fourier.kpoints)
a_fourier_coeff = DFTK.fft(basis_fourier, a_fourier)

# Derivative: multiply by i*G_x in Fourier space
f_fourier = similar(a_fourier_coeff)
for (iG, G) in enumerate(DFTK.G_vectors(basis_fourier, kpt))
    G_cart = basis_fourier.model.recip_lattice * G
    f_fourier[iG] = 2π * im * G_cart[1] * a_fourier_coeff[iG]
end

# Transform to real space
f_values = DFTK.ifft(basis_fourier, kpt, f_fourier)

# Ensure zero average (should already be satisfied, but enforce numerically)
f_values .-= sum(f_values) / length(f_values)

println("\nRight-hand side f = da/dx:")
println("  Min: ", minimum(real.(f_values)))
println("  Max: ", maximum(real.(f_values)))
println("  Average: ", sum(real.(f_values)) / length(f_values))

# Solve the problem
println("\n=== Solving -div(a(x)∇u(x)) = f(x) ===")
u, info = solve_linear_problem(basis_fourier, f_values; tol=1e-6, maxiter=200)
println("CG converged: $(info.converged) after $(info.n_iter) iterations")
println("Final residual: $(info.residual_norm)")

# Compute derivatives of u for visualization
u_fourier_coeff = DFTK.fft(basis_fourier, u)

# du/dx
dudx_fourier = similar(u_fourier_coeff)
for (iG, G) in enumerate(DFTK.G_vectors(basis_fourier, kpt))
    G_cart = basis_fourier.model.recip_lattice * G
    dudx_fourier[iG] = 2π * im * G_cart[1] * u_fourier_coeff[iG]
end
dudx = DFTK.ifft(basis_fourier, kpt, dudx_fourier)

# du/dy
dudy_fourier = similar(u_fourier_coeff)
for (iG, G) in enumerate(DFTK.G_vectors(basis_fourier, kpt))
    G_cart = basis_fourier.model.recip_lattice * G
    dudy_fourier[iG] = 2π * im * G_cart[2] * u_fourier_coeff[iG]
end
dudy = DFTK.ifft(basis_fourier, kpt, dudy_fourier)

# Visualize the results
r_vectors = DFTK.r_vectors_cart(basis_fourier)
x_coords = [r[1] for r in r_vectors]
y_coords = [r[2] for r in r_vectors]

# Reshape for plotting (2D)
nx, ny, nz = basis_fourier.fft_size
X = reshape(x_coords, nx, ny, nz)[:, :, 1]
Y = reshape(y_coords, nx, ny, nz)[:, :, 1]
A = reshape(real.(a_fourier), nx, ny, nz)[:, :, 1]
F = reshape(real.(f_values), nx, ny, nz)[:, :, 1]
U = reshape(real.(u), nx, ny, nz)[:, :, 1]
DUDx = reshape(real.(dudx), nx, ny, nz)[:, :, 1]
DUDy = reshape(real.(dudy), nx, ny, nz)[:, :, 1]

# Create plots
p1 = heatmap(X[:, 1], Y[1, :], A', title="Coefficient a(x)", 
             xlabel="x", ylabel="y", c=:viridis)
p2 = heatmap(X[:, 1], Y[1, :], F', title="RHS f(x) = da/dx", 
             xlabel="x", ylabel="y", c=:RdBu)
p3 = heatmap(X[:, 1], Y[1, :], U', title="Solution u(x)", 
             xlabel="x", ylabel="y", c=:plasma)
p4 = heatmap(X[:, 1], Y[1, :], DUDx', title="du/dx", 
             xlabel="x", ylabel="y", c=:viridis)
p5 = heatmap(X[:, 1], Y[1, :], DUDy', title="du/dy", 
             xlabel="x", ylabel="y", c=:viridis)

p = plot(p1, p2, p3, p4, p5, layout=(2, 3), size=(1800, 1200))
