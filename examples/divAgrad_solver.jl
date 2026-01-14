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

# Term for -∇⋅(a(x)∇) operator
struct TermDivAGrad{AT} <: DFTK.TermLinear
    A_values::AT  # Coefficient field a(x) on real-space grid
end

@DFTK.timing "ene_ops: divAgrad" function DFTK.ene_ops(term::TermDivAGrad,
                                                        basis::DFTK.PlaneWaveBasis{T}, 
                                                        ψ, occupation; kwargs...) where {T}
    # DivAgradOperator implements -½∇⋅(A∇), so use A = 2a to get -∇⋅(a∇)
    ops = [DFTK.DivAgradOperator(basis, kpt, 2 .* term.A_values) for kpt in basis.kpoints]
    (; E=T(Inf), ops)
end

# Build coefficient field a(x) from inclusions in real space
function build_coefficient_field_real(basis, inclusion_func, background_value, positions)
    T = real(eltype(basis))
    a_values = fill(T(background_value), basis.fft_size...)
    lattice = basis.model.lattice
    
    for (i, r) in enumerate(DFTK.r_vectors_cart(basis))
        inclusion_value = zero(T)
        for pos in positions, i1 in -1:1, i2 in -1:1, i3 in -1:1
            offset = lattice * [i1, i2, i3]
            inclusion_value += inclusion_func(r - (pos + offset))
        end
        a_values[i] += inclusion_value
    end
    a_values
end

# Build coefficient field a(x) from inclusions in Fourier space
function build_coefficient_field_fourier(basis, inclusion_fourier_func, background_value, positions)
    T = real(eltype(basis))
    a_fourier = zeros(Complex{T}, basis.fft_size...)
    
    for (iG, G) in enumerate(DFTK.G_vectors(basis))
        G_cart = basis.model.recip_lattice * G
        value = iG == 1 ? background_value * basis.model.unit_cell_volume : zero(Complex{T})
        for pos in positions
            value += DFTK.cis(-dot(G_cart, pos)) * inclusion_fourier_func(G_cart)
        end
        a_fourier[iG] = value / sqrt(basis.model.unit_cell_volume)
    end
    
    DFTK.enforce_real!(a_fourier, basis)
    DFTK.irfft(basis, a_fourier)
end

# Constructors for TermDivAGrad from real or Fourier space
struct DivAGradFromReal
    inclusion_real::Function
    background_value::Any
    positions::Vector
end
(divAgrad::DivAGradFromReal)(basis::DFTK.PlaneWaveBasis) =
    TermDivAGrad(build_coefficient_field_real(basis, divAgrad.inclusion_real, 
                                              divAgrad.background_value, divAgrad.positions))

struct DivAGradFromFourier
    inclusion_fourier::Function
    background_value::Any
    positions::Vector
end
(divAgrad::DivAGradFromFourier)(basis::DFTK.PlaneWaveBasis) =
    TermDivAGrad(build_coefficient_field_fourier(basis, divAgrad.inclusion_fourier,
                                                 divAgrad.background_value, divAgrad.positions))

# Pseudo-inverse preconditioner (zero DC component)
struct PseudoInversePreconditioner{T}
    diag::Vector{T}
    zero_idx::Int
end
function LinearAlgebra.ldiv!(y, P::PseudoInversePreconditioner, x)
    for i in eachindex(y)
        y[i] = i == P.zero_idx ? 0 : x[i] / P.diag[i]
    end
    y
end

# Solve -div(a(x)∇u(x)) = f(x) using CG iteration
function solve_linear_problem(basis, f; tol=1e-6, maxiter=1000)
    kpt = only(basis.kpoints)
    T = real(eltype(basis))
    
    # Build Hamiltonian (represents -div(a∇) operator)
    ψ_dummy = [DFTK.random_orbitals(basis, kpt, 1) for kpt in basis.kpoints]
    occupation_dummy = [fill(1.0, 1) for _ in basis.kpoints]
    hamk = DFTK.energy_hamiltonian(basis, ψ_dummy, occupation_dummy).ham.blocks[1]
    
    # Linear map for CG
    n_G = length(DFTK.G_vectors(basis, kpt))
    H_map = LinearMap{Complex{T}}((y, x) -> LinearAlgebra.mul!(y, hamk, x), 
                                   n_G, n_G; ishermitian=true, isposdef=false)
    
    # Preconditioner and projection (zero DC component)
    G_zero_idx = findfirst(G -> all(iszero, G), DFTK.G_vectors(basis, kpt))
    @assert !isnothing(G_zero_idx)
    kin_energies = [T(DFTK.norm2(kpt.coordinate + G) / 2) for G in DFTK.G_vectors_cart(basis, kpt)]
    P = PseudoInversePreconditioner(kin_energies, G_zero_idx)
    proj(x) = (y = copy(x); y[G_zero_idx] = 0; y)
    
    # Solve in Fourier space
    b = proj(DFTK.fft(basis, kpt, f))
    u_fourier = zeros(Complex{T}, n_G)
    info = DFTK.cg!(u_fourier, H_map, b; precon=P, proj, tol, maxiter)
    
    DFTK.ifft(basis, kpt, u_fourier), info
end

# Rectangular inclusion functions
rectangular_inclusion_real(wx, wy, wz, a_inc) = 
    r -> (abs(r[1]) <= wx && abs(r[2]) <= wy && abs(r[3]) <= wz) ? a_inc : 0.0

rectangular_inclusion_fourier(wx, wy, wz, a_inc) = function(G)
    T = eltype(G)
    my_sinc(x) = abs(x) < 1e-10 ? one(T) : sin(x) / x
    8 * wx * wy * wz * a_inc * my_sinc(G[1] * wx) * my_sinc(G[2] * wy) * my_sinc(G[3] * wz)
end

# Setup problem: 2D lattice with rectangular inclusion
a = 10.0
lattice = a .* [[1 0 0.]; [0 1 0]; [0 0 0]]
wx, wy, wz = 1, 1, 1.0
a_inc = 3.0
positions = [a .* [0.5, 0.5, 0.0]]
background_a = 1.0

terms = [DivAGradFromFourier(rectangular_inclusion_fourier(wx, wy, wz, a_inc),
                              background_a, positions)]
model = DFTK.Model(lattice; n_electrons=0, terms, spin_polarization=:spinless)
basis = DFTK.PlaneWaveBasis(model; Ecut=200, kgrid=(1, 1, 1))

# Build RHS: f(x) = da/dx
kpt = only(basis.kpoints)
a_real = basis.terms[1].A_values
a_fourier = fft(basis, kpt, complex.(a_real))

# Compute ∂a/∂x in Fourier space
f = [2π * im * (basis.model.recip_lattice * G)[1] * a_fourier[iG]
     for (iG, G) in enumerate(DFTK.G_vectors(basis, kpt))]
f_values = DFTK.ifft(basis, kpt, f)
f_values .-= sum(f_values) / length(f_values)  # Ensure zero average

# Solve and compute derivatives
println("\n=== Solving -div(a(x)∇u(x)) = f(x) ===")
u, info = solve_linear_problem(basis, f_values; tol=1e-6, maxiter=1000)
println("CG converged: $(info.converged) after $(info.n_iter) iterations")
println("Final residual: $(info.residual_norm)")

# Compute ∂u/∂x and ∂u/∂y
u_fourier = DFTK.fft(basis, kpt, u)
compute_derivative(u_f, dir) = DFTK.ifft(basis, kpt,
    [2π * im * (basis.model.recip_lattice * G)[dir] * u_f[iG]
     for (iG, G) in enumerate(DFTK.G_vectors(basis, kpt))])
dudx = compute_derivative(u_fourier, 1)
dudy = compute_derivative(u_fourier, 2)

# Visualize results
nx, ny, nz = basis.fft_size
reshape_2d(v) = reshape(real.(v), nx, ny, nz)[:, :, 1]

r_vectors = DFTK.r_vectors_cart(basis)
X = reshape([r[1] for r in r_vectors], nx, ny, nz)[:, :, 1]
Y = reshape([r[2] for r in r_vectors], nx, ny, nz)[:, :, 1]

A, F, U = reshape_2d.((a_real, f_values, u))
DUDx, DUDy = reshape_2d.((dudx, dudy))

p1 = heatmap(X[:, 1], Y[1, :], A', title="a(x)", xlabel="x", ylabel="y", c=:viridis)
p2 = heatmap(X[:, 1], Y[1, :], F', title="f(x) = ∂a/∂x", xlabel="x", ylabel="y", c=:RdBu)
p3 = heatmap(X[:, 1], Y[1, :], U', title="u(x)", xlabel="x", ylabel="y", c=:plasma)
p4 = heatmap(X[:, 1], Y[1, :], DUDx', title="∂u/∂x", xlabel="x", ylabel="y", c=:viridis)
p5 = heatmap(X[:, 1], Y[1, :], DUDy', title="∂u/∂y", xlabel="x", ylabel="y", c=:viridis)

plot(p1, p2, p3, p4, p5, layout=(2, 3), size=(1800, 1200))
