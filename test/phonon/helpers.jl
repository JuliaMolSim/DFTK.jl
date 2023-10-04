using Random
using ForwardDiff
using LinearAlgebra

# Helpers functions for tests.
# TODO: Temporary, explanations too scarce. To be changed with proper phonon computations.

# Convert back and forth between Vec3 and columnwise matrix
fold(x)   = hcat(x...)
unfold(x) = Vec3.(eachcol(x))

function prepare_system(; n_scell=1)
    positions = [Vec3([0.0, 0.0, 0.0])]
    for i in 1:n_scell-1
        push!(positions, Vec3(i * ones(3) / n_scell))
    end

    a = 5. * length(positions)
    lattice = a * [[1 0 0.]; [0 0 0.]; [0 0 0.]]

    Linv = DFTK.compute_inverse_lattice(lattice)
    directions = [[Linv * [i==j, 0.0, 0.0] for i in 1:n_scell] for j in 1:n_scell]

    params = Dict((:X, :X) => (; ε=1, σ=a / length(positions) /2^(1/6)))
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)

    (; positions, lattice, directions, params, V)
end

function prepare_3d_system(; n_scell=1)
    positions = [[0.0, 0.0, 0.0]]
    for i in 1:n_scell-1
        push!(positions, i * ones(3) / n_scell)
    end

    a = 5. * length(positions)
    lattice = a * rand(3, 3)

    (; positions, lattice)
end

# Compute phonons for a one-dimensional pairwise potential for a set of `q = 0` using
# supercell method
function test_supercell_q0(; n_scell=1, max_radius=1e3)
    case = prepare_system(; n_scell)
    Linv  = DFTK.compute_inverse_lattice(case.lattice)
    n_atoms = length(case.positions)

    directions = [reshape(vcat([[i==j, 0.0, 0.0] for i in 1:n_atoms]...), 3, :)
                  for j in 1:n_atoms]

    Φ = zeros(length(directions), n_atoms)
    for (i, direction) in enumerate(directions)
        Φ[i, :] = - ForwardDiff.derivative(0.0) do ε
            new_positions = unfold(fold(case.positions) .+ ε .* Linv * direction)
            forces = energy_forces_pairwise(eltype(ε).(case.lattice), fill(:X, n_atoms),
                                            new_positions, case.V, case.params;
                                            compute_forces=true, max_radius).forces
            [(Linv * f)[1] for f in forces]
        end
    end
    sqrt.(abs.(eigvals(Φ)))
end

# Compute phonons for a one-dimensional pairwise potential for a set of `q`-points
function test_ph_disp(; n_scell=1, max_radius=1e3, n_points=2)
    case = prepare_system(; n_scell)
    Linv  = DFTK.compute_inverse_lattice(case.lattice)
    n_atoms = length(case.positions)

    function pairwise_ph(q, d)
        energy_forces_pairwise(case.lattice, fill(:X, n_atoms), case.positions, case.V,
                               case.params; q=[q, 0, 0], ph_disp=d, compute_forces=true,
                               max_radius).forces
    end

    ph_bands = []
    for q in range(-1/2, 1/2, length=n_points+1)
        as = ComplexF64[]
        for d in case.directions
            res = -ForwardDiff.derivative(0.0) do ε
                forces = pairwise_ph(q, ε*d)
                [Linv' * f for f in forces]
            end
            [push!(as, r[1]) for r in res]
        end
        M = reshape(as, n_atoms, :)
        @assert norm(imag.(eigvals(M))) < 1e-8
        push!(ph_bands, sqrt.(abs.(real(eigvals(M)))))
    end
    ph_bands
end

"""
Real-space equivalent of `transfer_blochwave_kpt`.
"""
function transfer_blochwave_kpt_real(ψk_in, basis::PlaneWaveBasis, kpt_in, kpt_out, ΔG)
    ψk_out = zeros(eltype(ψk_in), length(kpt_out.G_vectors), size(ψk_in, 2))
    exp_ΔGr = DFTK.cis2pi.(-dot.(Ref(ΔG), r_vectors(basis)))
    for n in 1:size(ψk_in, 2)
        ψk_out[:, n] = fft(basis, kpt_out, exp_ΔGr .* ifft(basis, kpt_in, ψk_in[:, n]))
    end
    ψk_out
end

# Convert to Cartesian a dynamical matrix in reduced coordinates.
function dynmat_to_cart(basis, dynamical_matrix)
    model = basis.model
    positions = model.positions
    n_atoms = length(positions)
    lattice = model.lattice
    inv_lattice = DFTK.compute_inverse_lattice(lattice)

    cart_mat = zero.(dynamical_matrix)
    # The dynamical matrix `D` acts on vectors `dr` and gives a covector `dF`:
    #   dF = D · dr.
    # Thus the transformation between reduced and Cartesian coordinates is not a comatrix.
    # To transform `dynamical_matrix` from reduced coordinates to `cart_mat` in Cartesian
    # coordinates, we write
    #   dr_cart = lattice · dr_red,
    #   ⇒ dr_redᵀ · D_red · dr_red = dr_cartᵀ · lattice⁻ᵀ · D_red · lattice⁻¹ · dr_cart
    #                              = dr_cartᵀ · D_cart · dr_cart
    #   ⇒ D_cart = lattice⁻ᵀ · D_red · lattice⁻¹.
    for τ in 1:n_atoms
        for η in 1:n_atoms
            cart_mat[:, η, :, τ] = inv_lattice' * dynamical_matrix[:, η, :, τ] * inv_lattice
        end
    end
    reshape(cart_mat, 3*n_atoms, 3*n_atoms)
end

# We do not take the square root to compare results with machine precision.
function compute_ω²(matrix)
    Ω = eigvals(matrix)
    real(Ω)
end

function generate_random_supercell(; max_length=6)
    Random.seed!()
    n_max = min(max_length, 5)
    supercell_size = nothing
    while true
        supercell_size = rand(1:n_max, 3)
        prod(supercell_size) < max_length && break
    end
    supercell_size
end

function generate_supercell_qpoints(; supercell_size=generate_random_supercell())
    qpoints_list = Iterators.product([1:n_sc for n_sc in supercell_size]...)
    qpoints = map(qpoints_list) do n_sc
        DFTK.normalize_kpoint_coordinate.([n_sc[i] / supercell_size[i] for i in 1:3])
    end |> vec

    (; supercell_size, qpoints)
end

const REFERENCE_PH_CALC = "REFERENCE PH_CALC" in keys(ENV)
phonon = (
    supercell_size = REFERENCE_PH_CALC ? generate_random_supercell() : [2, 1, 3],
)
phonon = merge(phonon, (; generate_supercell_qpoints(; phonon.supercell_size).qpoints))
