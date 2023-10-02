using Test
using DFTK
using LinearAlgebra
using Random

include("helpers.jl")

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

@testset "Phonon: Shifting functions" begin
    Random.seed!()
    tol = 1e-12

    positions = [[0.0, 0.0, 0.0]]
    n_scell = 2
    for i in 1:n_scell-1
        push!(positions, i * ones(3) / n_scell)
    end
    n_atoms = length(positions)

    lattice = 5 * n_atoms * rand(3, 3)

    X = ElementGaussian(1.0, 0.5, :X)
    atoms = [X for _ in positions]

    model = Model(lattice, atoms, positions; n_electrons=n_atoms,
                  symmetries=false, spin_polarization=:collinear)
    kgrid = rand(2:10, 3)
    k1, k2, k3 = kgrid
    basis = PlaneWaveBasis(model; Ecut=100, kgrid)

    # We consider a smooth periodic function with Fourier coefficients given if the basis
    # e^(iG·x)
    ψ = rand(ComplexF64, size(r_vectors(basis)))

    # Random `q` shift
    q0 = rand(basis.kpoints).coordinate
    ishift = [rand(-k1*2:k1*2), rand(-k2*2:k2*2), rand(-k3*2:k3*2)]
    q = Vec3(q0 .* ishift)
    @testset "Transfer function" begin
        for kpt in unique(rand(basis.kpoints, 4))
            ψk = fft(basis, kpt, ψ)

            ψk_out_four = DFTK.multiply_by_expiqr(basis, kpt, q, ψk)
            ψk_out_real = let
                shifted_kcoord = kpt.coordinate .+ q
                index, ΔG = DFTK.find_equivalent_kpt(basis, shifted_kcoord, kpt.spin)
                kpt_out = basis.kpoints[index]
                transfer_blochwave_kpt_real(ψk, basis, kpt, kpt_out, ΔG)
            end
            @testset "Testing kpoint $(kpt.coordinate) on kgrid $kgrid" begin
                @test norm(ψk_out_four - ψk_out_real) < tol
            end
        end
    end

    @testset "Ordering function" begin
        kpoints_plus_q = DFTK.k_to_kpq_mapping(basis, q)
        ordering(kdata) = kdata[kpoints_plus_q]
        kcoords = getfield.(basis.kpoints, :coordinate)
        for (ik, kcoord) in enumerate(kcoords)
            @test mod.(kcoord + q .- tol, 1) ≈ mod.(ordering(kcoords)[ik] .- tol, 1)
        end
    end
end
