using Test
using DFTK
using IntervalArithmetic
using GenericLinearAlgebra

include("testcases.jl")

function discretized_hamiltonian(T, testcase)
    Ecut = 10  # Hartree

    spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
    atoms = [spec => testcase.positions]
    # disable symmetry for interval
    model = model_DFT(Array{T}(testcase.lattice), atoms, [:lda_x, :lda_c_vwn])

    # For interval arithmetic to give useful numbers,
    # the fft_size should be a power of 2
    fft_size = nextpow.(2, compute_fft_size(model, Ecut))
    basis = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1), fft_size=fft_size)

    Hamiltonian(basis; ρ=guess_density(basis))
end

@testset "Application of an LDA Hamiltonian with Intervals" begin
    T = Float64
    ham = discretized_hamiltonian(T, silicon)
    hamInt = discretized_hamiltonian(Interval{T}, silicon)
    @test length(ham.basis.model.symmetries) == length(hamInt.basis.model.symmetries)

    hamk = ham.blocks[1]
    hamIntk = hamInt.blocks[1]

    x = randn(Complex{T}, length(G_vectors(ham.basis, ham.basis.kpoints[1])))
    ref = hamk * x
    res = hamIntk * Interval.(x)

    # Small difference between interval arithmetic and normal application
    @test maximum(mid, abs.(res .- ref)) < 1e-9

    # Small error determined by interval arithmetic
    @test maximum(radius, abs.(res)) < 1e-9
end
