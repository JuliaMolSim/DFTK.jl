using Test
using DFTK

@testset "Comparison of diagonalisaton procedures" begin
    function test_solver(reference, eigensolver, prec_type)
        nev = length(reference.λ[1])
        println("Running $eigensolver with $prec_type ...")
        res = diagonalize_all_kblocks(eigensolver, reference.ham, nev, prec_type=prec_type)
        @test res.λ ≈ reference.λ
    end

    Ecut = 100
    lattice = Float64[5 0 0; 0 0 0; 0 0 0]
    model = Model(lattice, n_electrons=4, terms=[Kinetic()])
    basis = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1))
    ham = Hamiltonian(basis)
    reference = merge(diagonalize_all_kblocks(diag_full, ham, 6), (ham=ham,))

    test_solver(reference, diag_full, PreconditionerTPA)
    test_solver(reference, diag_full, PreconditionerNone)
    test_solver(reference, lobpcg_hyper, PreconditionerTPA)
end
