@testitem "Helium exchange energy" setup=[TestCases] begin
    using DFTK
    using LinearAlgebra
    silicon = TestCases.silicon

    lattice   = 10diagm(ones(3))
    positions = [0.5ones(3)]
    atoms     = [ElementCoulomb(:He)]
    model     = model_atomic(lattice, atoms, positions)
    basis     = PlaneWaveBasis(model; Ecut=15, kgrid=(1, 1, 1))
    scfres    = self_consistent_field(basis)

    Eh, oph = DFTK.ene_ops(Hartree()(basis), basis, scfres.ψ, scfres.occupation; scfres...)
    Ex, opx = DFTK.ene_ops(ExactExchange()(basis), basis, scfres.ψ, scfres.occupation; scfres...)
    @test abs(Ex + Eh) < 1e-12

    mat_h = real.(Matrix(only(oph)))
    mat_x = real.(Matrix(only(opx)))
    @show maximum(abs, mat_x - mat_x')
end
