using DFTK
using Test
include("testcases.jl")

# Quick and dirty testset that runs the Base.show code of the most important
# data structures of DFTK. Point is not to test that the correct thing is printed,
# rather to ensure that the code does not randomly stop working.
@testset "Test printing" begin
    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
    atoms  = [Si => silicon.positions]
    model  = model_LDA(silicon.lattice, atoms)
    basis  = PlaneWaveBasis(model; Ecut=5, kgrid=[1, 3, 2], kshift=[0, 0, 0])
    scfres = self_consistent_field(basis, tol=1e-2)

    println(model)
    display("text/plain", model)

    println(basis)
    display("text/plain", basis)

    println(scfres.energies)
    display("text/plain", scfres.energies)
end
