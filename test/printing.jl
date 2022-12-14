using DFTK
using Test
include("testcases.jl")

# Quick and dirty testset that runs the Base.show code of the most important
# data structures of DFTK. Point is not to test that the correct thing is printed,
# rather to ensure that the code does not randomly stop working.
@testset "Test printing" begin
    for εF in (nothing, 0.5)
        model  = model_LDA(magnesium.lattice, magnesium.atoms, magnesium.positions;
                           magnesium.temperature, εF, disable_electrostatics_check=true)
        basis  = PlaneWaveBasis(model; Ecut=5, kgrid=[1, 3, 2], kshift=[0, 0, 0])
        scfres = self_consistent_field(basis; nbandsalg=FixedBands(; n_bands_converge=6),
                                       tol=1e-3)

        println(model)
        display("text/plain", model)

        println(basis)
        display("text/plain", basis)

        println(basis.kpoints[1])
        display("text/plain", basis.kpoints[1])

        println(scfres.energies)
        display("text/plain", scfres.energies)
    end
end
