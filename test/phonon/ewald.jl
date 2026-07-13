@testmodule PhononEwald begin
using DFTK

function model_tested(lattice::AbstractMatrix, atoms::Vector{<:DFTK.Element},
                      positions::Vector{<:AbstractVector}; kwargs...)
    terms = [Kinetic(),
             Ewald()]
    if :temperature in keys(kwargs) && kwargs[:temperature] != 0
        terms = [terms..., Entropy()]
    end
    Model(lattice, atoms, positions; model_name="atomic", terms, kwargs...)
end
end

@testitem "Phonon: Ewald: comparison to ref testcase" #=
    =#    tags=[:phonon, :dont_test_mpi] setup=[Phonon, PhononEwald, TestCases] begin
    using DFTK
    using .Phonon: test_frequencies
    using .PhononEwald: model_tested

    ω_ref = [ -3.720615299046614e-12
               1.969314371029982e-11
               1.9739956911274832e-11
               0.00029302379784864935
               0.0002930237978486494
               0.000293023797851601
               0.0002930237978516018
               0.0005105451353059533
               0.0005105451353059533
               0.000510545135311239
               0.0005105451353112397
               0.0005676024288436319
               0.000591265950289604
               0.0005912659502958081
               0.0007328535013566558
               0.0007328535013566561
               0.0008109743140779055
               0.0008109743140779056
               0.000938673782810113
               0.000987619635716976
               0.0009876196357169761
               0.0010949497272589232
               0.0011998186659486743
               0.0011998186659486745
               0.001523238357971607
               0.0019593679918607546
               0.0022394777249719524
               0.0022394777249719524
               0.0024681196094789985
               0.0024681196094789993
               0.0024809296524054506
               0.0025805236057819345
               0.002614761988704579
               0.002614761988704579
               0.0026807773193116675
               0.0026807773193116675 ]
    test_frequencies(model_tested, TestCases.magnesium; ω_ref)
end

@testitem "Phonon: Ewald: comparison to automatic differentiation" #=
    =#    tags=[:phonon, :slow, :dont_test_mpi] setup=[Phonon, PhononEwald, TestCases] begin
    using DFTK
    using .Phonon: test_frequencies
    using .PhononEwald: model_tested

    test_frequencies(model_tested, TestCases.magnesium)
end
