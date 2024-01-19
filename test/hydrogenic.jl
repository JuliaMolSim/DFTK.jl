@testitem "Type-stability of radial_hydrogenic" begin
    using DFTK: radial_hydrogenic

    for n = [1, 2, 3]
        @inferred radial_hydrogenic([1.0, 1.0], n)
        @inferred radial_hydrogenic([1f0, 1f0], n)
    end
end