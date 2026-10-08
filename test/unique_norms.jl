@testitem "unique_norms_and_mapping" begin
    using DFTK
    using LinearAlgebra
    using Random

    # Various G-vectors with distinct values and 12 distinct norms
    Gs = Vec3{Int}[
        (0, 0, 0),                                          # norm 0
        (1, 0, 0), (0, 1, 0),                               # norm 1
        (1, 1, 0), (1, -1, 0), (-1, 1, 0),                  # norm √2
        (1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1),      # norm √3
        (2, 0, 0),                                          # norm 2
        (2, 1, 0), (-2, 1, 0), (1, 2, 0), (-1, 2, 0),       # norm √5
        (2, 1, 1), (-2, 1, 1),                              # norm √6
        (2, 2, 0), (-2, 2, 0), (2, 0, 2),                   # norm √8
        (3, 0, 0), (0, 3, 0), (0, 0, -3),                   # norm 3
        (3, 1, 0), (-3, 1, 0), (1, 3, 0), (-1, 3, 0),       # norm √10
        (3, 1, 1), (-3, 1, 1), (3, -1, 1),                  # norm √11
        (2, 2, 2), (-2, 2, 2)                               # norm √12
    ]

    ref_unique_ps = Float64[0, 1, sqrt(2), sqrt(3), 2, sqrt(5), sqrt(6),
                            sqrt(8), 3, sqrt(10), sqrt(11), sqrt(12)]
    ref_iG2ifnorm = [
        1,                                                  # norm 0
        2, 2,                                               # norm 1
        3, 3, 3,                                            # norm √2
        4, 4, 4, 4,                                         # norm √3
        5,                                                  # norm 2
        6, 6, 6, 6,                                         # norm √5
        7, 7,                                               # norm √6
        8, 8, 8,                                            # norm √8
        9, 9, 9,                                            # norm 3
        10, 10, 10, 10,                                     # norm √10
        11, 11, 11,                                         # norm √11
        12, 12                                              # norm √12
    ]

    (; unique_ps, iG2ifnorm) = DFTK.unique_norms_and_mapping(Gs)

    @test unique_ps ≈ ref_unique_ps
    @test iG2ifnorm == ref_iG2ifnorm

    # Check the desired property of the mapping directly
    for i = 1:length(Gs)
        @test norm(Gs[i]) ≈ unique_ps[iG2ifnorm[i]]
    end

    # Same test on a shuffled version of the input
    perm = randperm(length(Gs))
    shuffled_Gs = Gs[perm]
    shuffled_ref_iG2ifnorm = ref_iG2ifnorm[perm]

    (; unique_ps, iG2ifnorm) = DFTK.unique_norms_and_mapping(shuffled_Gs)

    @test unique_ps ≈ ref_unique_ps
    @test iG2ifnorm == shuffled_ref_iG2ifnorm

    for i = 1:length(shuffled_Gs)
        @test norm(shuffled_Gs[i]) ≈ unique_ps[iG2ifnorm[i]]
    end
end
