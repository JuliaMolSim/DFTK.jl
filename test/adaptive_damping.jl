using Test
using DFTK
import DFTK: AdaptiveDamping, ensure_damping_within_range

@testset "Damping adjustment in adaptive damping" begin
    damping = AdaptiveDamping(;α_min=0.05,
                               α_max=1.0,
                               α_trial_init=0.8,
                               α_trial_min=0.2,
                               α_trial_enhancement=1.1,
                               modeltol=0.1)

    # If within range we accept
    @test ensure_damping_within_range(damping,  0.2, 0.1) == 0.1
    @test ensure_damping_within_range(damping, -0.2, 0.1) == 0.1

    # If above maximum, than limit
    @test ensure_damping_within_range(damping,  1.5, 1.5) == 1.0
    @test ensure_damping_within_range(damping, -1.5, 1.5) == 1.0

    # Ensure shrinkage
    @test ensure_damping_within_range(damping,  0.2, 0.2) == 0.19
    @test ensure_damping_within_range(damping,  0.2, 0.5) == 0.19
    @test ensure_damping_within_range(damping, -0.2, 0.2) == 0.19
    @test ensure_damping_within_range(damping, -0.2, 0.5) == 0.19

    # ... but not below α_min
    @test ensure_damping_within_range(damping,  0.2, 0.01) == 0.05
    @test ensure_damping_within_range(damping, -0.2, 0.01) == 0.05

    # Ensure sign is kept
    @test ensure_damping_within_range(damping, -0.2, -0.07) == -0.07
    @test ensure_damping_within_range(damping,  0.2, -0.07) == -0.07
    @test ensure_damping_within_range(damping, -0.2, -0.2)  == -0.1
    @test ensure_damping_within_range(damping,  0.2, -0.2)  == -0.1
    @test ensure_damping_within_range(damping, -0.2, -0.5)  == -0.1
    @test ensure_damping_within_range(damping,  0.2, -0.5)  == -0.1

    # ... unless too small
    @test ensure_damping_within_range(damping, -0.2, -1e-3) == 0.05
    @test ensure_damping_within_range(damping,  0.2, -1e-3) == 0.05
end
