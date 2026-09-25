# Standalone CPU Ewald benchmark: 216 Al atoms (FCC primitive 6³, a = 4.05 Å).
# No SCF or pseudopotentials are needed. In a persistent Julia/Revise session:
#
# include("benchmark/ewald.jl")
# problem = ewald_bench_problem()
# ewald_bench_warm()  # compile Float64 methods on one atom
# plain = ewald_bench_time(DFTK.energy_forces_ewald, problem)
# ewald_bench_summary(plain)
# ewald_bench_warm(; direction=1)  # compile one-partial Dual methods
# strain = ewald_bench_time(DFTK.energy_forces_ewald, problem; direction=1)
# ewald_bench_summary(strain)
#
# direction=0 selects Float64; directions 1:6 select one Voigt strain component,
# matching compute_stresses_cart's Chunk{1}(). Run each timing separately with a
# timeout of 60 s. Repeat timings if needed. Six separate direction records can be
# combined with ewald_bench_stress_cost; the summary's 6× time is only an estimate.
#
# Optional comparison with the implementation before PR #1387 and that PR:
# git show 5a94a1407:src/terms/ewald.jl > /tmp/ewald-master.jl
# gh api -H 'Accept: application/vnd.github.raw' \
#   'repos/JuliaMolSim/DFTK.jl/contents/src/terms/ewald.jl?ref=665d82428a2305aac589b832c4f9482c6d804ddd' \
#   > /tmp/ewald-pr.jl
#
# implementations = ewald_bench_implementations("/tmp/ewald-master.jl", "/tmp/ewald-pr.jl")
# ewald_bench_warm(implementations.master; direction=1)
# old = ewald_bench_time(implementations.master, problem; direction=1)
# ewald_bench_compare(old, strain)
# Repeat for implementations.pr. Use ewald_bench_problem(; rattle=0.03) to compare
# nonzero forces on perturbed positions as well.

using DFTK
using LinearAlgebra
using ForwardDiff
using Unitful
using UnitfulAtomic

function ewald_bench_reference(path)
    # Load only the energy routines, leaving DFTK's types and methods untouched.
    source = read(path, String)
    first_function = findfirst("function default_η(", source)
    last_function = findfirst("# TODO: See if there is a way to express this with AD.", source)
    isnothing(first_function) && error("Cannot find default_η in $path")
    isnothing(last_function) && error("Cannot find end of Ewald energy functions in $path")
    source = source[first(first_function):prevind(source, first(last_function))]
    reference = Module(gensym(:EwaldReference))
    Core.eval(reference, quote
        using LinearAlgebra
        import ForwardDiff
        import SpecialFunctions: erfc
        using DFTK: Vec3, norm2, norm_cplx, cos2pi, sin2pi, cis2pi,
                    compute_recip_lattice, compute_inverse_lattice, compute_unit_cell_volume,
                    estimate_integer_lattice_bounds
    end)
    Base.include_string(reference, source, path)
    Base.invokelatest(getfield, reference, :energy_forces_ewald)
end

function ewald_bench_implementations(master_path, pr_path)
    (; master=ewald_bench_reference(master_path),
       pr=ewald_bench_reference(pr_path), current=DFTK.energy_forces_ewald)
end

function ewald_bench_problem(n=6; rattle=0.0)
    a = austrip(4.05u"Å")
    lattice = n * a / 2 * DFTK.Mat3{Float64}(0, 1, 1, 1, 0, 1, 1, 1, 0)
    positions = [DFTK.Vec3{Float64}(i, j, k) / n
                 for i = 0:n-1 for j = 0:n-1 for k = 0:n-1]
    if !iszero(rattle)
        positions = [r + rattle / n * DFTK.Vec3(sin(i), cos(2i), sin(3i))
                     for (i, r) in enumerate(positions)]
    end
    (; lattice, positions, charges=fill(3, length(positions)))
end

function ewald_bench_input(problem; direction=0)
    @assert 0 ≤ direction ≤ 6
    direction == 0 && return problem
    tag = typeof(ForwardDiff.Tag(ewald_bench_input, Float64))
    strain = [ForwardDiff.Dual{tag}(0.0, Float64(i == direction)) for i = 1:6]
    lattice = DFTK.voigt_strain_to_full(strain) * problem.lattice
    T = eltype(lattice)
    # Model(lattice, ...) also promotes reduced positions to the lattice scalar type.
    positions = DFTK.Vec3{T}.(problem.positions)
    (; lattice, positions, charges=problem.charges)
end

function ewald_bench_call(f, input)
    T = eltype(input.lattice)
    f(T, input.lattice, input.charges, input.positions, zero(DFTK.Vec3{T}), nothing)
end

function ewald_bench_warm(f=DFTK.energy_forces_ewald; direction=0)
    input = ewald_bench_input(ewald_bench_problem(1); direction)
    elapsed = @elapsed ewald_bench_call(f, input)
    (; direction, warmup_s=elapsed)
end

function ewald_bench_time(f=DFTK.energy_forces_ewald, problem=ewald_bench_problem();
                          direction=0)
    input = ewald_bench_input(problem; direction)
    GC.gc()
    timed = @timed ewald_bench_call(f, input)
    (; direction, n_atoms=length(input.positions), seconds=timed.time,
       bytes=timed.bytes, gc_seconds=timed.gctime, result=timed.value)
end

function ewald_bench_summary(record)
    (; record.direction, record.n_atoms, record.seconds, record.bytes,
       record.gc_seconds, energy=ForwardDiff.value(record.result.energy),
       six_direction_estimate_s=record.direction == 0 ? nothing : 6 * record.seconds)
end

function ewald_bench_compare(reference, candidate)
    @assert reference.direction == candidate.direction
    @assert reference.n_atoms == candidate.n_atoms
    ref = reference.result
    new = candidate.result
    energy_error = abs(ForwardDiff.value(new.energy) - ForwardDiff.value(ref.energy))
    forces_error = maximum(norm(ForwardDiff.value.(fnew) - ForwardDiff.value.(fref))
                           for (fref, fnew) in zip(ref.forces, new.forces))
    derivative_energy_error = derivative_forces_error = nothing
    if reference.direction != 0
        partial(x) = ForwardDiff.partials(x)[1]
        derivative_energy_error = abs(partial(new.energy) - partial(ref.energy))
        derivative_forces_error = maximum(norm(partial.(fnew) - partial.(fref))
                                          for (fref, fnew) in zip(ref.forces, new.forces))
    end
    (; speedup=reference.seconds / candidate.seconds, energy_error, forces_error,
       derivative_energy_error, derivative_forces_error)
end

function ewald_bench_stress_cost(records)
    @assert sort([record.direction for record in records]) == collect(1:6)
    (; seconds=sum(record.seconds for record in records),
       bytes=sum(record.bytes for record in records),
       gc_seconds=sum(record.gc_seconds for record in records))
end
