# GPU workarounds for atomic grid integrations of the AtomicLocal term. GPU kernels can
# only take isbits data as input, and Upf elements are far from being isbits. Since only
# a limited number of operations ever become rate limiting, we simply rewrite
# those in a GPU optimized way here.
function atomic_local_inner_loop!(form_factors_cpu, norm_indices, igroup,
                        element::ElementPsp{<:PspUpf}, arch::GPU{AT}) where {AT}

    x = @view element.psp.rgrid[1:3]
    uniform_grid = (x[2] - x[1]) ≈ (x[3] - x[2]) ? true : false

    rgrid = to_device(arch, @view element.psp.rgrid[1:element.psp.ircut])
    vloc = to_device(arch, @view element.psp.vloc[1:element.psp.ircut])
    ps = to_device(arch, collect(keys(norm_indices)))
    Zion = element.psp.Zion

    ints = map(ps) do p
        T = eltype(p)
        method = uniform_grid ? simpson_uniform : simpson_nonuniform
        if p == 0
            zero(T)
        else
            # GPU compilation error if branching done within generic simpson() function
            I = method(rgrid) do i, r
                 r * (r * vloc[i] - -Zion * erf(r)) * sphericalbesselj_fast(0, p * r)
            end
            4T(π) * (I + -Zion / p^2 * exp(-p^2 / T(4)))
        end
    end

    ints_cpu = to_cpu(ints)
    for (p, I) in zip(keys(norm_indices), ints_cpu)
        form_factors_cpu[norm_indices[p], igroup] = I
    end
end