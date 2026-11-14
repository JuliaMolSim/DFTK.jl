"""
Density-dependent kinetic energy from a libxc LDA, GGA, or meta-GGA kinetic functional,
such as `:gga_k_tfvw` for Thomas-Fermi plus von Weizsäcker.
"""
struct KineticDensityFunctional
    functional::Functional
    function KineticDensityFunctional(functional)
        fun = functional isa Functional ? functional : DispatchFunctional(functional)
        @assert kind(fun) == :k
        new(fun)
    end
end

Base.show(io::IO, kinetic::KineticDensityFunctional) =
    print(io, "KineticDensityFunctional($(kinetic.functional))")

function (kinetic::KineticDensityFunctional)(basis::PlaneWaveBasis{T}) where {T}
    TermDensityFunctional(Functional[kinetic.functional], one(T), zero(T), nothing, nothing)
end
