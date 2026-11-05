"""
Density-dependent kinetic energy from libxc LDA, GGA, and meta-GGA functionals, such as
`[:lda_k_tf, :gga_k_vw]` for Thomas-Fermi plus von Weizsäcker.
"""
struct KineticDensityFunctional
    functionals::Vector{Functional}
    scaling_factor::Real
end
function KineticDensityFunctional(functionals::AbstractVector; scaling_factor=1)
    fun = map(f -> f isa Functional ? f : DispatchFunctional(f), functionals)
    @assert !isempty(fun)
    @assert all(f -> kind(f) == :k, fun)
    KineticDensityFunctional(convert(Vector{Functional}, fun), scaling_factor)
end

function Base.show(io::IO, kinetic::KineticDensityFunctional)
    fac = isone(kinetic.scaling_factor) ? "" : ", scaling_factor=$(kinetic.scaling_factor)"
    fun = join(kinetic.functionals, ", ")
    print(io, "KineticDensityFunctional($fun$fac)")
end

function (kinetic::KineticDensityFunctional)(basis::PlaneWaveBasis)
    @assert basis.model.n_spin_components == 1
    # Reuse the existing functional evaluator, on valence density only.
    Xc(kinetic.functionals; scaling_factor=kinetic.scaling_factor, use_nlcc=false)(basis)
end
