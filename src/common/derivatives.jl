# Helper: derivatives of a position-dependent function `f`. First-order derivatives for a
# displacement of the atom `s` in the direction `α`.
function derivative_wrt_αs(f, positions::AbstractVector{Vec3{T}}, α, s) where {T}
    displacement = zero.(positions)
    displacement[s] = setindex(displacement[s], one(T), α)
    ForwardDiff.derivative(zero(T)) do ε
        f(ε*displacement .+ positions)
    end
end
