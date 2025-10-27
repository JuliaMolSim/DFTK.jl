# This file provides wrappers around ForwardDiff standard differentiation calls.
# Custom Dual tags can be passed to ForwardDiff, enabling function specialization
# for specific ForwardDiff.Dual{T,V,N} types, where the tag T can be fixed.
# ForwardDiff's default behaviour is to generate a new tag for each differentiation,
# leading to a recompilation of all functions involved. This causes large compilation 
# times when multiple differentiations are performed. By using fixed tags, we can reuse
# compiled code specialized for numerical type V and number of partials N.
# Non-default custom types should be used for nested differentiations, in order to avoid
# perturbation confusion (a different tag or a standard FD call for each nested layer).

using ForwardDiff

struct DerivativeTag end
function tagged_derivative(f, x::T; custom_tag=DerivativeTag) where T
    # explicit call to ForwardDiff.Tag() to trigger ForwardDiff.tagcount
    TagType = typeof(ForwardDiff.Tag(custom_tag(), T))
    x_dual = ForwardDiff.Dual{TagType, T, 1}(x, ForwardDiff.Partials((one(T),)))

    res = ForwardDiff.extract_derivative(TagType, f(x_dual))
    return res
end

struct GradientTag end
GradientConfig(f, x, ::Type{Tag}) where {Tag} = 
    ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk(x), Tag())
function tagged_gradient(f, x::AbstractArray{T}; custom_tag=GradientTag) where T
    # explicit call to ForwardDiff.Tag() to trigger ForwardDiff.tagcount
    TagType = typeof(ForwardDiff.Tag(custom_tag(), T)) 
    
    cfg = GradientConfig(f, x, TagType) 
    ForwardDiff.gradient(f, x, cfg, Val{false}()) 
end

struct JacobianTag end
JacobianConfig(f, x, ::Type{Tag}) where {Tag} = 
    ForwardDiff.JacobianConfig(f, x, ForwardDiff.Chunk(x), Tag())
function tagged_jacobian(f, x::AbstractArray{T}; custom_tag=JacobianTag) where T
    # explicit call to ForwardDiff.Tag() to trigger ForwardDiff.tagcount
    TagType = typeof(ForwardDiff.Tag(custom_tag(), T))
    
    cfg = JacobianConfig(f, x, TagType) 
    ForwardDiff.jacobian(f, x, cfg, Val{false}()) 
end