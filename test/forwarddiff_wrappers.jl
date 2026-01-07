@testmodule ForwardDiffWrappers begin
using ForwardDiff

# Wrappers around ForwardDiff to fix tags and reduce compilation time.
# Warning: fixing types without care can lead to perturbation confusion,
# this should only be used within the testing framework. Risk of perturbation
# confusion arises when nested derivatives of different functions are taken
# with a fixed tag. Only use these wrappers at the top-level call.
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
end
