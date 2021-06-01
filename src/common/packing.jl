# Packing routines used in direct_minimization and newton algorithms.
# They pack / unpack sets of planewaves to make them compatible to be used in
# algorithms from KrylovKit or Optim libraries

pack(φ) = vcat(Base.vec.(φ)...) # TODO as an optimization, do that lazily? See LazyArrays
function unpacking(φ) where T

    Nk = length(φ)

    lengths = [length(φ[ik]) for ik = 1:Nk]
    starts = copy(lengths)
    starts[1] = 1
    for ik = 1:Nk-1
        starts[ik+1] = starts[ik] + lengths[ik]
    end
    unpack(x) = [@views reshape(x[starts[ik]:starts[ik]+lengths[ik]-1], size(φ[ik]))
                 for ik = 1:Nk]

    unpack
end





