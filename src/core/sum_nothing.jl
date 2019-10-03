"""
Sum arbitrary arguments, ignoring those which are nothing
Returns either a number or nothing. Empty sums are nothing.
"""
sum_nothing() = nothing
sum_nothing(arg) = arg
sum_nothing(one, two) = one + two
sum_nothing(::Nothing, other) = other
sum_nothing(other, ::Nothing) = other
sum_nothing(::Nothing, ::Nothing) = nothing
sum_nothing(one, two, other...) = sum_nothing(sum_nothing(one, two), other...)

