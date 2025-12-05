import ExprTools: splitdef, combinedef
using Preferences

# Control whether timings are enabled or not, by default yes
# Note: TimerOutputs is not thread-safe, so do not use `@timeit`
# or `@timing` in threaded regions unless you know what you are doing.

"""TimerOutput object used to store DFTK timings."""
const timer = TimerOutput()

"""
Shortened version of the `@timeit` macro from `TimerOutputs`,
which writes to the DFTK timer.

Also wraps the code in [`push_range`](@ref)/[`pop_range`](@ref) calls for
instrumentation when running on the GPU.
"""
macro timing(args...)
    length(args) >= 1 || error("@timing requires at least one argument: an expression to time")
    length(args) <= 2 || error("@timing takes at most two arguments: a label and an expression")
    @static if @load_preference("timer_enabled", "true") == "true"
        # Copy of https://github.com/KristofferC/TimerOutputs.jl/blob/master/src/TimerOutput.jl#L174
        # because macros calling macros does not work easily in Julia
        blocks = TimerOutputs.timer_expr(__source__, __module__, false,
                                         :($(DFTK.timer)), args...)
        if blocks isa Expr
            # This should be a function definition wrapped in esc.
            @assert blocks.head == :escape
            @assert length(blocks.args) == 1

            # Split function definition
            def = splitdef(blocks.args[1])
            label = length(args) == 2 ? args[1] : string(def[:name])

            @gensym val
            def[:body] = quote
                $(push_range)($(label))
                $(Expr(
                    :tryfinally,
                    :($val = $(def[:body])),
                    :($(pop_range)()),
                ))
                $val
            end

            esc(combinedef(def))
        else
            # This should be a standard expression, for which a label must have been provided.
            @assert length(args) == 2
            label = args[1]

            Expr(:block,
                blocks[1],                  # the timing setup
                Expr(:tryfinally,
                    Expr(:block,
                        :(push_range($(esc(label)))),
                        Expr(:tryfinally,
                            :($(esc(args[end]))),   # the user expr
                            :(pop_range()),
                        ),
                    ),
                    :($(blocks[2]))         # the timing finally
                )
            )
        end
    else  # Disable taking timings
        :($(esc(last(args))))
    end
end

"""
Wraps the code in [`push_range`](@ref)/[`pop_range`](@ref) calls for
instrumentation when running on the GPU.
"""
macro instrument(label, expr)
    @static if @load_preference("timer_enabled", "true") == "true"
        Expr(:block,
            :(push_range($(esc(label)))),
            Expr(:tryfinally,
                :($(esc(expr))),
                :(pop_range()),
            ),
        )
    else
        :($(esc(expr)))
    end
end

function set_timer_enabled!(state=true)
    @set_preferences!("timer_enabled" => string(state))
    @info "timer_enabled preference changed. This is a permanent change, restart julia to see the effect."
end

# TODO: should probably use FunctionWrappers since closure cfunction won't work on arm...
"""
Registered pair of instrumentation callbacks.
We use function pointers to avoid the overhead of dynamic dispatch.
"""
struct InstrumentationCallback
    name
    # (message::Cstring,) -> Cvoid
    push_range::Base.CFunction
    # (sync_device::Bool,) -> Cvoid
    pop_range::Base.CFunction
end

const instrumentation_callbacks = InstrumentationCallback[]

function register_instrumentation_callback(name, push_cb, pop_cb)
    push!(instrumentation_callbacks, InstrumentationCallback(
        name,
        @cfunction($push_cb, Cvoid, (Cstring,)),
        @cfunction($pop_cb, Cvoid, (Bool,)),
    ))
    nothing
end

"""
Push a new range to the instrumentation callbacks.
This should be followed by a corresponding [`pop_range`](@ref) call,
preferably using a `try...finally` block.
"""
function push_range(message::String)
    for cb in instrumentation_callbacks
        ccall(Base.unsafe_convert(Ptr{Cvoid}, cb.push_range), Cvoid, (Cstring,), message)
    end
end

"""
Pop the current range from the instrumentation callbacks.
"""
function pop_range()
    for cb in instrumentation_callbacks
        ccall(Base.unsafe_convert(Ptr{Cvoid}, cb.pop_range), Cvoid, (Bool,), false)
    end
end
