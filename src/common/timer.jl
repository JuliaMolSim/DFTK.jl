import ExprTools: splitdef, combinedef
import Libdl
using Preferences

# Control whether timings are enabled or not, by default yes
# Note: TimerOutputs is not thread-safe, so do not use `@timeit`
# or `@timing` in threaded regions unless you know what you are doing.

"""TimerOutput object used to store DFTK timings."""
const timer = TimerOutput()

"""
Shortened version of the `@timeit` macro from `TimerOutputs`,
which writes to the DFTK timer.
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
                $(roctx_push)($(label))
                $(Expr(
                    :tryfinally,
                    :($val = $(def[:body])),
                    :($(roctx_pop)()),
                ))
                $val
            end

            esc(combinedef(def))
        else
            # This should be a standard expression, for which a label must have been provided.
            @assert length(args) == 2
            label = args[1]

            Expr(:block,
                :(roctx_push($(esc(label)))),
                Expr(:tryfinally,
                    Expr(:block,
                        blocks[1],                  # the timing setup
                        Expr(:tryfinally,
                            :($(esc(args[end]))),   # the user expr
                            :($(blocks[2]))         # the timing finally
                        ),
                    ),
                    :(roctx_pop()),
                )
            )
        end
    else  # Disable taking timings
        :($(esc(last(args))))
    end
end

function set_timer_enabled!(state=true)
    @set_preferences!("timer_enabled" => string(state))
    @info "timer_enabled preference changed. This is a permanent change, restart julia to see the effect."
end

# TODO: decide where to put it; it should remain trigger dynamically to avoid invalidating all precompilations
const roctx_lib = Ref{String}("")

function init_roctx()
    roctx_lib[] = Libdl.find_library("libroctx64")
end

function roctx_push(message)
    if roctx_lib[] != ""
        ccall((:roctxRangePushA, roctx_lib[]), Cvoid, (Cstring,), message)
    end
end

function roctx_pop()
    if roctx_lib[] != ""
        ccall((:roctxRangePop, roctx_lib[]), Cvoid, ())
    end
end
