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
    @static if @load_preference("timer_enabled", "true") == "true"
        # Copy of https://github.com/KristofferC/TimerOutputs.jl/blob/master/src/TimerOutput.jl#L174
        # because macros calling macros does not work easily in Julia
        blocks = TimerOutputs.timer_expr(__source__, __module__, false,
                                         :($(DFTK.timer)), args...)
        if blocks isa Expr
            blocks
        else
            Expr(:block,
                blocks[1],                  # the timing setup
                Expr(:tryfinally,
                    :($(esc(args[end]))),   # the user expr
                    :($(blocks[2]))         # the timing finally
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
