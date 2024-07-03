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
        TimerOutputs.timer_expr(__module__, false, :($(DFTK.timer)), args...)
    else  # Disable taking timings
        :($(esc(last(args))))
    end
end

function set_timer_enabled!(state=true)
    @set_preferences!("timer_enabled" => string(state))
    @info "timer_enabled preference changed. This is a permanent change, restart julia to see the effect."
end
