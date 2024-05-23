using Preferences

# Control whether timings are enabled or not, by default no
# Note: TimerOutputs is not thread-safe, so do not use `@timeit`
# or `@timing` in threaded regions unless you know what you are doing.
function set_timer_enabled!(state=false)
    @set_preferences!("timer_enabled" => string(state))
    @info "timer_enabled preference changed. This is a permanent change, restart julia to see the effect."
end

"""TimerOutput object used to store DFTK timings."""
const timer = TimerOutput()

"""
Shortened version of the `@timeit` macro from `TimerOutputs`,
which writes to the DFTK timer.
"""
macro timing(args...)
    @static if @load_preference("timer_enabled", "false") == "true"
        TimerOutputs.timer_expr(__module__, false, :($(DFTK.timer)), args...)
    else  # Disable taking timings
        :($(esc(last(args))))
    end
end
