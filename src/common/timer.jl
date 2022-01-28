# Control whether timings are enabled or not, by default yes
# Note: TimerOutputs is not thread-safe, so do not use `@timeit`
# or `@timing` in threaded regions unless you know what you are doing.
if get(ENV, "DFTK_TIMING", "1") == "1"
    timer_enabled() = true
else
    timer_enabled() = false
end

"""TimerOutput object used to store DFTK timings."""
const timer = TimerOutput()

"""
Shortened version of the `@timeit` macro from `TimerOutputs`,
which writes to the DFTK timer.
"""
macro timing(args...)
    if DFTK.timer_enabled()
        TimerOutputs.timer_expr(__module__, false, :($(DFTK.timer)), args...)
    else  # Disable taking timings
        :($(esc(last(args))))
    end
end
