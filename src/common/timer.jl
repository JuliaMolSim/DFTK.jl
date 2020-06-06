# Control whether timings are enabled or not, by default no
if parse(Bool, get(ENV, "DFTK_TIMING", "0"))
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

"""
Similar to `@timing`, but disabled in parallel runs.
Should be used to time threaded regions,
since TimerOutputs is not thread-safe and breaks otherwise.
"""
macro timing_seq(args...)
    if Threads.nthreads() == 1 && DFTK.timer_enabled()
        TimerOutputs.timer_expr(__module__, false, :($(DFTK.timer)), args...)
    else  # Disable taking timings
        :($(esc(last(args))))
    end
end
