# Control whether timings are enabled or not, by default yes
# if get(ENV, "DFTK_TIMING", "1") == "1"
#     timer_enabled() = :parallel
# elseif ENV["DFTK_TIMING"] == "all"
#     timer_enabled() = :all
# else
#     timer_enabled() = :none
# end

# temporarily disable all timers for Zygote (try-catch not supported) TODO re-enable
timer_enabled() = :none

"""TimerOutput object used to store DFTK timings."""
const timer = TimerOutput()

"""
Shortened version of the `@timeit` macro from `TimerOutputs`,
which writes to the DFTK timer.
"""
macro timing(args...)
    if DFTK.timer_enabled() in (:parallel, :all)
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
    if DFTK.timer_enabled() == :all
        TimerOutputs.timer_expr(__module__, false, :($(DFTK.timer)), args...)
    else  # Disable taking timings
        :($(esc(last(args))))
    end
end
