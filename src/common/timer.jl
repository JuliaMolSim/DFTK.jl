const to = TimerOutput()
# creating a new macro to shorten the call :
# replaces `@timeit to [label] [block]` by `@timer [label] [block]`
macro timer(args...)
    TimerOutputs.timer_expr(__module__, false, to, args...)
end
