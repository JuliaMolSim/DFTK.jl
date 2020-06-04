const timer = TimerOutput()
# creating a new macro to shorten the call :
# replaces `@timeit to [label] [block]` by `@timer [label] [block]`
macro timing(args...)
    TimerOutputs.timer_expr(__module__, false, timer, args...)
end
