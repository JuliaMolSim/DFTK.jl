"""
Split an iterable evenly into N chunks, which will be returned.
"""
function split_evenly(itr, N)
    count     = length(itr) ÷ N
    remainder = length(itr) % N

    map(0:N-1) do i
        # The first remainder chunks get count + 1 tasks
        if i < remainder
            start = 1 + i * (count + 1)
            stop  = start + count
        else
            start = 1 + i * count + remainder
            stop  = start + count - 1
        end
        itr[start:stop]
    end
end
