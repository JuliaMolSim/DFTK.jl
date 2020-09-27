# Control whether expensive assertions are enabled or not
if get(ENV, "DFTK_PROFILE", "develop") == "performance"
    assert_expensive_enabled() = false
else
    assert_expensive_enabled() = true
end

macro assert_expensive(expr)
    esc(:(
        if $(@__MODULE__).assert_expensive_enabled()
            @assert($expr)
        end
    ))
end

macro assert_expensive(expr, text)
    esc(:(
        if $(@__MODULE__).assert_expensive_enabled()
            @assert($expr, $text)
        end
    ))
end
