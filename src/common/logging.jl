using Logging

# Removing most of format for `@info` in default logger.
function meta_formatter(level::LogLevel, args...)
    color = Logging.default_logcolor(level)
    Info == level && return color, "", ""
    Logging.default_metafmt(level, args...)
end

# Bypasses everything to ConsoleLogger but Info which just shows message without any
# formatting.
Base.@kwdef struct DFTKLogger <: AbstractLogger
    io::IO
    min_level::LogLevel = Info
    fallback = ConsoleLogger(io, min_level; meta_formatter)
end
function Logging.handle_message(logger::DFTKLogger, level, msg, args...; kwargs...)
    level == Info && return level < logger.min_level ? nothing : println(logger.io, msg)
    Logging.handle_message(logger.fallback, level, msg, args...; kwargs...)
end
Logging.min_enabled_level(logger::DFTKLogger) = logger.min_level
Logging.shouldlog(::DFTKLogger, args...) = true
