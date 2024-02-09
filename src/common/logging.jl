using Logging
#using Preferences

# Bypasses everything to ConsoleLogger but Info which just shows message without any
# formatting.
Base.@kwdef struct DFTKLogger <: AbstractLogger
    io::IO
    min_level::LogLevel = Info
    fallback = ConsoleLogger(io, min_level)
end
function Logging.handle_message(logger::DFTKLogger, level, msg, args...; kwargs...)
    level == Info && return level < logger.min_level ? nothing : println(logger.io, msg)
    Logging.handle_message(logger.fallback, level, msg, args...; kwargs...)
end
Logging.min_enabled_level(logger::DFTKLogger) = logger.min_level
Logging.shouldlog(::DFTKLogger, args...) = true

# Minimum log level is read from LocalPreferences.toml; defaults to Info.
#min_level = @load_preference("min_log_level"; default="0")
default_logger() = DFTKLogger(; io=stdout)
