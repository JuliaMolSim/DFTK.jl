using Logging

function meta_formatter(level::LogLevel, args...)
    color = Logging.default_logcolor(level)
    Info == level && return color, "", ""
    Logging.default_metafmt(level, args...)
end
global_logger(ConsoleLogger(stdout, Info; meta_formatter))
