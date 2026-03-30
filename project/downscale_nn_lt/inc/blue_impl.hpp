// Made by Matthew Strumillo 2024.07.20
//
#pragma once
//
#define CONSOLE_COLOR_ENABLED
#define LOGGER_TIME_FORMAT "%f"
#define MEMORY_EXIT_SIZE 64
#define MEMORY_TYPE u64
//
#include <blue/error.hpp>

#define BINIT(msg) { \
    TIMESTAMP_BEGIN = TIMESTAMP::GetCurrent (); \
    DEBUG (DEBUG_FLAG_LOGGING) putc ('\n', stdout); \
    LOGINFO (msg); \
}

#define BSTOP(msg) { \
    LOGMEMORY (); \
	LOGINFO (msg); \
	DEBUG (DEBUG_FLAG_LOGGING) putc ('\n', stdout); \
}

#define REG 
