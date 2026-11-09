#pragma once

// Printing messages and stopping the run; a message starts with the part of the run it is about, e.g. "VORONOI: ...".

namespace logging {

    // print from the rank that calls it; the caller puts the rank in the message
    void print_from_this_rank(const char* format, ...) __attribute__((format(printf, 1, 2)));

    // printf if rank 0
    void print(const char* format, ...) __attribute__((format(printf, 1, 2)));

    // as print, with "label:" in front of the text
    void print_labelled(const char* label, const char* format, ...) __attribute__((format(printf, 2, 3)));

    // from now on the text of labelled lines starts in one column
    void align_labelled_text();

    [[noreturn]] void stop_run(const char* format, ...) __attribute__((format(printf, 1, 2)));

} // namespace logging
