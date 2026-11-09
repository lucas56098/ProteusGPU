// implements printing messages and stopping the run (logging.h)

#include "logging/logging.h"

#include "mpi/mpi.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace logging {

    namespace {

        constexpr int MAX_LINE_LENGTH = 1024;

        constexpr int COLUMN_OF_ALIGNED_TEXT = 12;

        bool text_is_aligned = false;

        void print_line(FILE* stream, bool with_rank, const char* format, va_list args);

    } // namespace

    void print_from_this_rank(const char* format, ...) {
        va_list args;
        va_start(args, format);
        print_line(stdout, false, format, args);
        va_end(args);
    }

    void print(const char* format, ...) {
        if (!mpi::is_root()) return;
        va_list args;
        va_start(args, format);
        print_line(stdout, false, format, args);
        va_end(args);
    }

    void print_labelled(const char* label, const char* format, ...) {
        if (!mpi::is_root()) return;
        char    text[MAX_LINE_LENGTH];
        va_list args;
        va_start(args, format);
        std::vsnprintf(text, sizeof text, format, args);
        va_end(args);
        const int label_with_colon = (int)std::strlen(label) + 1;
        const int spaces           = text_is_aligned ? std::max(1, COLUMN_OF_ALIGNED_TEXT - label_with_colon) : 1;
        std::fprintf(stdout, "%s:%*s%s\n", label, spaces, "", text);
        std::fflush(stdout);
    }

    void align_labelled_text() {
        text_is_aligned = true;
    }

    void stop_run(const char* format, ...) {
        va_list args;
        va_start(args, format);
        print_line(stderr, true, format, args);
        va_end(args);
        mpi::stop_all_ranks();
    }

    namespace {

        // prints whole line at once so that messages of different ranks do not mix
        void print_line(FILE* stream, bool with_rank, const char* format, va_list args) {
            char message[MAX_LINE_LENGTH];
            std::vsnprintf(message, sizeof message, format, args);
            if (with_rank) {
                std::fprintf(stream, "[rank %d] %s\n", mpi::rank(), message);
            } else {
                std::fprintf(stream, "%s\n", message);
            }
            std::fflush(stream);
        }

    } // namespace

} // namespace logging
