#ifndef LOG_H
#define LOG_H
#pragma once

#include <fstream>
#include <ostream>
#include <string>

namespace logging {

    // File-backed logger that writes on rank 0 by default and falls back to a no-op sink elsewhere.
    class FileLogger {
      public:
        FileLogger() = default;
        explicit FileLogger(const std::string& path);
        ~FileLogger();

        FileLogger(FileLogger&&) noexcept            = default;
        FileLogger& operator=(FileLogger&&) noexcept = default;
        FileLogger(const FileLogger&)                = delete;
        FileLogger& operator=(const FileLogger&)     = delete;

        void          flush();
        std::ostream& root();

      private:
        std::ofstream file;
    };

    // std::cout on rank 0, a no-op sink elsewhere
    std::ostream& root();

    int       sum_global(int local);
    long long sum_global(long long local);
    int       max_global(int local);
    int       omp_threads();

} // namespace logging

#endif // LOG_H
