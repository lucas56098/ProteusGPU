// implements the cooling table (cooling.h)

#include "astro/cooling.h"

#include "logging/logging.h"

#include <fstream>
#include <sstream>
#include <vector>

namespace astro {

    void read_cooling_table(const std::string& path, CoolingTable& table) {
        std::ifstream file(path);
        if (!file) logging::stop_run("COOLING: cannot open the cooling table %s", path.c_str());

        std::vector<double> temperature, rate;
        std::string         line;
        while (std::getline(file, line)) {
            const size_t first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos || line[first] == '#') continue;
            std::istringstream row(line);
            double             T, L;
            if (row >> T >> L) {
                temperature.push_back(T);
                rate.push_back(L);
            }
        }
        const int n = (int)temperature.size();
        if (n < 2) logging::stop_run("COOLING: the cooling table %s needs at least two rows", path.c_str());

        // the slopes between the rows, and Y from the hottest row down
        table.n_rows = n;
        table.temperature.grow_to_at_least(n);
        table.rate.grow_to_at_least(n);
        table.slope.grow_to_at_least(n - 1);
        table.scaled_time.grow_to_at_least(n);
        for (int k = 0; k < n; k++) {
            table.temperature[k] = temperature[k];
            table.rate[k]        = rate[k];
        }
        for (int k = 0; k < n - 1; k++)
            table.slope[k] = std::log(rate[k + 1] / rate[k]) / std::log(temperature[k + 1] / temperature[k]);

        const double hottest_temperature = temperature[n - 1];
        const double hottest_rate        = rate[n - 1];
        table.scaled_time[n - 1]         = 0.0;
        for (int k = n - 2; k >= 0; k--) {
            const double a       = table.slope[k];
            const double ratio   = temperature[k + 1] / temperature[k];
            const double factor  = (hottest_rate / hottest_temperature) * temperature[k] / rate[k];
            const double piece   = (std::fabs(1.0 - a) > 1e-6) ? factor / (1.0 - a) * (std::pow(ratio, 1.0 - a) - 1.0)
                                                               : factor * std::log(ratio);
            table.scaled_time[k] = table.scaled_time[k + 1] + piece;
        }
        logging::print("COOLING: loaded %d node table, T = [%g, %g] K", n, temperature[0], hottest_temperature);
    }

} // namespace astro
