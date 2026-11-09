// implements the reading of the command line and param.txt (params.h)

#include "params/params.h"

#include "logging/logging.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>

namespace params {

    namespace {

        constexpr double SMALLEST_POSITIVE = std::numeric_limits<double>::min();
        constexpr double NO_UPPER_END      = std::numeric_limits<double>::max();

        // the keys and values of a parameter file, and which keys were read
        class ParameterFile {
          public:
            explicit ParameterFile(const char* path);

            // value of key, a number in [smallest, largest] or a text; meaning goes into the message when it is wrong
            double number(const char* key, double smallest, double largest, const char* meaning);
            std::string text(const char* key, const char* meaning);

            bool has(const char* key) const { return value_of.count(key) > 0; }
            int n_keys() const { return (int)value_of.size(); }
            void stop_if_a_key_was_not_read() const;

          private:
            std::string                        path;
            std::map<std::string, std::string> value_of;
            std::set<std::string>              read;

            const std::string& value(const char* key, const char* meaning);
        };

        void read_astro_parameters(ParameterFile& file, astro::Parameters& astro);
        std::string trimmed(const std::string& text);
        std::string range_in_words(double smallest, double largest);

    } // namespace

    // every parameter of a run: its key in the file, its range, what it means
    RunParameters read_parameters(int argc, char* argv[]) {
        const bool flag_is_known = argc == 2 || (argc == 3 && (!strcmp(argv[2], "0") || !strcmp(argv[2], "1")));
        if (!flag_is_known) {
            logging::stop_run("INPUT: start a run as %s <parameter file> [1 to go on from the last snapshot]", argv[0]);
        }
        const char*   path = argv[1];
        ParameterFile file(path);
        RunParameters run;
        run.from_last_snapshot = argc == 3 && !strcmp(argv[2], "1");

        run.ic_file          = file.text("ic_file", "the initial conditions, an HDF5 file");
        run.output_directory = file.text("output_directory", "where the snapshots go");
        run.output_dt    = file.number("output_dt", SMALLEST_POSITIVE, NO_UPPER_END, "the time between two snapshots");
        run.time_end     = file.number("time_end", SMALLEST_POSITIVE, NO_UPPER_END, "the time the run ends at");
        run.cfl_fraction = file.number("CFL_frac", SMALLEST_POSITIVE, 1.0, "the share of its CFL step a cell takes");
        run.memory_limit_in_gb =
            file.number("memory_limit_in_gb", SMALLEST_POSITIVE, NO_UPPER_END, "the memory one rank may use, in GB");
        if constexpr (config::regularization_speed_by_volume > 0.0) {
            if (file.has("vol_ref_cell_size")) {
                run.reference_cell_size = file.number(
                    "vol_ref_cell_size", SMALLEST_POSITIVE, NO_UPPER_END, "the cell size of the volume regularization");
            }
        }
        if constexpr (config::astro_physics) read_astro_parameters(file, run.astro);

        file.stop_if_a_key_was_not_read();
        logging::print("INPUT: Loaded %d parameters from %s", file.n_keys(), path);
        return run;
    }

    namespace {

        // only the keys of the switches turned on
        void read_astro_parameters(ParameterFile& file, astro::Parameters& astro) {
            astro.units.length_in_cm =
                file.number("UnitLength_in_cm", SMALLEST_POSITIVE, NO_UPPER_END, "the length of the code unit in cm");
            astro.units.mass_in_g =
                file.number("UnitMass_in_g", SMALLEST_POSITIVE, NO_UPPER_END, "the mass of the code unit in g");
            astro.units.velocity_in_cm_per_s = file.number(
                "UnitVelocity_in_cm_per_s", SMALLEST_POSITIVE, NO_UPPER_END, "the velocity of the code unit in cm/s");

            if constexpr (config::nfw) {
                astro.halo_mass_in_solar_masses =
                    file.number("M_NFW", SMALLEST_POSITIVE, NO_UPPER_END, "the mass of the NFW halo in Msun");
                astro.halo_concentration =
                    file.number("c_NFW", SMALLEST_POSITIVE, NO_UPPER_END, "the concentration of the NFW halo");
                astro.hubble_constant_in_km_per_s_mpc =
                    file.number("H0", SMALLEST_POSITIVE, NO_UPPER_END, "the Hubble constant in km/s/Mpc");
            }
            if constexpr (config::hernquist || config::star_feedback) {
                astro.galaxy_mass_in_solar_masses =
                    file.number("M_BCG", 0.0, NO_UPPER_END, "the mass of the galaxy in Msun");
                astro.galaxy_radius_in_kpc =
                    file.number("R_BCG", SMALLEST_POSITIVE, NO_UPPER_END, "the radius of the galaxy in kpc");
            }
            if constexpr (config::smbh) {
                astro.black_hole_mass_in_solar_masses =
                    file.number("M_BH", 0.0, NO_UPPER_END, "the black hole mass in Msun");
                astro.black_hole_softening_in_kpc = file.number(
                    "smbh_softening", SMALLEST_POSITIVE, NO_UPPER_END, "the softening of the black hole in kpc");
            }
            if constexpr (config::cooling) {
                astro.cooling_table =
                    file.text("cooling_table", "the cooling table: temperature in K, rate in erg cm^3/s");
                astro.temperature_floor_in_k =
                    file.number("T_floor", SMALLEST_POSITIVE, NO_UPPER_END, "the lowest temperature in K");
            }
            if constexpr (config::star_feedback) {
                astro.supernova_rate_per_year_and_solar_mass =
                    file.number("Gamma_SNIa", 0.0, NO_UPPER_END, "the supernova Ia rate per year and Msun of stars");
                astro.supernova_energy_in_erg =
                    file.number("E_SNIa", 0.0, NO_UPPER_END, "the energy of a supernova Ia in erg");
                astro.mass_loss_rate_per_s =
                    file.number("alpha_SNIa", 0.0, NO_UPPER_END, "the mass the stars lose per time and mass, per s");
                astro.star_formation_efficiency =
                    file.number("eff_SF", 0.0, 1.0, "the share of the rest energy star formation gives as heat");
                astro.star_formation_density_in_per_cm3 =
                    file.number("n_SF", 0.0, NO_UPPER_END, "the hydrogen density above which stars form, per cm^3");
                astro.star_formation_temperature_in_k =
                    file.number("T_SF", 0.0, NO_UPPER_END, "the temperature below which stars form, in K");
                astro.star_formation_radius_in_kpc =
                    file.number("R_SF", 0.0, NO_UPPER_END, "the radius inside which stars form, in kpc");
            }
            if constexpr (config::star_feedback || config::agn) {
                astro.accretion_radius_in_kpc = file.number(
                    "R_acc", 0.0, NO_UPPER_END, "the radius of the cold gas the black hole feeds on, in kpc");
            }
            if constexpr (config::agn) {
                astro.cold_temperature_in_k = file.number(
                    "T_cold_acc", 0.0, NO_UPPER_END, "the temperature below which gas feeds the black hole, in K");
                astro.accretion_time_in_myr = file.number(
                    "t_acc", SMALLEST_POSITIVE, NO_UPPER_END, "the time over which the cold gas falls in, in Myr");
                astro.feedback_efficiency =
                    file.number("eta_agn", 0.0, 1.0, "the share of the rest energy that comes back out");
                astro.heated_radius_in_kpc =
                    file.number("R_T", SMALLEST_POSITIVE, NO_UPPER_END, "the radius of the heated sphere in kpc");
                astro.thermal_share = file.number("f_T", 0.0, 1.0, "the share of the feedback as heat");
            }
            if constexpr (config::agn_kinetic) {
                astro.kinetic_share = file.number("f_K", 0.0, 1.0, "the share of the feedback in the jets");
                astro.jet_radius_in_kpc =
                    file.number("R_jet", SMALLEST_POSITIVE, NO_UPPER_END, "the radius of a jet in kpc");
                astro.jet_height_in_kpc =
                    file.number("h_jet", SMALLEST_POSITIVE, NO_UPPER_END, "the height of the zone of a jet in kpc");
                astro.jet_distance_in_kpc =
                    file.number("L_jet", 0.0, NO_UPPER_END, "the distance of a jet from the centre in kpc");
                astro.jet_speed_cap_in_speed_of_light =
                    file.number("v_cap", SMALLEST_POSITIVE, 1.0, "the largest speed in a jet, in speeds of light");
            }
            if constexpr (config::limiters) {
                astro.limited_radius_in_kpc =
                    file.number("R_lim", 0.0, NO_UPPER_END, "the radius of the limiters in kpc");
                astro.temperature_cap_in_k =
                    file.number("T_max_lim", SMALLEST_POSITIVE, NO_UPPER_END, "the highest temperature there, in K");
                astro.speed_cap_in_speed_of_light =
                    file.number("v_cap_lim", SMALLEST_POSITIVE, 1.0, "the largest speed there, in speeds of light");
            }
        }

        // every line "key = value", a # starting a comment, also behind a value
        ParameterFile::ParameterFile(const char* path) : path(path) {
            std::ifstream file(path);
            if (!file.is_open()) logging::stop_run("INPUT: cannot open the parameter file %s", path);

            std::string line;
            for (int line_number = 1; std::getline(file, line); line_number++) {
                line = trimmed(line.substr(0, line.find('#')));
                if (line.empty()) continue;

                const size_t      equals = line.find('=');
                const std::string key    = (equals == std::string::npos) ? "" : trimmed(line.substr(0, equals));
                const std::string value  = (equals == std::string::npos) ? "" : trimmed(line.substr(equals + 1));
                if (key.empty() || value.empty()) {
                    logging::stop_run(
                        "INPUT: line %d of %s is not \"key = value\": %s", line_number, path, line.c_str());
                }
                if (!value_of.emplace(key, value).second) {
                    logging::stop_run("INPUT: %s is set twice in %s, again on line %d", key.c_str(), path, line_number);
                }
            }
        }

        double ParameterFile::number(const char* key, double smallest, double largest, const char* meaning) {
            const std::string& text_value = value(key, meaning);
            char*              end        = nullptr;
            const double       number     = std::strtod(text_value.c_str(), &end);
            if (end == text_value.c_str() || *end != '\0') {
                logging::stop_run("INPUT: %s = %s in %s is not a number; %s is %s",
                                  key,
                                  text_value.c_str(),
                                  path.c_str(),
                                  key,
                                  meaning);
            }
            if (!(number >= smallest && number <= largest)) {
                logging::stop_run("INPUT: %s = %s in %s must be %s; %s is %s",
                                  key,
                                  text_value.c_str(),
                                  path.c_str(),
                                  range_in_words(smallest, largest).c_str(),
                                  key,
                                  meaning);
            }
            return number;
        }

        std::string ParameterFile::text(const char* key, const char* meaning) {
            return value(key, meaning);
        }

        const std::string& ParameterFile::value(const char* key, const char* meaning) {
            const auto found = value_of.find(key);
            if (found == value_of.end())
                logging::stop_run("INPUT: %s is missing in %s; it is %s", key, path.c_str(), meaning);
            read.insert(key);
            return found->second;
        }

        // a key no parameter of this run reads is a typo or left over from another code
        void ParameterFile::stop_if_a_key_was_not_read() const {
            std::string not_read;
            for (const auto& entry : value_of) {
                if (read.count(entry.first) == 0) not_read += (not_read.empty() ? "" : ", ") + entry.first;
            }
            if (!not_read.empty()) {
                logging::stop_run("INPUT: %s in %s is not a parameter of this run", not_read.c_str(), path.c_str());
            }
        }

        std::string range_in_words(double smallest, double largest) {
            char low[64], high[64];
            std::snprintf(low, sizeof(low), "at least %g", smallest);
            std::snprintf(high, sizeof(high), " and at most %g", largest);
            return std::string((smallest == SMALLEST_POSITIVE) ? "above 0" : low) +
                   ((largest == NO_UPPER_END) ? "" : high);
        }

        std::string trimmed(const std::string& text) {
            const size_t first = text.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) return "";
            const size_t last = text.find_last_not_of(" \t\r\n");
            return text.substr(first, last - first + 1);
        }

    } // namespace

} // namespace params
