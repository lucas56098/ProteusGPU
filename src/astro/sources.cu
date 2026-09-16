// calls the source terms in order (sources.h)

#include "agn.h"
#include "cooling.h"
#include "gravity.h"
#include "limiters.h"
#include "sources.h"
#include "stars.h"

namespace astro {

    // reads the parameters of every source term that is compiled in
    void sources_init() {
#ifdef GRAVITY_ENABLED
        gravity_init();
#endif
#ifdef COOLING
        cooling_init();
#endif
#ifdef SF_FEEDBACK
        stars_init();
#endif
#ifdef AGN_ENABLED
        agn_init();
#endif
#ifdef LIMITERS
        limiters_init();
#endif
    }

    // once per step, before the timestep is chosen
    void sources_prepare() {
#ifdef AGN_ENABLED
        agn_prepare();
#endif
    }

    void apply_sources_first_half(double dt_half) {
        (void)dt_half;
#ifdef GRAVITY_ENABLED
        gravity_apply(dt_half);
#endif
#ifdef COOLING
        cooling_apply(dt_half);
#endif
#ifdef SF_FEEDBACK
        stars_apply(dt_half);
#endif
#ifdef AGN_ENABLED
        agn_apply(dt_half);
#endif
    }

    // reverse order, so the whole step stays symmetric
    void apply_sources_second_half(double dt_half) {
        (void)dt_half;
#ifdef AGN_ENABLED
        agn_apply(dt_half);
#endif
#ifdef SF_FEEDBACK
        stars_apply(dt_half);
#endif
#ifdef COOLING
        cooling_apply(dt_half);
#endif
#ifdef GRAVITY_ENABLED
        gravity_apply(dt_half);
#endif
#ifdef LIMITERS
        limiters_apply();
#endif
    }

} // namespace astro
