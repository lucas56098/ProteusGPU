/* Strang-split source-term orchestration */
#include "sources.h"
#include "agn.h"
#include "cooling.h"
#include "gravity.h"
#include "limiters.h"
#include "stars.h"

namespace astro {

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

    // first-half order: gravity, cooling, stars, agn
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

    // reversed order of the first half, then the global hard clamps (idempotent -- once per full
    // step is enough since the CFL already bounds the post-AGN sound speed inside R_T)
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
