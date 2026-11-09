#pragma once

// cells may use the layers physics and time, but not integrator, which comes after it.

#include "hydro/hydro.h"
#include "integrator/integrator.h" // forbidden: cells may not use integrator
#include "timesteps/timesteps.h"
