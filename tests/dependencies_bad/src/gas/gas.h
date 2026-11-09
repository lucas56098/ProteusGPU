#pragma once

// gas may use math, but not through a path with "../".

#include "../math/math.h" // forbidden: a path with ../
#include "math/math.h"
