// ngbtree may use keys, path and device directly, and math and the others through them; of each only the public header.

#include "device/device.h"
#include "device/memory.h" // forbidden: a private header of device
#include "keys/keys.h"
#include "math/math.h"
#include "missing.h" // forbidden: not a header in this folder
#include "ngbtree/ngbtree.h"
#include "search.cu" // forbidden: a .cu file
#include "tree.h"
