#pragma once

// the math functions, bitwise equal on CPU and GPU, and the radius of a ball (included by math.h)

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>

// ported from fdlibm as in FreeBSD's lib/msun (s_cbrt.c, e_log.c, e_exp.c, e_pow.c), so CPU and GPU give the same bits

// cube_root, natural_log:
//   Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//   Developed at SunPro, a Sun Microsystems, Inc. business.
//   Permission to use, copy, modify, and distribute this
//   software is freely granted, provided that this notice
//   is preserved.
//   (cube_root optimized by Bruce D. Evans.)
//
// exponential, power:
//   Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
//   Permission to use, copy, modify, and distribute this
//   software is freely granted, provided that this notice
//   is preserved.

namespace fdlibm {

    HD inline uint64_t bits_of(double x) {
        uint64_t bits;
        memcpy(&bits, &x, sizeof bits);
        return bits;
    }
    HD inline double from_bits(uint64_t bits) {
        double x;
        memcpy(&x, &bits, sizeof x);
        return x;
    }
    HD inline int32_t high_word(double x) {
        return (int32_t)(bits_of(x) >> 32);
    }
    HD inline uint32_t low_word(double x) {
        return (uint32_t)bits_of(x);
    }
    HD inline double from_words(uint32_t high, uint32_t low) {
        return from_bits(((uint64_t)high << 32) | low);
    }
    HD inline double with_high_word(double x, uint32_t high) {
        return from_words(high, low_word(x));
    }
    HD inline double with_low_word(double x, uint32_t low) {
        return from_words((uint32_t)high_word(x), low);
    }

} // namespace fdlibm

HD inline double cube_root(double x) {
    using namespace fdlibm;
    constexpr uint32_t B1 = 715094163; // (1023 - 1023/3 - 0.03306235651) 2^20
    constexpr uint32_t B2 = 696219795; // (1023 - 1023/3 - 54/3 - 0.03306235651) 2^20
    // |1/cbrt(x) - p(x)| < 2^-23.5
    constexpr double P0 = 1.87595182427177009643;
    constexpr double P1 = -1.88497979543377169875;
    constexpr double P2 = 1.621429720105354466140;
    constexpr double P3 = -0.758397934778766047437;
    constexpr double P4 = 0.145996192886612446982;

    const uint32_t low  = low_word(x);
    const uint32_t sign = (uint32_t)high_word(x) & 0x80000000;
    const int32_t  hx   = high_word(x) & 0x7fffffff;
    if (hx >= 0x7ff00000) return x + x; // NaN, inf

    // rough cbrt to 5 bits, from the exponent
    double t;
    if (hx < 0x00100000) { // zero or subnormal
        if ((hx | low) == 0) return x;
        t = from_words(0x43500000, 0) * x;
        t = from_words(sign | ((high_word(t) & 0x7fffffff) / 3 + B2), 0);
    } else {
        t = from_words(sign | (hx / 3 + B1), 0);
    }

    // to 23 bits: t p(t^3 / x)
    double r = (t * t) * (t / x);
    t        = t * ((P0 + r * (P1 + r * P2)) + ((r * r) * r) * (P3 + r * P4));

    // round t away from zero to 23 bits
    t = from_bits((bits_of(t) + 0x80000000) & 0xffffffffc0000000ULL);

    // one Halley step to 53 bits
    const double s = t * t;
    r              = x / s;
    const double w = t + t;
    r              = (r - t) / (w + r);
    return t + t * r;
}

HD inline double natural_log(double x) {
    using namespace fdlibm;
    constexpr double ln2_hi = 6.93147180369123816490e-01;
    constexpr double ln2_lo = 1.90821492927058770002e-10;
    constexpr double two54  = 1.80143985094819840000e+16;
    constexpr double Lg1    = 6.666666666666735130e-01;
    constexpr double Lg2    = 3.999999999940941908e-01;
    constexpr double Lg3    = 2.857142874366239149e-01;
    constexpr double Lg4    = 2.222219843214978396e-01;
    constexpr double Lg5    = 1.818357216161805012e-01;
    constexpr double Lg6    = 1.531383769920937332e-01;
    constexpr double Lg7    = 1.479819860511658591e-01;

    int32_t        hx = high_word(x);
    const uint32_t lx = low_word(x);

    int32_t k = 0;
    if (hx < 0x00100000) { // x < 2^-1022
        if (((hx & 0x7fffffff) | lx) == 0) return -INFINITY;
        if (hx < 0) return (x - x) / (x - x);
        k -= 54;
        x *= two54;
        hx = high_word(x);
    }
    if (hx >= 0x7ff00000) return x + x;

    // x = 2^k (1 + f) with sqrt(2)/2 < 1 + f < sqrt(2)
    k += (hx >> 20) - 1023;
    hx &= 0x000fffff;
    const int32_t halved = (hx + 0x95f64) & 0x100000;
    x                    = with_high_word(x, hx | (halved ^ 0x3ff00000));
    k += (halved >> 20);
    const double f  = x - 1.0;
    const double dk = (double)k;

    if ((0x000fffff & (2 + hx)) < 3) { // |f| < 2^-20
        if (f == 0.0) return (k == 0) ? 0.0 : dk * ln2_hi + dk * ln2_lo;
        const double R = f * f * (0.5 - 0.33333333333333333 * f);
        return (k == 0) ? f - R : dk * ln2_hi - ((R - dk * ln2_lo) - f);
    }

    // log(1 + f) = 2s + s R(s^2) with s = f / (2 + f)
    const double s  = f / (2.0 + f);
    const double z  = s * s;
    const double w  = z * z;
    const double t1 = w * (Lg2 + w * (Lg4 + w * Lg6));
    const double t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
    const double R  = t2 + t1;
    if (((hx - 0x6147a) | (0x6b851 - hx)) > 0) {
        const double hfsq = 0.5 * f * f;
        if (k == 0) return f - (hfsq - s * (hfsq + R));
        return dk * ln2_hi - ((hfsq - (s * (hfsq + R) + dk * ln2_lo)) - f);
    }
    if (k == 0) return f - s * (f - R);
    return dk * ln2_hi - ((s * (f - R) - dk * ln2_lo) - f);
}

HD inline double exponential(double x) {
    using namespace fdlibm;
    constexpr double o_threshold = 7.09782712893383973096e+02;
    constexpr double u_threshold = -7.45133219101941108420e+02;
    constexpr double ln2HI       = 6.93147180369123816490e-01;
    constexpr double ln2LO       = 1.90821492927058770002e-10;
    constexpr double invln2      = 1.44269504088896338700e+00;
    constexpr double P1          = 1.66666666666666019037e-01;
    constexpr double P2          = -2.77777777770155933842e-03;
    constexpr double P3          = 6.61375632143793436117e-05;
    constexpr double P4          = -1.65339022054652515390e-06;
    constexpr double P5          = 4.13813679705723846039e-08;
    constexpr double twom1000    = 9.33263618503218878990e-302;

    uint32_t      hx  = (uint32_t)high_word(x);
    const int32_t xsb = (hx >> 31) & 1;
    hx &= 0x7fffffff;

    if (hx >= 0x40862E42) { // |x| >= 709.78
        if (hx >= 0x7ff00000) {
            if (((hx & 0xfffff) | low_word(x)) != 0) return x + x;
            return (xsb == 0) ? x : 0.0;
        }
        if (x > o_threshold) return INFINITY;
        if (x < u_threshold) return 0.0;
    }

    // x = k ln2 + r with |r| <= ln2 / 2, r = hi - lo
    double  hi = 0.0, lo = 0.0;
    int32_t k = 0;
    if (hx > 0x3fd62e42) { // |x| > ln2 / 2
        if (hx < 0x3FF0A2B2) { // and |x| < 1.5 ln2
            hi = xsb ? x + ln2HI : x - ln2HI;
            lo = xsb ? -ln2LO : ln2LO;
            k  = 1 - xsb - xsb;
        } else {
            k              = (int32_t)(invln2 * x + (xsb ? -0.5 : 0.5));
            const double t = k;
            hi             = x - t * ln2HI;
            lo             = t * ln2LO;
        }
        x = hi - lo;
    } else if (hx < 0x3e300000) { // |x| < 2^-28
        return 1.0 + x;
    }

    // e^r from a rational fit, then times 2^k
    const double t = x * x;
    const double twopk =
        (k >= -1021) ? from_words((uint32_t)(0x3ff + k) << 20, 0) : from_words((uint32_t)(0x3ff + (k + 1000)) << 20, 0);
    const double c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    if (k == 0) return 1.0 - ((x * c) / (c - 2.0) - x);
    const double y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi);
    if (k >= -1021) return (k == 1024) ? y * 2.0 * 0x1p1023 : y * twopk;
    return y * twopk * twom1000;
}

HD inline double power(double x, double y) {
    using namespace fdlibm;
    constexpr double dp_h1 = 5.84962487220764160156e-01;
    constexpr double dp_l1 = 1.35003920212974897128e-08;
    constexpr double two53 = 9007199254740992.0;
    constexpr double thrd  = 3.3333333333333331e-01;
    // (3/2) (log(x) - 2s - 2/3 s^3)
    constexpr double L1      = 5.99999999999994648725e-01;
    constexpr double L2      = 4.28571428578550184252e-01;
    constexpr double L3      = 3.33333329818377432918e-01;
    constexpr double L4      = 2.72728123808534006489e-01;
    constexpr double L5      = 2.30660745775561754067e-01;
    constexpr double L6      = 2.06975017800338417784e-01;
    constexpr double P1      = 1.66666666666666019037e-01;
    constexpr double P2      = -2.77777777770155933842e-03;
    constexpr double P3      = 6.61375632143793436117e-05;
    constexpr double P4      = -1.65339022054652515390e-06;
    constexpr double P5      = 4.13813679705723846039e-08;
    constexpr double lg2     = 6.93147180559945286227e-01;
    constexpr double lg2_h   = 6.93147182464599609375e-01;
    constexpr double lg2_l   = -1.90465429995776804525e-09;
    constexpr double ovt     = 8.0085662595372944372e-17; // -(1024 - log2(overflow + 0.5 ulp))
    constexpr double cp      = 9.61796693925975554329e-01;
    constexpr double cp_h    = 9.61796700954437255859e-01;
    constexpr double cp_l    = -7.02846165095275826516e-09;
    constexpr double ivln2   = 1.44269504088896338700e+00;
    constexpr double ivln2_h = 1.44269502162933349609e+00;
    constexpr double ivln2_l = 1.92596299112661746887e-08;

    const int32_t  hx = high_word(x);
    const uint32_t lx = low_word(x);
    const int32_t  hy = high_word(y);
    const uint32_t ly = low_word(y);
    int32_t        ix = hx & 0x7fffffff;
    const int32_t  iy = hy & 0x7fffffff;

    if ((iy | ly) == 0) return 1.0;
    if (hx == 0x3ff00000 && lx == 0) return 1.0; // 1^y = 1, also for NaN
    if (ix > 0x7ff00000 || (ix == 0x7ff00000 && lx != 0) || iy > 0x7ff00000 || (iy == 0x7ff00000 && ly != 0))
        return x + y;

    // for x < 0: y not an integer (0), odd (1) or even (2)
    int32_t yisint = 0;
    if (hx < 0) {
        if (iy >= 0x43400000) {
            yisint = 2;
        } else if (iy >= 0x3ff00000) {
            const int32_t e = (iy >> 20) - 0x3ff;
            if (e > 20) {
                const uint32_t j = ly >> (52 - e);
                if ((j << (52 - e)) == ly) yisint = 2 - (int32_t)(j & 1);
            } else if (ly == 0) {
                const int32_t j = iy >> (20 - e);
                if ((j << (20 - e)) == iy) yisint = 2 - (j & 1);
            }
        }
    }

    // y = +-inf, +-1, 2, 0.5
    if (ly == 0) {
        if (iy == 0x7ff00000) {
            if (((ix - 0x3ff00000) | lx) == 0) return 1.0;
            if (ix >= 0x3ff00000) return (hy >= 0) ? y : 0.0;
            return (hy < 0) ? -y : 0.0;
        }
        if (iy == 0x3ff00000) return (hy < 0) ? 1.0 / x : x;
        if (hy == 0x40000000) return x * x;
        if (hy == 0x3fe00000 && hx >= 0) return sqrt(x);
    }

    // x = +-0, +-inf, +-1
    double ax = fabs(x);
    if (lx == 0 && (ix == 0x7ff00000 || ix == 0 || ix == 0x3ff00000)) {
        double z = ax;
        if (hy < 0) z = 1.0 / z;
        if (hx < 0) {
            if (((ix - 0x3ff00000) | yisint) == 0)
                z = (z - z) / (z - z);
            else if (yisint == 1)
                z = -z;
        }
        return z;
    }

    // (x < 0)^non-integer = NaN; s the sign of the result
    int32_t n = (int32_t)((uint32_t)hx >> 31) - 1;
    if ((n | yisint) == 0) return (x - x) / (x - x);
    const double s = ((n | (yisint - 1)) == 0) ? -1.0 : 1.0;

    // log2(|x|) = t1 + t2 in extra precision
    double t1, t2;
    if (iy > 0x41e00000) { // |y| > 2^31
        if (iy > 0x43f00000) { // |y| > 2^64: over- or underflow
            if (ix <= 0x3fefffff) return (hy < 0) ? INFINITY : 0.0;
            if (ix >= 0x3ff00000) return (hy > 0) ? INFINITY : 0.0;
        }
        if (ix < 0x3fefffff) return (hy < 0) ? s * INFINITY : s * 0.0;
        if (ix > 0x3ff00000) return (hy > 0) ? s * INFINITY : s * 0.0;
        // |1 - x| <= 2^-20: log(x) = t - t^2/2 + t^3/3 - t^4/4 with t = x - 1
        const double t = ax - 1.0;
        const double w = (t * t) * (0.5 - t * (thrd - t * 0.25));
        const double u = ivln2_h * t;
        const double v = t * ivln2_l - w * ivln2;
        t1             = with_low_word(u + v, 0);
        t2             = v - (t1 - u);
    } else {
        n = 0;
        if (ix < 0x00100000) { // subnormal
            ax *= two53;
            n -= 53;
            ix = high_word(ax);
        }
        n += (ix >> 20) - 0x3ff;
        const int32_t j = ix & 0x000fffff;
        ix              = j | 0x3ff00000;
        int32_t k       = 0;
        if (j <= 0x3988E) { // |x| < sqrt(3/2)
            k = 0;
        } else if (j < 0xBB67A) { // |x| < sqrt(3)
            k = 1;
        } else {
            k = 0;
            n += 1;
            ix -= 0x00100000;
        }
        ax                = with_high_word(ax, ix);
        const double bp   = k ? 1.5 : 1.0;
        const double dp_h = k ? dp_h1 : 0.0;
        const double dp_l = k ? dp_l1 : 0.0;

        double       u   = ax - bp;
        double       v   = 1.0 / (ax + bp);
        const double ss  = u * v;
        const double s_h = with_low_word(ss, 0);
        double       t_h = from_words(((ix >> 1) | 0x20000000) + 0x00080000 + (k << 18), 0);
        double       t_l = ax - (t_h - bp);
        const double s_l = v * ((u - s_h * t_h) - s_h * t_l);

        double s2 = ss * ss;
        double r  = s2 * s2 * (L1 + s2 * (L2 + s2 * (L3 + s2 * (L4 + s2 * (L5 + s2 * L6)))));
        r += s_l * (s_h + ss);
        s2  = s_h * s_h;
        t_h = with_low_word(3.0 + s2 + r, 0);
        t_l = r - ((t_h - 3.0) - s2);
        u   = s_h * t_h;
        v   = s_l * t_h + t_l * ss;

        // log2(ax) = n + dp_h + z_h + z_l
        const double p_h = with_low_word(u + v, 0);
        const double p_l = v - (p_h - u);
        const double z_h = cp_h * p_h;
        const double z_l = cp_l * p_h + p_l * cp + dp_l;
        const double t   = n;
        t1               = with_low_word(((z_h + z_l) + dp_h) + t, 0);
        t2               = z_l - (((t1 - t) - dp_h) - z_h);
    }

    // y log2(x) = p_h + p_l, with y split into y_h + (y - y_h)
    const double y_h = with_low_word(y, 0);
    const double p_l = (y - y_h) * t1 + y * t2;
    double       p_h = y_h * t1;
    double       z   = p_l + p_h;
    int32_t      j   = high_word(z);
    int32_t      i   = (int32_t)low_word(z);
    if (j >= 0x40900000) { // z >= 1024
        if (((j - 0x40900000) | i) != 0) return s * INFINITY;
        if (p_l + ovt > z - p_h) return s * INFINITY;
    } else if ((j & 0x7fffffff) >= 0x4090cc00) { // z <= -1075
        if (((j - 0xc090cc00) | i) != 0) return s * 0.0;
        if (p_l <= z - p_h) return s * 0.0;
    }

    // 2^(p_h + p_l): n the integer nearest to z, then e^((z - n) ln2) as in exponential
    i         = j & 0x7fffffff;
    int32_t k = (i >> 20) - 0x3ff;
    n         = 0;
    if (i > 0x3fe00000) { // |z| > 0.5
        n              = j + (0x00100000 >> (k + 1));
        k              = ((n & 0x7fffffff) >> 20) - 0x3ff;
        const double t = from_words(n & ~(0x000fffff >> k), 0);
        n              = ((n & 0x000fffff) | 0x00100000) >> (20 - k);
        if (j < 0) n = -n;
        p_h -= t;
    }
    double       t  = with_low_word(p_l + p_h, 0);
    const double u  = t * lg2_h;
    const double v  = (p_l - (t - p_h)) * lg2 + t * lg2_l;
    z               = u + v;
    const double w  = v - (z - u);
    t               = z * z;
    const double t3 = z - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    const double r  = (z * t3) / (t3 - 2.0) - (w + z * w);
    z               = 1.0 - (r - z);
    j               = high_word(z) + (int32_t)((uint32_t)n << 20);
    if ((j >> 20) <= 0)
        z = ldexp(z, n);
    else
        z = with_high_word(z, j);
    return s * z;
}

constexpr double PI = 3.14159265358979323846;

HD inline double radius_of_ball(double volume) {
    if constexpr (config::dim == 2)
        return sqrt(volume / PI);
    else
        return cube_root(3.0 * volume / (4.0 * PI));
}
