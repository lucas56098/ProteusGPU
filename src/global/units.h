#ifndef UNITS_H
#define UNITS_H
#pragma once

// Code units and the factors that turn them into cgs.

// set once in begrun from the param file, all 1 without ASTRO_PHYSICS
struct Units {
    // cgs value of one code length, mass and velocity
    double UnitLength_in_cm         = 1.0;
    double UnitMass_in_g            = 1.0;
    double UnitVelocity_in_cm_per_s = 1.0;

    void set_base(double length_in_cm, double mass_in_g, double velocity_in_cm_per_s) {
        UnitLength_in_cm         = length_in_cm;
        UnitMass_in_g            = mass_in_g;
        UnitVelocity_in_cm_per_s = velocity_in_cm_per_s;
    }

    // derived factors
    double UnitTime_in_s() const { return UnitLength_in_cm / UnitVelocity_in_cm_per_s; }
    double UnitDensity_in_cgs() const {
        return UnitMass_in_g / (UnitLength_in_cm * UnitLength_in_cm * UnitLength_in_cm);
    }

    // gravitational constant in code units
    double G_in_code_units() const {
        constexpr double G_cgs = 6.67430e-8;
        const double     T     = UnitTime_in_s();
        return G_cgs * UnitMass_in_g * T * T / (UnitLength_in_cm * UnitLength_in_cm * UnitLength_in_cm);
    }
};

extern Units units; // defined in globals.cu

#endif
