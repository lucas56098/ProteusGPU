#ifndef UNITS_H
#define UNITS_H
#pragma once

// Code-unit system. The simulation itself runs dimensionless; this type converts
// between code units and cgs at the edges only (param load, physics constants,
// snapshot metadata). Base units are length, mass and velocity (AREPO convention):
// the primitive variables map onto them with no leftover — rho = mass/length^3,
// v = velocity, E = mass*velocity^2/length^3.
//
// A factor named Unit<X>_in_cgs is "the cgs value of one code unit of X".

struct Units {
    // base factors. Defaults give a trivial system where code units == cgs.
    double UnitLength_in_cm         = 1.0;
    double UnitMass_in_g            = 1.0;
    double UnitVelocity_in_cm_per_s = 1.0;

    void set_base(double length_in_cm, double mass_in_g, double velocity_in_cm_per_s) {
        UnitLength_in_cm         = length_in_cm;
        UnitMass_in_g            = mass_in_g;
        UnitVelocity_in_cm_per_s = velocity_in_cm_per_s;
    }

    // derived factors, computed on demand from the three base ones
    double UnitTime_in_s() const { return UnitLength_in_cm / UnitVelocity_in_cm_per_s; }
    double UnitDensity_in_cgs() const {
        return UnitMass_in_g / (UnitLength_in_cm * UnitLength_in_cm * UnitLength_in_cm);
    }
    double UnitPressure_in_cgs() const {
        return UnitDensity_in_cgs() * UnitVelocity_in_cm_per_s * UnitVelocity_in_cm_per_s;
    }
    double UnitEnergy_in_cgs() const { return UnitMass_in_g * UnitVelocity_in_cm_per_s * UnitVelocity_in_cm_per_s; }

    // Newton's constant expressed in code units (G_cgs has units cm^3 g^-1 s^-2)
    double G_in_code_units() const {
        constexpr double G_cgs = 6.67430e-8;
        const double     T      = UnitTime_in_s();
        return G_cgs * UnitMass_in_g * T * T / (UnitLength_in_cm * UnitLength_in_cm * UnitLength_in_cm);
    }
};

extern Units units;

#endif // UNITS_H
