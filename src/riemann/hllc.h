#pragma once

// the HLLC solver of Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics (included by riemann.h)

#include <cmath>

namespace riemann {

    HD inline gas::Flux flux_of_state(const gas::Primitive& w);
    HD inline gas::Flux star_flux(const gas::Primitive& w, const gas::Flux& flux, double wave, double contact);

    HD inline gas::Flux hllc(const gas::Primitive& left, const gas::Primitive& right) {

        const gas::Flux left_flux  = flux_of_state(left);
        const gas::Flux right_flux = flux_of_state(right);

        // left, contact and right wave
        const double v_left     = left.velocity[0];
        const double v_right    = right.velocity[0];
        const double left_wave  = fmin(v_left - gas::sound_speed(left), v_right - gas::sound_speed(right));
        const double right_wave = fmax(v_left + gas::sound_speed(left), v_right + gas::sound_speed(right));
        const double contact    = (right.pressure - left.pressure + left.density * v_left * (left_wave - v_left) -
                                right.density * v_right * (right_wave - v_right)) /
                               (left.density * (left_wave - v_left) - right.density * (right_wave - v_right));

        // the flux depends on where the face sits in the fan
        if (0.0 <= left_wave) return left_flux;
        if (contact >= 0.0) return star_flux(left, left_flux, left_wave, contact);
        if (0.0 <= right_wave) return star_flux(right, right_flux, right_wave, contact);
        return right_flux;
    }

    // flux of a state along x
    HD inline gas::Flux flux_of_state(const gas::Primitive& w) {
        const double mass_flux = w.density * w.velocity[0];
        gas::Flux    flux;
        flux.mass     = mass_flux;
        flux.momentum = mass_flux * w.velocity;
        flux.momentum[0] += w.pressure;
        flux.energy = (gas::energy_density(w) + w.pressure) * w.velocity[0];
        return flux;
    }

    HD inline gas::Flux star_flux(const gas::Primitive& w, const gas::Flux& flux, double wave, double contact) {
        const double v             = w.velocity[0];
        const double star_pressure = w.pressure + w.density * (wave - v) * (contact - v);
        const double inverse       = 1.0 / (wave - contact);

        gas::Flux star;
        star.mass = (contact * (wave * w.density - flux.mass)) * inverse;
        for (int d = 0; d < config::dim; d++)
            star.momentum[d] = contact * (wave * w.density * w.velocity[d] - flux.momentum[d]);
        star.momentum[0] += wave * star_pressure;
        star.momentum = star.momentum * inverse;
        star.energy =
            (contact * (wave * gas::energy_density(w) - flux.energy) + wave * star_pressure * contact) * inverse;
        return star;
    }

} // namespace riemann
