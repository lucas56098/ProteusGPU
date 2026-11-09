#pragma once

// load, store and the prediction of gradients.h (included by gradients.h)

namespace gradients {

    HD inline gas::Primitive rate_of_change(const gas::Primitive& w, const Gradient& gradient);

    HD inline Gradient load(const GradientPointers& gradients, int cell) {
        Gradient value;
        value.density = gradients.density[cell];
        for (int d = 0; d < config::dim; d++)
            value.velocity[d] = gradients.velocity[d][cell];
        value.pressure = gradients.pressure[cell];
        value.anchor   = gradients.anchor[cell];
        return value;
    }

    HD inline void store(const GradientPointers& gradients, int cell, const Gradient& value) {
        gradients.density[cell] = value.density;
        for (int d = 0; d < config::dim; d++)
            gradients.velocity[d][cell] = value.velocity[d];
        gradients.pressure[cell] = value.pressure;
        gradients.anchor[cell]   = value.anchor;
    }

    HD inline gas::Primitive predicted(const gas::Primitive& w, const Gradient& gradient, const Vec& r, double dt) {
        gas::Primitive p;
        p.density = w.density + dot(gradient.density, r);
        for (int d = 0; d < config::dim; d++)
            p.velocity[d] = w.velocity[d] + dot(gradient.velocity[d], r);
        p.pressure = w.pressure + dot(gradient.pressure, r);

        if (dt != 0.0) {
            const gas::Primitive rate = rate_of_change(w, gradient);
            p.density += dt * rate.density;
            p.velocity += dt * rate.velocity;
            p.pressure += dt * rate.pressure;
        }
        return (p.density > 0.0 && p.pressure > 0.0) ? p : w;
    }

    // rates from the Euler equations; a moving mesh leaves out v . grad, which the offset to the point carries
    HD inline gas::Primitive rate_of_change(const gas::Primitive& w, const Gradient& gradient) {
        double divergence = 0.0;
        for (int d = 0; d < config::dim; d++)
            divergence += gradient.velocity[d][d];

        gas::Primitive rate;
        if constexpr (config::moving_mesh) {
            rate.density = -w.density * divergence;
            for (int d = 0; d < config::dim; d++)
                rate.velocity[d] = -gradient.pressure[d] / w.density;
            rate.pressure = -config::adiabatic_index * w.pressure * divergence;
        } else {
            rate.density = -(dot(w.velocity, gradient.density) + w.density * divergence);
            for (int d = 0; d < config::dim; d++)
                rate.velocity[d] = -dot(w.velocity, gradient.velocity[d]) - gradient.pressure[d] / w.density;
            rate.pressure = -(dot(w.velocity, gradient.pressure) + config::adiabatic_index * w.pressure * divergence);
        }
        return rate;
    }

} // namespace gradients
