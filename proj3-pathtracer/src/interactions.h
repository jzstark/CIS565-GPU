#pragma once

#include "intersections.h"
#include <glm/glm.hpp>

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);


    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
        PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    

    thrust::uniform_real_distribution<float> u01(0, 1);

    // Offset the origin to avoid self-intersection
    pathSegment.ray.origin = intersect + 0.001f * normal;

    // Color
    //float lightTerm = glm::dot(normal, glm::vec3(0.0f, 1.0f, 0.0f));
    //pathSegment.color *= (m.color * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * m.color) * 0.7f;
    //pathSegment.color *= u01(rng);

    float cosine_weight = glm::dot(pathSegment.ray.direction, normal);

    // If the material is emissive, the path should terminate
    if (m.emittance > 0.0f) {
        pathSegment.remainingBounces = 0;
        return;
    }

    float probDiffuse = 1.0f;
    float probSpecular = 0.0f;
    float probRefractive = 0.0f;

    if (m.hasReflective > 0.0f) {
        probSpecular = 0.5f;
        probDiffuse = 0.5f;
    }
    if (m.hasRefractive > 0.0f) {
        probRefractive = 0.5f;
        probDiffuse = 0.5f;
    }

    float xi = u01(rng);

    // Specular reflection  反射
    if (m.hasReflective > 0.0f && xi < probSpecular) {
        glm::vec3 incident = glm::normalize(pathSegment.ray.direction);
        glm::vec3 reflected = glm::reflect(incident, normal);
        pathSegment.ray.direction = reflected;

        pathSegment.color *= m.specular.color / probSpecular;
    }
    // Refractive (glass) 折射
    else if (m.hasRefractive > 0.0f && xi < probRefractive) {
        glm::vec3 incident = glm::normalize(pathSegment.ray.direction);
        float eta = m.indexOfRefraction;
        float cosi = glm::dot(incident, normal);
        float etai = 1.0f, etat = eta;
        glm::vec3 n = normal;
        if (cosi > 0) {
            n = -normal;
            etai = eta;
            etat = 1.0f;
        }
        float etaRatio = etai / etat;
        float k = 1.0f - etaRatio * etaRatio * (1.0f - cosi * cosi);
        if (k < 0.0f) {
            // Total internal reflection
            pathSegment.ray.direction = glm::reflect(incident, normal);
        }
        else {
            pathSegment.ray.direction = glm::normalize(etaRatio * incident + (etaRatio * cosi - sqrtf(k)) * n);
        }
        pathSegment.color *= m.color / probRefractive;
    }
    // Diffuse -- 散射
    else {
		float f_r = 1.0f; // / PI; -- if not adding the Pi part, the generated figure will look quite nice, not dark
        pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
        pathSegment.color *= m.color * f_r / probDiffuse;
    }

    // pathSegment.color *= cosine_weight;

    pathSegment.remainingBounces--;
}
