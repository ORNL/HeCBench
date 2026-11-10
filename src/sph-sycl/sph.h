#ifndef SPH_
#define SPH_

#ifndef NUM_FLUID_PARTICLES
#define NUM_FLUID_PARTICLES 2048
#endif

// Three doubles. sycl::vec<double, 3> is stored as four, which pads these particles.
struct vec3d {
    double x;
    double y;
    double z;
};

static_assert(sizeof(vec3d) == 3 * sizeof(double));
static_assert(alignof(vec3d) == alignof(double));

////////////////////////////////////////////////
// Structures
////////////////////////////////////////////////

struct boundary_particle {
    vec3d pos; // position
    vec3d n;   // position
} ;

struct fluid_particle {
    double density;
    double pressure;
    vec3d pos;     // position
    vec3d v;       // velocity
    vec3d v_half;  // half step velocity
    vec3d a;       // acceleration
};

struct param {
    double rest_density;
    double mass_particle;
    double spacing_particle;
    double smoothing_radius;
    double g;
    double time_step;
    double alpha;
    double surface_tension;
    double speed_sound;
    int number_particles;
    int number_fluid_particles;
    int number_boundary_particles;
    int number_steps;
    int steps_per_frame;
}; // Simulation paramaters

struct AABB {
    double min_x;
    double max_x;
    double min_y;
    double max_y;
    double min_z;
    double max_z;
} ; //Axis aligned bounding box



////////////////////////////////////////////////
// Function prototypes
////////////////////////////////////////////////

void constructBoundaryBox(boundary_particle *boundary_particles, AABB* boundary, param *params);
void eulerStart(fluid_particle* fluid_particles, boundary_particle *boundary_particles, param *params);
void initParticles(fluid_particle** fluid_particles, boundary_particle** boundary_particles, AABB* water, AABB* boundary, param* params);
void initParams(AABB* water_volume, AABB* boundary_volume, param* params);
void finalizeParticles(fluid_particle *fluid_particles, boundary_particle *boundary_particles);
void writeFile(fluid_particle *particles, param *params);
void writeBoundaryFile(boundary_particle *boundary, param *params);


#endif

