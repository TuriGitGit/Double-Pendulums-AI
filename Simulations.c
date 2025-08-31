// Simulations.c

// Uses RK4 with fixed timestep.

// Build (OpenMP):
//  export OMP_NUM_THREADS={threads}
// note: you can effectively disable multi-threading with export OMP_NUM_THREADS=1, otherwise allocate as many threads as you wish
//
//  gcc -O3 -march=native -ffast-math -fno-unsafe-math-optimizations -funroll-loops -fno-trapping-math -fno-signaling-nans -fopenmp Simulations.c -lm -o simulations
// WARNING SOME FLAGS USED '-fno-trapping-math' '-fno-signaling-nans' CAN CAUSE MATH ERRORS LIKE DIV BY 0 TO GO UNCHECKED
// ONLY USE THEM AFTER TESTING WITHOUT THEM TO CONFIRM THE CODE IS SAFE
// note: '-ffast-math' can cause rounding errors and reduced accuracy in effort to be fast
// using '-fno-unsafe-math-optimizations' is recommended to properly avoid this

// Run:
// ./simulations

// Output: "Simulated {} double pendulums in {} seconds"

#define _GNU_SOURCE // currently means it only works on linux systems, later ill make it portable and include the slower but windows and mac friendly functions

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// changeable variables
const long int N = 500*1000*1000; // how many simulations to run

const int STEPS = 300; // how many RK4 steps per simulation, more steps = better accuracy. 300 accumulates an avg of 0.000001 meters of error.

int SEED = 1; // what base seed to use. note: for a truly deterministic CSV file to be created you must 'export OMP_NUM_THREADS=1' or use a different buffer method.
// for ease in creating chunked simulations the SEED can be input as an arg like so:
// for seed in {1..5}; do ./simulations $seed; done
// this is useful if you want to train on more data then can fit in your RAM and you dont want to deal with pythons slow chunking shenanigans


const double DT = 1.0 / STEPS; 
const double HALFDT = DT / 2;
const double SIXTHDT = DT / 6.0;

const double G = 9.8; // plan to make these variable, but i will need a larger network and much more training time and data. so it may have to wait till i get a better cpu + gpu
const double M1 = 1.0;
const double M2 = 1.0;


static inline void derivs(const double *restrict s, double *restrict dsdt, double l1, double l2) {
    const double delta = s[2] - s[0];
    double si, c;
    sincos(delta, &si, &c);
    const double sin_s2 = sin(s[2]);
    const double s1_squared = s[1] * s[1];
    // precompute
    const double m_total = M1 + M2;
    const double m_total_g = m_total * G;
    const double M2_l1 = M2 * l1;
    const double M2_l2_s3_squared = M2 * l2 * s[3] * s[3];
    const double m_total_1 = m_total * l1;
    const double m_total_g_sin_s0 = m_total_g * sin(s[0]);
    const double M2_l2_s3_squared_si = M2_l2_s3_squared * si;
    const double s1_squared_si = s1_squared * si;
    const double den1 = m_total_1 - M2_l1 * c * c;
    const double den2 = (l2 / l1) * den1;

    dsdt[0] = s[1];
    dsdt[1] = (
        M2_l1 * s1_squared_si * c +
        M2 * G * sin_s2 * c +
        M2_l2_s3_squared_si -
        m_total_g_sin_s0
    ) / den1;

    dsdt[2] = s[3];
    dsdt[3] = (
        - M2_l2_s3_squared_si * c +
        m_total_g_sin_s0 * c -
        m_total_1 * s1_squared_si -
        m_total_g * sin_s2
    ) / den2;
}

static inline double clampAngle(double theta) {
    return theta - 2.0 * M_PI * round(theta / (2.0 * M_PI));
}

static inline void RK4Step(double *restrict s, double l1, double l2) {
    // the compiler can't be trusted to always unroll this function properly, so it is manually unrolled.
    double k1[4], k2[4], k3[4], k4[4], tmp[4];

    derivs(s, k1, l1, l2);
    tmp[0] = s[0] + HALFDT * k1[0];
    tmp[1] = s[1] + HALFDT * k1[1];
    tmp[2] = s[2] + HALFDT * k1[2];
    tmp[3] = s[3] + HALFDT * k1[3];

    derivs(tmp, k2, l1, l2);
    tmp[0] = s[0] + HALFDT * k2[0];
    tmp[1] = s[1] + HALFDT * k2[1];
    tmp[2] = s[2] + HALFDT * k2[2];
    tmp[3] = s[3] + HALFDT * k2[3];

    derivs(tmp, k3, l1, l2);
    tmp[0] = s[0] + DT * k3[0];
    tmp[1] = s[1] + DT * k3[1];
    tmp[2] = s[2] + DT * k3[2];
    tmp[3] = s[3] + DT * k3[3];

    derivs(tmp, k4, l1, l2);
    s[0] += SIXTHDT * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0]);
    s[1] += SIXTHDT * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1]);
    s[2] += SIXTHDT * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2]);
    s[3] += SIXTHDT * (k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3]);

    //s[0] = clampAngle(s[0]); clamping every step could potentially help in cases where the pendulum loops around hundreds of times, but this isnt realistically a concern
    //s[2] = clampAngle(s[2]); so instead move clamp outside just so that the NN's input is nicer.
}


static inline unsigned long long splitMix64(unsigned long long *x) {
    unsigned long long z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline double randu01(unsigned long long *state) {
    const unsigned long long r = splitMix64(state);
    return (r >> 11) * (1.0/9007199254740992.0);
}


static inline void runSim(unsigned long long *rng_state,
                           double *out_sin_theta1, double *out_cos_theta1,
                           double *out_sin_theta2, double *out_cos_theta2,
                           double *out_l1, double *out_l2,
                           double *out_t,
                           double *out_x2_end, double *out_y2_end) {
    
    const double t = 0.2 * (1+ (int)(10 * randu01(rng_state))); // range [0.2, 2.0] step 0.2 at some point i will have to make these not stepped or atleast not stepped so harshly but then i have to deal with nsteps being a double/float instead of an int... not fun
    const int nsteps = t / DT; // currently doing this is fine since of how t is stepped, but i will likely have to implement rounding and increase DT later.
    const double l1 = 0.4 + 0.01 * (int)(61 * randu01(rng_state)); // range [0.4, 1.0] step 0.01
    const double l2 = 0.4 + 0.01 * (int)(61 * randu01(rng_state)); // range [0.4, 1.0] step 0.01
    const double theta1 = randu01(rng_state) * 2.0 * M_PI;
    const double theta2 = randu01(rng_state) * 2.0 * M_PI;
    const double omega1 = 0.0;
    const double omega2 = 0.0;
    double s[4] = {theta1, omega1, theta2, omega2};

    double sin_theta1, cos_theta1;
    sincos(theta1, &sin_theta1, &cos_theta1);
    double sin_theta2, cos_theta2;
    sincos(theta2, &sin_theta2, &cos_theta2);

    for (int i = 0; i < nsteps; ++i) {
        RK4Step(s, l1, l2);
    }
    s[0] = clampAngle(s[0]);
    s[2] = clampAngle(s[2]);

    double sin_s0, cos_s0;
    sincos(s[0], &sin_s0, &cos_s0);
    double sin_s2, cos_s2;
    sincos(s[2], &sin_s2, &cos_s2);
    const double x1_end = l1 * sin_s0;
    const double y1_end = -l1 * cos_s0;
    const double x2_end = l2 * sin_s2 + x1_end;
    const double y2_end = -l2 * cos_s2 + y1_end;

    // Output
    *out_sin_theta1 = sin_theta1;
    *out_cos_theta1 = cos_theta1;
    *out_sin_theta2 = sin_theta2;
    *out_cos_theta2 = cos_theta2;
    *out_l1 = l1;
    *out_l2 = l2;
    *out_t = t;
    *out_x2_end = x2_end;
    *out_y2_end = y2_end;
}


int main(int argc, char** argv) {
    if (argc >= 2) {SEED = atoi(argv[1]);}
    char filename[128];
    snprintf(filename, sizeof(filename), "%ld simulations at %.5f using %d.csv", N, DT, SEED);
    
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("fopen");
        return 1;
    }
    fprintf(fp, "sin_theta1,cos_theta1,sin_theta2,cos_theta2,l1,l2,t,x2_end,y2_end\n");

    double start_time = 0.0, end_time = 0.0;
#ifdef _OPENMP
    start_time = omp_get_wtime();
#else
    start_time = (double)clock() / CLOCKS_PER_SEC;
#endif

#pragma omp parallel
{
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif

    // 1.5 GB buffer per thread. make it larger if your write speed is slow and your RAM can handle it, smaller if your RAM cannot handle it
    const size_t buffer_size = 1.5 * 1024 * 1024 * 1024;
    const size_t buffer_buf = buffer_size - 512;

    char *buffer = (char*)malloc(buffer_size);
    if (!buffer) {
        fprintf(stderr, "Thread %d: malloc failed\n", tid);
        exit(1);
    }
    size_t offset = 0;

    unsigned long long rng_state = SEED ^ (0x9E3779B97F4A7C15ULL * (unsigned long long)(tid+1));

#pragma omp for schedule(static)
    for (long i = 0; i < N; ++i) {
        unsigned long long sim_seed = rng_state ^ (0xBF58476D1CE4E5B9ULL * (unsigned long long)(i+1));
        double sin_th1, cos_th1, sin_th2, cos_th2, l1, l2, t, x2e, y2e;
        runSim(&sim_seed, &sin_th1, &cos_th1, &sin_th2, &cos_th2, &l1, &l2, &t, &x2e, &y2e);

        // fppend to buffer
        int written = snprintf(buffer + offset, buffer_size - offset,
                               "%.5f,%.5f,%.5f,%.5f,%.2f,%.2f,%.1f,%.5f,%.5f\n",
                               sin_th1, cos_th1, sin_th2, cos_th2, l1, l2, t, x2e, y2e);

        if (written < 0) {
            fprintf(stderr, "Thread %d: snprintf error\n", tid);
            exit(1);
        }
        offset += written;

        if (offset >= buffer_buf) {
#pragma omp critical
            {
                fwrite(buffer, 1, offset, fp);
            }
            offset = 0;
        }
    }

    if (offset > 0) {
#pragma omp critical
        {
            fwrite(buffer, 1, offset, fp);
        }
    }
    free(buffer);
}

#ifdef _OPENMP
    end_time = omp_get_wtime();
#else
    end_time = (double)clock() / CLOCKS_PER_SEC;
#endif
    fclose(fp);
    printf("Simulated %ld double pendulums in %.3f seconds.\n", N, end_time - start_time);
    return 0;
}
