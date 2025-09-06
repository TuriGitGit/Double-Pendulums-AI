// Simulations.c

// Uses RK4 with fixed timestep.

// WARNING: SOME FLAGS USED '-fno-trapping-math' '-fno-signaling-nans' CAN CAUSE MATH ERRORS LIKE DIV BY 0 TO GO UNCHECKED
// ONLY USE THEM AFTER TESTING WITHOUT THEM TO CONFIRM THE CODE IS SAFE

// '-ffast-math' can cause rounding errors and reduced accuracy in effort to be fast
// using '-fno-unsafe-math-optimizations' is recommended to properly avoid this

// Output: "Simulated {} double pendulums in {} seconds"

#define _GNU_SOURCE // ignored if not using linux, allows the use of GNU's optimised sincos() function with math.h

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

#if !defined(__GNU_LIBRARY__) // slower but universal sincos function that works on windows and mac. this is about 20-30% slower... fuck microsoft and apple.
static inline void portable_sincos(double x, double *s, double *c) {
    *s = sin(x);
    *c = cos(x);
}
#define sincos portable_sincos
#endif


// changeable variables

    // bytes of buffer per thread. make it larger if your disc write speed is slow and your RAM can handle it, smaller if your RAM cannot handle it
    const size_t buffer_size = 1.5 * 1024 * 1024 * 1024; // note: 'buffer_size' times the env var 'OMP_NUM_THREADS' equals RAM allocated. for me this is 1.5GB * 12 = 18GB

    // how many simulations to run
    const long long int SIMS = 400*1000*1000; // ~68.28MB per million simulations, fit to less than your RAM size

    // how many RK4 steps per second of simulation, more steps = better accuracy. 300 accumulates an avg of 0.000001 meters of error per second.
    const int STEPS = 300;

    // base seed. note: for a truly deterministic CSV file to be created you must set env var OMP_NUM_THREADS=1 or use a different buffer method.
    int SEED = 1;
        // for ease in creating chunked simulations the SEED can be input as an arg like so:
            // for i in {1..5}; do ./simulations $i; done       (linux / mac)
            // 1..5 | ForEach-Object { .\simulations.exe $_ }   (windows)


const double DT = 1.0 / STEPS; 
const double HALFDT = DT / 2.0;
const double SIXTHDT = DT / 6.0;

const double G = 9.8; // plan to make these variable, but i will need a larger network and much more training time and data. so it may have to wait till i get a better cpu + gpu
const double M1 = 1.0;
const double M2 = 1.0;
const double M1_M2 = M1 + M2;
const double M1_M2_G = M1_M2 * G;

typedef struct {
    double m_total_g; 
    double M2_l1;
    double M2_l2;
    double m_total_l;
    double l1;
    double l2;
    double l2_div_l1;
} PendulumConsts;

static inline void derivs(const double *restrict s, double *restrict dsdt, const PendulumConsts *consts) {
    // precompute
    double si, c;
    sincos(s[2] - s[0], &si, &c);
    const double sin_s2 = sin(s[2]);
    const double m_total_g_sin_s0 = consts->m_total_g * sin(s[0]);
    const double M2_l2_s3_squared_si = consts->M2_l2 * s[3] * s[3] * si;
    const double s1_squared_si = s[1] * s[1] * si;
    const double den1 = consts->m_total_l - consts->M2_l1 * c * c;

    dsdt[0] = s[1];
    dsdt[1] = (
        consts->M2_l1 * s1_squared_si * c +
        M2 * G * sin_s2 * c +
        M2_l2_s3_squared_si -
        m_total_g_sin_s0
    ) / den1;

    dsdt[2] = s[3];
    dsdt[3] = (
        - M2_l2_s3_squared_si * c +
        m_total_g_sin_s0 * c -
        consts->m_total_l * s1_squared_si -
        consts->m_total_g * sin_s2
    ) / (consts->l2_div_l1 * den1);
}

static inline double clampAngle(double theta) {
    return theta - 2.0 * M_PI * round(theta / (2.0 * M_PI)); // [-pi, pi]
}

static inline void RK4Step(double *restrict s, const PendulumConsts *consts) {
    double k1[4], k2[4], k3[4], k4[4], tmp[4];

    derivs(s, k1, consts);
    tmp[0] = s[0] + HALFDT * k1[0];
    tmp[1] = s[1] + HALFDT * k1[1];
    tmp[2] = s[2] + HALFDT * k1[2];
    tmp[3] = s[3] + HALFDT * k1[3];

    derivs(tmp, k2, consts);
    tmp[0] = s[0] + HALFDT * k2[0];
    tmp[1] = s[1] + HALFDT * k2[1];
    tmp[2] = s[2] + HALFDT * k2[2];
    tmp[3] = s[3] + HALFDT * k2[3];

    derivs(tmp, k3, consts);
    tmp[0] = s[0] + DT * k3[0];
    tmp[1] = s[1] + DT * k3[1];
    tmp[2] = s[2] + DT * k3[2];
    tmp[3] = s[3] + DT * k3[3];

    derivs(tmp, k4, consts);
    s[0] += SIXTHDT * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0]);
    s[1] += SIXTHDT * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1]);
    s[2] += SIXTHDT * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2]);
    s[3] += SIXTHDT * (k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3]);
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
    
    const int nsteps = 10 + 10 * (int)(randu01(rng_state) * (4 * STEPS) / 10);  // [10, 1200] step 10
    const double l1 = 0.4 + 0.01 * (int)(61 * randu01(rng_state));              // [0.4, 1.0] step 0.01
    const double l2 = 0.4 + 0.01 * (int)(61 * randu01(rng_state));              // [0.4, 1.0] step 0.01
    const double theta1 = randu01(rng_state) * 2.0 * M_PI;                      // [0, 2pi)
    const double theta2 = randu01(rng_state) * 2.0 * M_PI;                      // [0, 2pi)
    const double omega1 = 0.0;                                                  // will make variable later
    const double omega2 = 0.0;                                                  // will make variable later
    const double t = (double)nsteps / STEPS;                                    // [0.033..., 2] step 0.033...
    double s[4] = {theta1, omega1, theta2, omega2};

    PendulumConsts consts = {
        .m_total_g = M1_M2_G,
        .M2_l1 = M2 * l1,
        .M2_l2 = M2 * l2,
        .m_total_l = M1_M2 * l1,
        .l1 = l1,
        .l2 = l2,
        .l2_div_l1 = l2 / l1
    };

    double sin_theta1, cos_theta1;
    sincos(theta1, &sin_theta1, &cos_theta1);
    double sin_theta2, cos_theta2;
    sincos(theta2, &sin_theta2, &cos_theta2);

    for (int i = 0; i < nsteps; ++i) {
        RK4Step(s, &consts);
        if ((i+1) % STEPS == 0) {
            s[0] = clampAngle(s[0]);
            s[2] = clampAngle(s[2]);
        }
    }
    s[0] = clampAngle(s[0]);
    s[2] = clampAngle(s[2]);

    double sin_s0, cos_s0, sin_s2, cos_s2;
    sincos(s[0], &sin_s0, &cos_s0);
    sincos(s[2], &sin_s2, &cos_s2);
    const double x1_end = l1 * sin_s0;
    const double y1_end = -l1 * cos_s0;
    const double x2_end = l2 * sin_s2 + x1_end;
    const double y2_end = -l2 * cos_s2 + y1_end;

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
    snprintf(filename, sizeof(filename), "%lld simulations at %d per second using %d seed.csv", SIMS, STEPS, SEED);
    
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

    const size_t buffer_buf = buffer_size - 512;

    char *buffer = (char*)malloc(buffer_size);
    if (!buffer) {
        fprintf(stderr, "Thread %d: malloc failed\n", tid);
        exit(1);
    }
    size_t offset = 0;

    unsigned long long rng_state = SEED ^ (0x9E3779B97F4A7C15ULL * (unsigned long long)(tid+1));

#pragma omp for schedule(dynamic, 50)
    for (long i = 0; i < SIMS; ++i) {
        unsigned long long sim_seed = rng_state ^ (0xBF58476D1CE4E5B9ULL * (unsigned long long)(i+1));
        double sin_th1, cos_th1, sin_th2, cos_th2, l1, l2, t, x2e, y2e;
        runSim(&sim_seed, &sin_th1, &cos_th1, &sin_th2, &cos_th2, &l1, &l2, &t, &x2e, &y2e);

        int written = snprintf(buffer + offset, buffer_size - offset,
                               "%.5f,%.5f,%.5f,%.5f,%.2f,%.2f,%.4f,%.5f,%.5f\n",
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
    printf("Simulated %lld double pendulums in %.3f seconds.\n", SIMS, end_time - start_time);

    FILE *f = fopen("config.json", "w");
    if (f == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"SIMS\": %lld,\n", SIMS);
    fprintf(f, "  \"STEPS\": %d,\n", STEPS);
    fprintf(f, "  \"SEED\": %d\n", SEED);
    fprintf(f, "}\n");

    fclose(f);
    
    return 0;
}
