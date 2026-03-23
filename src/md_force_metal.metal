/*
 * Metal Compute Shader — LJ Force via Piecewise Polynomial
 *
 * Replaces the analytical LJ chain (8 cycles serial dependency)
 * with a piecewise degree-4 polynomial evaluated via Horner's method
 * (4 FMA cycles). Coefficients live in constant address space —
 * no memory fetch, baked into the shader binary.
 *
 * 16 non-uniform intervals on r² ∈ [0.64, 6.25], quadratic spacing
 * (denser at small r² where the repulsive wall is steepest).
 *
 * Interval selection: binary search replaced by direct computation
 * from the quadratic spacing formula.
 */

#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE 128

struct MDMetalParams {
    uint   n_real;
    uint   n_padded;
    float  lbox;
    float  inv_box;
    float  rc2;
    float  v_shift;
};

/* Piecewise polynomial: 16 intervals, degree 4
 * Non-uniform spacing: edges[i] = R2_MIN + (R2_MAX - R2_MIN) * (i/16)^2
 * Invert: i = 16 * sqrt((r2 - R2_MIN) / (R2_MAX - R2_MIN))
 */
#define CHEB_N 32
#define CHEB_R2_MIN 0.64f
#define CHEB_R2_MAX 6.25f
#define CHEB_SCALE 5.70071066f   /* N / (R2_MAX - R2_MIN) = 32/5.61 */

constant float cheb_c0[32] = {
    9.47944850e+02f, 1.46092631e+02f, 2.63469741e+01f, 3.40005511e+00f,
    -1.26899368e+00f, -1.93556141e+00f, -1.71947168e+00f, -1.36787961e+00f,
    -1.05532306e+00f, -8.10128015e-01f, -6.24949633e-01f, -4.86390346e-01f,
    -3.82475865e-01f, -3.03971447e-01f, -2.44099990e-01f, -1.97969746e-01f,
    -1.62058800e-01f, -1.33821148e-01f, -1.11402802e-01f, -9.34417106e-02f,
    -7.89277007e-02f, -6.71043207e-02f, -5.73997030e-02f, -4.93775147e-02f,
    -4.27018669e-02f, -3.71119901e-02f, -3.24037929e-02f, -2.84163123e-02f,
    -2.50216728e-02f, -2.21175807e-02f, -1.96216689e-02f, -1.74672022e-02f
};
constant float cheb_c1[32] = {
    -1.91489494e+03f, -2.53531532e+02f, -4.57147206e+01f, -9.41326302e+00f,
    -1.74241329e+00f, -6.16267225e-03f, 3.36909583e-01f, 3.42924060e-01f,
    2.78997079e-01f, 2.13029163e-01f, 1.59660965e-01f, 1.19467016e-01f,
    8.98930711e-02f, 6.82341909e-02f, 5.23145813e-02f, 4.05258223e-02f,
    3.17153485e-02f, 2.50654017e-02f, 1.99958161e-02f, 1.60929885e-02f,
    1.30598798e-02f, 1.06813205e-02f, 8.80001317e-03f, 7.29989425e-03f,
    6.09453443e-03f, 5.11899270e-03f, 4.32404965e-03f, 3.67208993e-03f,
    3.13413686e-03f, 2.68769836e-03f, 2.31518888e-03f, 2.00276399e-03f
};
constant float cheb_c2[32] = {
    2.00100235e+03f, 2.21982862e+02f, 3.58072464e+01f, 7.23438048e+00f,
    1.63503098e+00f, 3.58483490e-01f, 4.70129501e-02f, -2.49024249e-02f,
    -3.49816645e-02f, -3.02035121e-02f, -2.32291361e-02f, -1.71987802e-02f,
    -1.25976225e-02f, -9.23495922e-03f, -6.81164605e-03f, -5.06773689e-03f,
    -3.80696332e-03f, -2.88856582e-03f, -2.21355170e-03f, -1.71267431e-03f,
    -1.33741192e-03f, -1.05358009e-03f, -8.36916730e-04f, -6.70056902e-04f,
    -5.40461807e-04f, -4.38996015e-04f, -3.58944342e-04f, -2.95327616e-04f,
    -2.44422664e-04f, -2.03422801e-04f, -1.70195667e-04f, -1.43109058e-04f
};
constant float cheb_c3[32] = {
    -1.22325192e+03f, -1.17463126e+02f, -1.69334527e+01f, -3.17578851e+00f,
    -7.05327283e-01f, -1.71550036e-01f, -4.14382533e-02f, -7.77253385e-03f,
    7.13341046e-04f, 2.36927906e-03f, 2.24472834e-03f, 1.76119275e-03f,
    1.29951278e-03f, 9.38628457e-04f, 6.74737404e-04f, 4.86471254e-04f,
    3.53105999e-04f, 2.58510184e-04f, 1.91041620e-04f, 1.42549058e-04f,
    1.07388561e-04f, 8.16589701e-05f, 6.26549785e-05f, 4.84893344e-05f,
    3.78356272e-05f, 2.97539302e-05f, 2.35724623e-05f, 1.88069234e-05f,
    1.51051687e-05f, 1.22089891e-05f, 9.92750351e-06f, 8.11848549e-06f
};
constant float cheb_c4[32] = {
    3.35631647e+02f, 2.92917961e+01f, 3.89706418e+00f, 6.86116044e-01f,
    1.46240560e-01f, 3.53418650e-02f, 9.11367181e-03f, 2.30903264e-03f,
    4.66702525e-04f, -1.64577499e-05f, -1.17269211e-04f, -1.14965505e-04f,
    -9.05614274e-05f, -6.64179278e-05f, -4.74397577e-05f, -3.36182843e-05f,
    -2.38450574e-05f, -1.70034245e-05f, -1.22173301e-05f, -8.85530015e-06f,
    -6.47771618e-06f, -4.78272502e-06f, -3.56379939e-06f, -2.67935898e-06f,
    -2.03186033e-06f, -1.55364681e-06f, -1.19743050e-06f, -9.29889880e-07f,
    -7.27348491e-07f, -5.72842346e-07f, -4.54116474e-07f, -3.62246231e-07f
};

/* PE coefficients — precomputed separately */
/* For now, compute PE analytically (it's not on the critical path) */

kernel void lj_force_kernel(
    const device float *x       [[ buffer(0) ]],
    const device float *y       [[ buffer(1) ]],
    const device float *z       [[ buffer(2) ]],
    device float       *fx      [[ buffer(3) ]],
    device float       *fy      [[ buffer(4) ]],
    device float       *fz      [[ buffer(5) ]],
    device float       *pe_out  [[ buffer(6) ]],
    constant MDMetalParams &params [[ buffer(7) ]],
    uint tid     [[ thread_position_in_grid ]],
    uint lid     [[ thread_position_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]]
) [[max_total_threads_per_threadgroup(TILE_SIZE)]] {

    threadgroup half sx[TILE_SIZE];
    threadgroup half sy[TILE_SIZE];
    threadgroup half sz[TILE_SIZE];

    const uint   n_real  = params.n_real;
    const uint   n_pad   = params.n_padded;
    const float  box     = params.lbox;
    const float  inv_box = params.inv_box;
    const float  rc2     = params.rc2;
    const float  v_shift = params.v_shift;
    const half   h_box   = (half)box;
    const half   h_ibox  = (half)inv_box;

    const bool valid_i = (tid < n_real);
    const float xi_f = valid_i ? x[tid] : 0.0f;
    const float yi_f = valid_i ? y[tid] : 0.0f;
    const float zi_f = valid_i ? z[tid] : 0.0f;
    const half  xi_h = (half)xi_f;
    const half  yi_h = (half)yi_f;
    const half  zi_h = (half)zi_f;

    float fxi = 0.0f, fyi = 0.0f, fzi = 0.0f;
    float pei = 0.0f;

    for (uint tile_start = 0; tile_start < n_pad; tile_start += TILE_SIZE) {
        uint j_idx = tile_start + lid;
        if (j_idx < n_real) {
            sx[lid] = (half)x[j_idx];
            sy[lid] = (half)y[j_idx];
            sz[lid] = (half)z[j_idx];
        } else {
            sx[lid] = (half)1e4f;
            sy[lid] = (half)1e4f;
            sz[lid] = (half)1e4f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_i) {
            for (uint k = 0; k < TILE_SIZE; k++) {
                /* Distance in half precision */
                half dx_h = xi_h - sx[k];
                half dy_h = yi_h - sy[k];
                half dz_h = zi_h - sz[k];
                dx_h -= h_box * rint(dx_h * h_ibox);
                dy_h -= h_box * rint(dy_h * h_ibox);
                dz_h -= h_box * rint(dz_h * h_ibox);

                float dx = (float)dx_h;
                float dy = (float)dy_h;
                float dz = (float)dz_h;
                float r2 = fma(dx, dx, fma(dy, dy, dz * dz));

                /* In-range mask: not self, within cutoff, within table range */
                bool in_range = (r2 > CHEB_R2_MIN && r2 < rc2);
                float mask = select(0.0f, 1.0f, in_range);

                /*
                 * PIECEWISE POLYNOMIAL EVALUATION
                 *
                 * Interval selection from quadratic spacing:
                 *   edge[i] = R2_MIN + RANGE * (i/N)^2
                 *   i = N * sqrt((r2 - R2_MIN) / RANGE)
                 *
                 * Then local coordinate t = (r2 - edge[i]) / (edge[i+1] - edge[i])
                 *
                 * Horner: (((c4*t + c3)*t + c2)*t + c1)*t + c0
                 * = 4 FMA ops. No division. No reciprocal.
                 */
                /*
                 * Uniform spacing: idx = (r2 - R2_MIN) * N/RANGE
                 * No sqrt, no division for t (multiply by precomputed reciprocal).
                 * t = (r2 - edge[idx]) * inv_dr  where inv_dr = N/RANGE
                 */
                float safe_r2 = select(1.0f, r2, in_range);
                float scaled = (safe_r2 - CHEB_R2_MIN) * CHEB_SCALE;
                uint idx = min(uint(scaled), uint(CHEB_N - 1));
                float t = scaled - float(idx);

                /* Horner evaluation: 4 chained FMA */
                float f_over_r = fma(fma(fma(fma(
                    cheb_c4[idx], t, cheb_c3[idx]),
                    t, cheb_c2[idx]),
                    t, cheb_c1[idx]),
                    t, cheb_c0[idx]);

                f_over_r *= mask;

                fxi = fma(f_over_r, dx, fxi);
                fyi = fma(f_over_r, dy, fyi);
                fzi = fma(f_over_r, dz, fzi);

                /*
                 * PE still analytical — it's summed once per step,
                 * not on the critical path. The Horner chain for force
                 * is what matters for throughput.
                 */
                if (in_range) {
                    float inv_r2 = 1.0f / safe_r2;
                    float inv_r6 = inv_r2 * inv_r2 * inv_r2;
                    float inv_r12 = inv_r6 * inv_r6;
                    pei += 4.0f * (inv_r12 - inv_r6) - v_shift;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_i) {
        fx[tid]     = fxi;
        fy[tid]     = fyi;
        fz[tid]     = fzi;
        pe_out[tid] = pei * 0.5f;
    }
}
