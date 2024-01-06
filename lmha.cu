#include <algorithm>


// Linear Multi-head Attention Parameter
template< typename T >
struct Lmha_params {

    // B: batch size
    // L: timesteps sequence length
    // H: number of head
    // E: feature dimentions of query tensor
    // M: feature dimentions of value tensor
    int B, L, H, E, M;

    // The output buffer. Dimensions [B, H, L, M]
    T *out;

    // The input query tensor ptr. Dimensions [B, H, L, E]
    const T *q;
    // The input query tensor ptr. Dimensions [B, H, L, E]
    const T *k;
    // The input value tensor. Dimensions [B, H, L, M]
    const T *v;

    // The strides for the different tensors
    int q_stride_B, q_stride_H, q_stride_L;
    int k_stride_B, k_stride_H, k_stride_L;
    int v_stride_B, v_stride_H, v_stride_L;
    int o_stride_B, o_stride_H, o_stride_L;
};


static inline __device__ __host__ int div_up(int m, int n) {
    return (m + n - 1) / n;
}

template< int E, typename Params >
static inline __device__ __host__ int get_smem_buffer_size(const Params &params) {
    int M = div_up(params.M, 4) * 4; // aligned with 4 bytes
    return 2 * E + 2 * M;
}

// Linear Multi-head Attention Kernel: https://arxiv.org/pdf/2006.16236.pdf
template< int E, int THREADS_PER_HEAD, bool GO_BACKWARD >
__global__ void lmha_kernel(Lmha_params<float> params) {

    // Make sure E is a multiple of 4
    static_assert(E % 4 == 0, "");

    // The amount of shared memory per buffer (2 buffers for double-buffering)
    const int smem_buffer_size = get_smem_buffer_size<E>(params);
    // The M dimension for shared memory(aligned with 4 bytes)
    const int M = div_up(params.M, 4) * 4;

    // Shared memory to store Q, K and V. Size is 2*smem_buffer_size
    extern __shared__ float smem_[];

    // The various shared memory buffers
    float *smem_q = &smem_[0 * E];
    float *smem_k = &smem_[1 * E];
    float *smem_v = &smem_[2 * E];
    float *smem_o = &smem_[2 * E + M];

    // The index of the shared memory buffer (for double-buffering)
    int smem_curr = 0;

    // The sequence processed by that block
    const int bi = blockIdx.y;
    // The head processed by that block
    const int hi = blockIdx.x;
    // The linear index of the thread
    const int tidx = threadIdx.x;

    // The offset to the position loaded by the thread in Query
    int offset_q = bi * params.q_stride_B + hi * params.q_stride_H + tidx;
    // The offset to the position loaded by the thread in Key
    int offset_k = bi * params.k_stride_B + hi * params.k_stride_H + tidx;
    // The offset to the position loaded by the thread in Value and Output
    int offset_v = bi * params.v_stride_B + hi * params.v_stride_H + tidx;
    int offset_o = bi * params.o_stride_B + hi * params.o_stride_H + tidx;

    // If go backward, account for the extra offset, start from end to begin
    if (GO_BACKWARD) {
        offset_k += (params.L - 1) * params.k_stride_L;
        offset_v += (params.L - 1) * params.v_stride_L;
        offset_q += (params.L - 1) * params.q_stride_L;
        offset_o += (params.L - 1) * params.o_stride_L;
    }

    // Determine the base pointers for Query and Key
    const float *ptr_q = &params.q[offset_q];
    const float *ptr_k = &params.k[offset_k];
    // Determine the base pointers for Value
    const float *ptr_v = &params.v[offset_v];
    // The output pointer
    float *out_ptr = &params.out[offset_o];

    // Whether is an active thread for Query and Key
    const int active_qk = tidx < params.E;
    // Whether is an active thread for Value
    const int active_v = tidx < params.M;

    // Trigger the memory loads Query, Key and Value
    float ldg_q = active_qk ? (*ptr_q) : 0.f;
    float ldg_k = active_qk ? (*ptr_k) : 0.f;
    float ldg_v = active_v ? (*ptr_v) : 0.f;

    // Store Query and Key elements to shared memory
    if (tidx < E) {
        smem_q[smem_curr*smem_buffer_size + tidx] = ldg_q;
        smem_k[smem_curr*smem_buffer_size + tidx] = ldg_k;
    }
    // Store Value element to shared memory
    if (tidx < M) {
        smem_v[smem_curr*smem_buffer_size + tidx] = ldg_v;
    }

    // The number of FLOAT4s per head
    constexpr int FLOAT4s_PER_HEAD = E / 4;
    // The number of FLOAT4s per thread
    constexpr int FLOAT4s_PER_THREAD = FLOAT4s_PER_HEAD / THREADS_PER_HEAD;

    // The storage for the K*V^T elements
    float4 kv[FLOAT4s_PER_THREAD];
    #pragma unroll
    for (int ii = 0; ii < FLOAT4s_PER_THREAD; ++ii) {
        kv[ii] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    // Move the load pointers
    if (GO_BACKWARD) {
        ptr_q -= params.q_stride_L;
        ptr_k -= params.k_stride_L;
        ptr_v -= params.v_stride_L;
    } else {
        ptr_q += params.q_stride_L;
        ptr_k += params.k_stride_L;
        ptr_v += params.v_stride_L;
    }

    // The position of the thread in the Value dimension
    int vo = tidx / THREADS_PER_HEAD;
    int vi = tidx % THREADS_PER_HEAD;

    // Iterate over the timesteps(params.L), tensor shape: [N, H, L, E/M]
    for (int time = 0; time < params.L; ++time) {
        // Make sure the data is in shared memory
        __syncthreads();

        // Each thread loads a single Value element
        float v = 0.f;
        if (vo < params.M) {
            v = *reinterpret_cast<const float *>(&smem_v[smem_curr*smem_buffer_size + vo]);
        }

        float sum = 0.f;
        #pragma unroll
        for (int ii = 0; ii < FLOAT4s_PER_THREAD; ++ii) {
            // Each thread loads 4 Key elements from shared memory
            int idx = tidx % THREADS_PER_HEAD * 4 + ii * THREADS_PER_HEAD * 4;
            float4 key = *reinterpret_cast<const float4*>(&smem_k[smem_curr*smem_buffer_size + idx]);
            // Each thread loads 4 Query elements from shared memory
            float4 query = *reinterpret_cast<const float4*>(&smem_q[smem_curr*smem_buffer_size + idx]);

            // Update the K*V^T product
            kv[ii].x += key.x * v;
            kv[ii].y += key.y * v;
            kv[ii].z += key.z * v;
            kv[ii].w += key.w * v;

            // Compute the partial output value for that thread
            sum += query.x * kv[ii].x;
            sum += query.y * kv[ii].y;
            sum += query.z * kv[ii].z;
            sum += query.w * kv[ii].w;
        }

        // Finalize the computation of the sum (if we have more than 1 thread per head)
        if (THREADS_PER_HEAD > 1) {

            // Finalize the sum for each head
            #pragma unroll
            for (int delta = THREADS_PER_HEAD / 2; delta >= 1; delta /= 2) {
                sum += __shfl_xor_sync(uint32_t(-1), sum, delta);
            }

            // Store to shared memory
            if (vo < M && vi == 0) {
                smem_o[smem_curr*smem_buffer_size + vo] = sum;
            }

            // Make sure the data is in shared memory
            __syncthreads();

            // Active threads read the data to store
            if (active_v) {
                sum = smem_o[smem_curr*smem_buffer_size + tidx];
            }

        }

        // Store the output
        if (active_v) {
            *out_ptr = sum;
        }

        // Whether is the last iteration
        int is_last = time == params.L - 1;

        // Trigger the next loads for Query and Key
        if (!is_last && active_qk) {
            ldg_q = *ptr_q;
            ldg_k = *ptr_k;
        }

        // Trigger the next loads for Value
        if (!is_last && active_v) {
            ldg_v = *ptr_v;
        }

        // Move the next load pointers
        if (GO_BACKWARD) {
            ptr_q -= params.q_stride_L;
            ptr_k -= params.k_stride_L;
            ptr_v -= params.v_stride_L;
            out_ptr -= params.o_stride_L;
        } else {
            ptr_q += params.q_stride_L;
            ptr_k += params.k_stride_L;
            ptr_v += params.v_stride_L;
            out_ptr += params.o_stride_L;
        }

        // Move to the next shared memory buffer
        smem_curr = (smem_curr + 1) % 2;

        // Store to shared memory for Query and Key elements
        if (!is_last && tidx < E) {
            smem_q[smem_curr*smem_buffer_size + tidx] = ldg_q;
            smem_k[smem_curr*smem_buffer_size + tidx] = ldg_k;
        }

        // Store to shared memory for Value element
        if (!is_last && tidx < M) {
            smem_v[smem_curr*smem_buffer_size + tidx] = ldg_v;
        }
    }
}

template< int E, int THREADS_PER_HEAD, bool GO_BACKWARD>
int lmha_(const Lmha_params<float> &params) {
    // The M dimension rounded up to 4
    int M = div_up(params.M, 4) * 4;

    // The number of threads in the block.
    int threads_per_block = div_up(std::max(E, M*THREADS_PER_HEAD), 32) * 32;
    // Beyond range
    if (threads_per_block > 512 || params.B > 65535) {
        return 1;
    }

    // Prepare the kernel
    dim3 grid(params.H, params.B);
    size_t smem = get_smem_buffer_size<E>(params) * sizeof(float) * 2; // double buffer size
    lmha_kernel<E, THREADS_PER_HEAD, GO_BACKWARD><<<grid, threads_per_block, smem>>>(params);
    return 0;
}

template< bool GO_BACKWARD>
int lmha(const Lmha_params<float> &params) {
    int res = 1;
    if (params.E <=  32) {
        res = lmha_< 32, 1, GO_BACKWARD>(params);
    } else if (params.E <=  48) {
        res = lmha_< 48, 1, GO_BACKWARD>(params);
    } else if (params.E <=  64) {
        res = lmha_< 64, 1, GO_BACKWARD>(params);
    } else if (params.E <= 128) {
        res = lmha_<128, 2, GO_BACKWARD>(params);
    } else if (params.E <= 256) {
        res = lmha_<256, 4, GO_BACKWARD>(params);
    }
    return res;
}