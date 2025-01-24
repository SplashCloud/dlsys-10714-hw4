#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
#define BLOCK_TILE 16
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

CudaDims CudaTwoDim(size_t row, size_t col) {
  CudaDims dim;
  size_t base_thread_num = 16;
  size_t num_blocks_x = (row + base_thread_num - 1) / base_thread_num;
  size_t num_blocks_y = (col + base_thread_num - 1) / base_thread_num;
  dim.block = dim3(base_thread_num, base_thread_num, 1);
  dim.grid = dim3(num_blocks_x, num_blocks_y, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides

__device__ CudaVec GetStridesFromShape(const CudaVec* shape) {
  CudaVec strides;
  strides.size = shape->size;
  strides.data[strides.size - 1] = 1;
  for (int i = strides.size - 2; i >= 0; i--) {
    strides.data[i] = strides.data[i+1] * shape->data[i+1];
  }
  return strides;
}

// compact physical location
//    -> compact logical index (non-compact logical index)
//                -> non-compact physical location
__device__ size_t CmptPhyLocToNonCmptPhyLoc(size_t cmpt_phy_loc, CudaVec shape, CudaVec strides, size_t offset) {
  // get the strides of the *out* from shape
  CudaVec out_strides = GetStridesFromShape(&shape);
  // compact physical location convert to (non-)compact logical index
  CudaVec logical_index;
  logical_index.size = out_strides.size;
  for (size_t i = 0; i < logical_index.size; i++) {
    logical_index.data[i] = cmpt_phy_loc / out_strides.data[i];
    cmpt_phy_loc = cmpt_phy_loc % out_strides.data[i];
  }
  // calc the non-compact physical location
  size_t non_cmpt_phy_loc = 0;
  for (size_t i = 0; i < logical_index.size; i++) {
    non_cmpt_phy_loc += logical_index.data[i] * strides.data[i];
  }
  non_cmpt_phy_loc += offset;
  return non_cmpt_phy_loc;
}

__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid < size) {
    out[gid] = a[CmptPhyLocToNonCmptPhyLoc(gid, shape, strides, offset)];
  }
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[CmptPhyLocToNonCmptPhyLoc(gid, shape, strides, offset)] = a[gid];
  }
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                          VecToCuda(strides), offset);
  /// END SOLUTION
}

__global__ void ScalarSetitemKernel(const scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[CmptPhyLocToNonCmptPhyLoc(gid, shape, strides, offset)] = val;
  }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape), VecToCuda(strides), offset);
  /// END SOLUTION
}


/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */


////////////////////////////////////////////////////////////////////////////////
// Kernel Function MACRO
////////////////////////////////////////////////////////////////////////////////

#define DEFINE_EWISE_BINARY_KERNEL(OPERATION) \
__global__ void Ewise##OPERATION##Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) { \
    out[gid] = OPERATION(a[gid], b[gid]); \
  } \
}

#define DEFINE_EWISE_UNARY_KERNEL(OPERATION) \
__global__ void Ewise##OPERATION##Kernel(const scalar_t* a, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) { \
    out[gid] = OPERATION(a[gid]); \
  } \
}

#define DEFINE_SCALAR_KERNEL(OPERATION) \
__global__ void Scalar##OPERATION##Kernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) { \
    out[gid] = OPERATION(a[gid], val); \
  } \
}

////////////////////////////////////////////////////////////////////////////////
// Host Function MACRO
////////////////////////////////////////////////////////////////////////////////

#define DEFINE_EWISE_BINARY(OPERATION) \
void Ewise##OPERATION(const CudaArray& a, const CudaArray& b, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Ewise##OPERATION##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size); \
}

#define DEFINE_EWISE_UNARY(OPERATION) \
void Ewise##OPERATION(const CudaArray& a, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Ewise##OPERATION##Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size); \
}

#define DEFINE_SCALAR(OPERATION) \
void Scalar##OPERATION(const CudaArray& a, scalar_t val, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Scalar##OPERATION##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size); \
}

////////////////////////////////////////////////////////////////////////////////
// Device Operators
////////////////////////////////////////////////////////////////////////////////

__device__ scalar_t Add(scalar_t a, scalar_t b) {
  return a + b;
}

__device__ scalar_t Mul(scalar_t a, scalar_t b) {
  return a * b;
}

__device__ scalar_t Div(scalar_t a, scalar_t b) {
  return a / b;
}

__device__ scalar_t Power(scalar_t a, scalar_t b) {
  return pow(a, b);
}

__device__ scalar_t Maximum(scalar_t a, scalar_t b) {
  return a > b ? a : b;
}

__device__ scalar_t Eq(scalar_t a, scalar_t b) {
  return a == b ? 1.0 : 0.0;
}

__device__ scalar_t Ge(scalar_t a, scalar_t b) {
  return a >= b ? 1.0 : 0.0;
}

__device__ scalar_t Log(scalar_t a) {
  return log(a);
}

__device__ scalar_t Exp(scalar_t a) {
  return exp(a);
}

__device__ scalar_t Tanh(scalar_t a) {
  return tanh(a);
}

////////////////////////////////////////////////////////////////////////////////
// Define Kernel Function
////////////////////////////////////////////////////////////////////////////////

DEFINE_EWISE_BINARY_KERNEL(Add)
DEFINE_SCALAR_KERNEL(Add)
DEFINE_EWISE_BINARY_KERNEL(Mul)
DEFINE_SCALAR_KERNEL(Mul)
DEFINE_EWISE_BINARY_KERNEL(Div)
DEFINE_SCALAR_KERNEL(Div)
DEFINE_SCALAR_KERNEL(Power)
DEFINE_EWISE_BINARY_KERNEL(Maximum)
DEFINE_SCALAR_KERNEL(Maximum)
DEFINE_EWISE_BINARY_KERNEL(Eq)
DEFINE_SCALAR_KERNEL(Eq)
DEFINE_EWISE_BINARY_KERNEL(Ge)
DEFINE_SCALAR_KERNEL(Ge)
DEFINE_EWISE_UNARY_KERNEL(Log)
DEFINE_EWISE_UNARY_KERNEL(Exp)
DEFINE_EWISE_UNARY_KERNEL(Tanh)

////////////////////////////////////////////////////////////////////////////////
// Define Host Function
////////////////////////////////////////////////////////////////////////////////

DEFINE_EWISE_BINARY(Add)
DEFINE_SCALAR(Add)
DEFINE_EWISE_BINARY(Mul)
DEFINE_SCALAR(Mul)
DEFINE_EWISE_BINARY(Div)
DEFINE_SCALAR(Div)
DEFINE_SCALAR(Power)
DEFINE_EWISE_BINARY(Maximum)
DEFINE_SCALAR(Maximum)
DEFINE_EWISE_BINARY(Eq)
DEFINE_SCALAR(Eq)
DEFINE_EWISE_BINARY(Ge)
DEFINE_SCALAR(Ge)
DEFINE_EWISE_UNARY(Log)
DEFINE_EWISE_UNARY(Exp)
DEFINE_EWISE_UNARY(Tanh)


////////////////////////////////////////////////////////////////////////////////
// Different Matmul Kernel
////////////////////////////////////////////////////////////////////////////////

__global__ void NoTiledMatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, 
                                      uint32_t N, uint32_t P) {
  size_t xbase = blockIdx.x * blockDim.x + threadIdx.x;
  size_t ybase = blockIdx.y * blockDim.y + threadIdx.y;

  if (xbase < M && ybase < P) {
    scalar_t value = 0.0;
    for (size_t i = 0; i < N; i++) {
      value += a[xbase * N + i] * b[i * P + ybase];
    }
    out[xbase * P + ybase] = value;
  }
}

__global__ void ThreadTiledMatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, 
                                      uint32_t N, uint32_t P) {
  size_t xbase = blockIdx.x * blockDim.x + threadIdx.x;
  size_t ybase = blockIdx.y * blockDim.y + threadIdx.y;

  if (xbase < M / TILE && ybase < P / TILE) {
    scalar_t c0[TILE][TILE] = {0};
    for (size_t i = 0; i < N; i++) {
      // fetch the data
      scalar_t a0[TILE] = {0};
      scalar_t b0[TILE] = {0};
      for (size_t j = xbase * TILE; j < xbase * TILE + TILE; j++) {
        a0[j-xbase*TILE] = a[j*N+i];
      }
      for (size_t j = ybase * TILE; j < ybase * TILE + TILE; j++) {
        b0[j-ybase*TILE] = b[i*P+j];
      }
      // computation
      for (size_t x = 0; x < TILE; x++) {
        for (size_t y = 0; y < TILE; y++) {
          c0[x][y] += a0[x] * b0[y];
        }
      }
    }
    // store the data
    for (size_t x = xbase * TILE; x < xbase * TILE + TILE; x++) {
      for (size_t y = ybase * TILE; y < ybase * TILE + TILE; y++) {
        out[x*P+y] = c0[x-xbase*TILE][y-ybase*TILE];
      }
    }
  }
}

__global__ void BlockTiledMatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, 
                                      uint32_t N, uint32_t P) {
  __shared__ scalar_t sma[BLOCK_TILE][BLOCK_TILE], smb[BLOCK_TILE][BLOCK_TILE];
  size_t xblock = blockIdx.x;
  size_t yblock = blockIdx.y;
  size_t xthread = threadIdx.x;
  size_t ythread = threadIdx.y;
  size_t tid = xthread * blockDim.y + ythread; // 这个线程在block中的索引
  scalar_t c0 = 0;
  for (size_t i = 0; i < N / BLOCK_TILE; i ++) {
    __syncthreads();
    // cooperative fetching
    // fetch一个block内需要的sma和smb
    for (size_t j = 0; j < BLOCK_TILE * BLOCK_TILE / BASE_THREAD_NUM; j++) {
      size_t x = (j * BASE_THREAD_NUM + tid) / BLOCK_TILE;
      size_t y = (j * BASE_THREAD_NUM + tid) % BLOCK_TILE;
      sma[x][y] = a[(xblock*BLOCK_TILE+x)*N+(i*BLOCK_TILE+y)];
      smb[x][y] = b[(i*BLOCK_TILE+x)*P+(yblock*BLOCK_TILE+y)];
    }
    __syncthreads();
    for (size_t k = 0; k < BLOCK_TILE; k++) {
      scalar_t a0 = sma[xthread][k];
      scalar_t b0 = smb[k][ythread];
      c0 += a0 * b0;
    }
  }
  size_t xbase = xblock * blockDim.x + xthread;
  size_t ybase = yblock * blockDim.y + ythread;
  out[xbase*P+ybase] = c0;
}

__global__ void AllTiledMatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, 
                                      uint32_t N, uint32_t P) {
  #define S 64
  #define L 4
  // 为了和BlockTiled进行对比，保持每个block中的shared memory中的元素数量不变，都是256
  __shared__ scalar_t sma[S][L], smb[L][S];
  size_t xblock = blockIdx.x;
  size_t yblock = blockIdx.y;
  size_t xthread = threadIdx.x;
  size_t ythread = threadIdx.y;
  size_t tid = xthread * blockDim.y + ythread; // 这个线程在block中的索引
  scalar_t a0[TILE] = {0};
  scalar_t b0[TILE] = {0};
  scalar_t c0[TILE][TILE] = {0};
  for (size_t i = 0; i < N / L; i++) {
    __syncthreads();
    // cooperative fetching
    // fetch一个block内需要的sma和smb
    for (size_t j = 0; j < S * L / BASE_THREAD_NUM; j++) {
      size_t ax = (j * BASE_THREAD_NUM + tid) / L;
      size_t ay = (j * BASE_THREAD_NUM + tid) % L;
      size_t bx = (j * BASE_THREAD_NUM + tid) / S;
      size_t by = (j * BASE_THREAD_NUM + tid) % S;
      sma[ax][ay] = a[(xblock*S+ax)*N+(i*L+ay)];
      smb[bx][by] = b[(i*L+bx)*P+(yblock*S+by)];
    }
    __syncthreads();
    // using thread tiled
    for (size_t k = 0; k < L; k++) {
      for (size_t m = xthread * TILE; m < xthread * TILE + TILE; m++) {
        a0[m - xthread*TILE] = sma[m][k];
      }
      for (size_t m = ythread * TILE; m < ythread * TILE + TILE; m++) {
        b0[m - ythread*TILE] = smb[k][m];
      }
      for (size_t x = 0; x < TILE; x++) {
        for (size_t y = 0; y < TILE; y++) {
          c0[x][y] += a0[x] * b0[y];
        }
      }
    }
  }
  size_t xbase = xblock * blockDim.x + xthread;
  size_t ybase = yblock * blockDim.y + ythread;
  for (size_t x = xbase * TILE; x < xbase * TILE + TILE; x++) {
    for (size_t y = ybase * TILE; y < ybase * TILE + TILE; y++) {
      out[x*P+y] = c0[x-xbase*TILE][y-ybase*TILE];
    }
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  bool thread_tiled = M % TILE == 0 && P % TILE == 0;
  bool block_tiled = M % BLOCK_TILE == 0 && N % BLOCK_TILE == 0 && P % BLOCK_TILE == 0;
  bool all_tiled = M % S == 0 && N % L == 0 && P % S == 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  if(all_tiled) {
    // printf("AllTiledMatmulKernel\n");
    CudaDims dim = CudaTwoDim(M / TILE, P / TILE);
    AllTiledMatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  } else if (block_tiled) {
    // printf("BlockTiledMatmulKernel\n");
    CudaDims dim = CudaTwoDim(M, P);
    BlockTiledMatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  } else if (thread_tiled) {
    // printf("ThreadTiledMatmulKernel\n");
    CudaDims dim = CudaTwoDim(M / TILE, P / TILE);
    ThreadTiledMatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  } else {
    // printf("NoTiledMatmulKernel\n");
    CudaDims dim = CudaTwoDim(M, P);
    NoTiledMatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "GPU time: " << milliseconds << " ms" << std::endl;
  // cudaError_t err = cudaGetLastError();
  // if (err != cudaSuccess) {
  //     printf("CUDA Kernel Error: %s\n", cudaGetErrorString(err));
  // }
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t out_size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < out_size) {
    scalar_t maximum = a[gid*reduce_size];
    for (size_t i = 0; i < reduce_size; i++) {
      scalar_t value = a[gid*reduce_size+i];
      maximum = value > maximum ? value : maximum;
    }
    out[gid] = maximum;
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t out_size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < out_size) {
    scalar_t sum = 0;
    for (size_t i = 0; i < reduce_size; i++) {
      sum += a[gid*reduce_size+i];
    }
    out[gid] = sum;
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
