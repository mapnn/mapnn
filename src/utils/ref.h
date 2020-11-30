#pragma once
// exchange-add operation for atomic operations on reference counters
#ifdef NNOKM_XADD
  // allow to use user-defined macro
#elif defined __GNUC__ || defined __clang__
#  if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ && !defined __EMSCRIPTEN__ && !defined(__CUDACC__)  && !defined __INTEL_COMPILER
#    ifdef __ATOMIC_ACQ_REL
#      define NNOKM_XADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)
#    else
#      define NNOKM_XADD(addr, delta) __atomic_fetch_add((_Atomic(int)*)(addr), delta, 4)
#    endif
#  else
#    if defined __ATOMIC_ACQ_REL && !defined __clang__
       // version for gcc >= 4.7
#      define NNOKM_XADD(addr, delta) (int)__atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#    else
#      define NNOKM_XADD(addr, delta) (int)__sync_fetch_and_add((unsigned*)(addr), (unsigned)(delta))
#    endif
#  endif
#elif defined _MSC_VER && !defined RC_INVOKED
#  include <intrin.h>
#  define NNOKM_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)
#endif

#define MALLOC_ALIGN    16
static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n-1) & -n;
}
static inline void* fastMalloc(size_t size)
{
#if _MSC_VER
    return _aligned_malloc(size, MALLOC_ALIGN);
#elif _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
    void* ptr = 0;
    if (posix_memalign(&ptr, MALLOC_ALIGN, size))
        ptr = 0;
    return ptr;
#elif __ANDROID__ && __ANDROID_API__ < 17
    return memalign(MALLOC_ALIGN, size);
#else
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + MALLOC_ALIGN);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
#endif
}

static inline void fastFree(void* ptr)
{
    if (ptr)
    {
#if _MSC_VER
        _aligned_free(ptr);
#elif _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
        free(ptr);
#elif __ANDROID__ && __ANDROID_API__ < 17
        free(ptr);
#else
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
#endif
    }
}
