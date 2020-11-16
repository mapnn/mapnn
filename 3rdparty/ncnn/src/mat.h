#ifndef NCNN_MAT_H
#define NCNN_MAT_H
namespace ncnn {
class Mat {
public:
    Mat(int w, void* data, size_t elemsize = 4u);
    Mat(int w, int h, void* data, size_t elemsize = 4u);
    Mat(int w, int h, int c, void* data, size_t elemsize = 4u);
    Mat(int w, void* data, size_t elemsize, int elempack);
    Mat(int w, int h, void* data, size_t elemsize, int elempack);
    Mat(int w, int h, int c, void* data, size_t elemsize, int elempack);
    ~Mat() = default;
    Mat& operator=(const Mat& m);
    void fill(float v);
#if __ARM_NEON
    void fill(float32x4_t _v);
#endif // __ARM_NEON
    void create(int w, size_t elemsize = 4u, Allocator* allocator = 0);
    void create(int w, int h, size_t elemsize = 4u, Allocator* allocator = 0);
    void create(int w, int h, int c, size_t elemsize = 4u, Allocator* allocator = 0);
    void create(int w, size_t elemsize, int elempack, Allocator* allocator = 0);
    void create(int w, int h, size_t elemsize, int elempack, Allocator* allocator = 0);
    void create(int w, int h, int c, size_t elemsize, int elempack, Allocator* allocator = 0);
    size_t total() const;
    bool empty() const;
    Mat channel(int c);
    const Mat channel(int c) const;
    float* row(int y);
    const float* row(int y) const;
    template<typename T> operator T*();
    template<typename T> operator const T*() const;
    float& operator[](int i);
    const float& operator[](int i) const;
    size_t cstep;
    void* data;
    // element size in bytes
    // 4 = float32/int32
    // 2 = float16
    // 1 = int8/uint8
    // 0 = empty
    size_t elemsize;

    // packed count inside element
    // c/1-h-w-1  h/1-w-1  w/1-1  scalar
    // c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
    // c/8-h-w-8  h/8-w-8  w/8-8  avx/fp16
    int elempack;
    int dims;
    int w;
    int h;
    int c;
};

inline Mat::Mat(int _w, void* _data, size_t _elemsize)
    : data(_data), elemsize(_elemsize), elempack(1), dims(1), w(_w), h(1), c(1)
{
    cstep = w;
}

inline Mat::Mat(int _w, int _h, void* _data, size_t _elemsize)
    : data(_data), elemsize(_elemsize), elempack(1), dims(2), w(_w), h(_h), c(1)
{
    cstep = w * h;
}

inline Mat::Mat(int _w, int _h, int _c, void* _data, size_t _elemsize)
    : data(_data), elemsize(_elemsize), elempack(1), dims(3), w(_w), h(_h), c(_c)
{
    cstep = w * h;
}

inline Mat::Mat(int _w, void* _data, size_t _elemsize, int _elempack)
    : data(_data), elemsize(_elemsize), elempack(_elempack), dims(1), w(_w), h(1), c(1)
{
    cstep = w;
}

inline Mat::Mat(int _w, int _h, void* _data, size_t _elemsize, int _elempack)
    : data(_data), elemsize(_elemsize), elempack(_elempack), dims(2), w(_w), h(_h), c(1)
{
    cstep = w * h;
}

inline Mat::Mat(int _w, int _h, int _c, void* _data, size_t _elemsize, int _elempack)
    : data(_data), elemsize(_elemsize), elempack(_elempack), dims(3), w(_w), h(_h), c(_c)
{
    cstep = w * h;
}

inline void Mat::fill(float _v)
{
    int size = (int)total();
    float* ptr = (float*)data;

    int remain = size;

    for (; remain>0; remain--)
    {
        *ptr++ = _v;
    }
}

#if __ARM_NEON
inline void Mat::fill(float32x4_t _v)
{
    int size = total();
    float* ptr = (float*)data;
    for (int i=0; i<size; i++)
    {
        vst1q_f32(ptr, _v);
        ptr += 4;
    }
}
#endif
inline size_t Mat::total() const
{
    return cstep * c;
}

inline bool Mat::empty() const
{
    return data == 0 || total() == 0;
}

inline Mat Mat::channel(int _c)
{
    return Mat(w, h, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack);
}

inline const Mat Mat::channel(int _c) const
{
    return Mat(w, h, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack);
}

inline float* Mat::row(int y)
{
    return (float*)((unsigned char*)data + w * y * elemsize);
}

inline const float* Mat::row(int y) const
{
    return (const float*)((unsigned char*)data + w * y * elemsize);
}
template <typename T>
inline Mat::operator T*()
{
    return (T*)data;
}

template <typename T>
inline Mat::operator const T*() const
{
    return (const T*)data;
}
inline float& Mat::operator[](int i)
{
    return ((float*)data)[i];
}

inline const float& Mat::operator[](int i) const
{
    return ((const float*)data)[i];
}
inline void Mat::create(int _w, size_t _elemsize, Allocator* allocator)
{
    if(_w != w) printf("[ERROR] ncnn mat error w(%d != %d)\n", _w, w);
}

inline void Mat::create(int _w, int _h, size_t _elemsize, Allocator* allocator)
{
    if(_w != w) printf("[ERROR] ncnn mat error w(%d != %d)\n", _w, w);
    if(_h != h) printf("[ERROR] ncnn mat error h(%d != %d)\n", _h, h);
}

inline void Mat::create(int _w, int _h, int _c, size_t _elemsize, Allocator* allocator)
{
    if(_w != w) printf("[ERROR] ncnn mat error w(%d != %d)\n", _w, w);
    if(_h != h) printf("[ERROR] ncnn mat error h(%d != %d)\n", _h, h);
    if(_c != c) printf("[ERROR] ncnn mat error c(%d != %d)\n", _c, c);
}

inline void Mat::create(int _w, size_t _elemsize, int _elempack, Allocator* allocator)
{
    if(_w != w) printf("[ERROR] ncnn mat error w(%d != %d)\n", _w, w);
}

inline void Mat::create(int _w, int _h, size_t _elemsize, int _elempack, Allocator* allocator)
{
    if(_w != w) printf("[ERROR] ncnn mat error w(%d != %d)\n", _w, w);
    if(_h != h) printf("[ERROR] ncnn mat error h(%d != %d)\n", _h, h);
}

inline void Mat::create(int _w, int _h, int _c, size_t _elemsize, int _elempack, Allocator* allocator)
{
    if(_w != w) printf("[ERROR] ncnn mat error w(%d != %d)\n", _w, w);
    if(_h != h) printf("[ERROR] ncnn mat error h(%d != %d)\n", _h, h);
    if(_c != c) printf("[ERROR] ncnn mat error c(%d != %d)\n", _c, c);
}
}
#endif // NCNN_MAT_H
