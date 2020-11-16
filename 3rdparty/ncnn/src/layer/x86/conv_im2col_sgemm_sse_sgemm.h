// BUG1989 is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
//
#include "option.h"
#include "mat.h"
namespace ncnn{
static void conv_im2col_sgemm_sse_sgemm(const Mat &bottom_blob, Mat &top_blob, const Mat & kernel_tm, const Mat& _bias, 
            const int kernel_w, const int kernel_h, const int stride_w, const int stride_h, const Option& opt, int inch)
{
    Mat bottom_tm = bottom_blob;
    //size_t elemsize = bottom_blob.elemsize;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* bias = _bias;

    // sgemm(int M, int N, int L, float* A, float* B, float* C)
    {
        //int M = outch;                    // outch
        int N = outw * outh;                // outsize or out stride
        int L = kernel_w * kernel_h * inch; // ksize * inch

        int nn_outch = 0;
        int remain_outch_start = 0;

        nn_outch = outch >> 2;
        remain_outch_start = nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp=0; pp<nn_outch; pp++)
        {
            int i =  pp * 4;

            float* output0 = top_blob.channel(i);
            float* output1 = top_blob.channel(i+1);
            float* output2 = top_blob.channel(i+2);
            float* output3 = top_blob.channel(i+3);

            const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + i : zeros;

            int j=0;
            for (; j+3<N; j=j+4)
            {
                const float* vb = bottom_tm.channel(j/4);
                const float* va = kernel_tm.channel(i/4);
#if 0 //TODO: BUG for googlenet
                __m128 _sum0 = _mm_set1_ps(biasptr[0]);
                __m128 _sum1 = _mm_set1_ps(biasptr[1]);
                __m128 _sum2 = _mm_set1_ps(biasptr[2]);
                __m128 _sum3 = _mm_set1_ps(biasptr[3]);

                int k=0;
                for (; k+3<L; k=k+4)
                {
                    // k0
                    __m128 _vb = _mm_loadu_ps(vb);
                    __m128 _va0 = _mm_set1_ps(va[0]);
                    __m128 _va1 = _mm_set1_ps(va[1]);
                    __m128 _va2 = _mm_set1_ps(va[2]);
                    __m128 _va3 = _mm_set1_ps(va[3]);
                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb, _va0));// sum0 = (a00-a03) * k00
                    _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_vb, _va1));// sum1 = (a00-a03) * k10
                    _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_vb, _va2));// sum2 = (a00-a03) * k20
                    _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_vb, _va3));// sum3 = (a00-a03) * k30

                    // k1
                    _vb = _mm_loadu_ps(vb+4);
                    _va0 = _mm_set1_ps(va[4]);
                    _va1 = _mm_set1_ps(va[5]);
                    _va2 = _mm_set1_ps(va[6]);
                    _va3 = _mm_set1_ps(va[7]);
                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb, _va0));// sum0 = (a10-a13) * k01
                    _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_vb, _va1));// sum1 = (a10-a13) * k11
                    _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_vb, _va2));// sum2 = (a10-a13) * k21
                    _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_vb, _va3));// sum3 = (a10-a13) * k31

                    // k2
                    _vb = _mm_loadu_ps(vb+8);
                    _va0 = _mm_set1_ps(va[8]);
                    _va1 = _mm_set1_ps(va[9]);
                    _va2 = _mm_set1_ps(va[10]);
                    _va3 = _mm_set1_ps(va[11]);
                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb, _va0));// sum0 = (a20-a23) * k02
                    _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_vb, _va1));// sum1 = (a20-a23) * k12
                    _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_vb, _va2));// sum2 = (a20-a23) * k22
                    _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_vb, _va3));// sum3 = (a20-a23) * k32

                    // k3
                    _vb = _mm_loadu_ps(vb+12);
                    _va0 = _mm_set1_ps(va[12]);
                    _va1 = _mm_set1_ps(va[13]);
                    _va2 = _mm_set1_ps(va[14]);
                    _va3 = _mm_set1_ps(va[15]);
                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb, _va0));// sum0 = (a30-a33) * k03
                    _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_vb, _va1));// sum1 = (a30-a33) * k13
                    _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_vb, _va2));// sum2 = (a30-a33) * k23
                    _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_vb, _va3));// sum3 = (a30-a33) * k33

                    va += 16;
                    vb += 16;
                }

                for (; k<L; k++)
                {
                    // k0
                    __m128 _vb = _mm_loadu_ps(vb);
                    __m128 _va0 = _mm_set1_ps(va[0]);
                    __m128 _va1 = _mm_set1_ps(va[1]);
                    __m128 _va2 = _mm_set1_ps(va[2]);
                    __m128 _va3 = _mm_set1_ps(va[3]);
                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb, _va0));// sum0 = (a00-a03) * k00
                    _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_vb, _va1));// sum1 = (a00-a03) * k10
                    _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_vb, _va2));// sum2 = (a00-a03) * k20
                    _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_vb, _va3));// sum3 = (a00-a03) * k30
                    
                    va += 4;
                    vb += 4;
                }
                _mm_storeu_ps(output0, _sum0);
                _mm_storeu_ps(output1, _sum1);
                _mm_storeu_ps(output2, _sum2);
                _mm_storeu_ps(output3, _sum3);
#else
                float sum0[4] = {0};
                float sum1[4] = {0};
                float sum2[4] = {0};
                float sum3[4] = {0};
               
                int k=0;
                for (; k+7<L; k=k+8)
                {
                    for (int n=0; n<4; n++)
                    {
                        sum0[n] += va[0] * vb[n];
                        sum1[n] += va[1] * vb[n];
                        sum2[n] += va[2] * vb[n];
                        sum3[n] += va[3] * vb[n];
                        va += 4;

                        sum0[n] += va[0] * vb[n+4];
                        sum1[n] += va[1] * vb[n+4];
                        sum2[n] += va[2] * vb[n+4];
                        sum3[n] += va[3] * vb[n+4];
                        va += 4;

                        sum0[n] += va[0] * vb[n+8];
                        sum1[n] += va[1] * vb[n+8];
                        sum2[n] += va[2] * vb[n+8];
                        sum3[n] += va[3] * vb[n+8];
                        va += 4;

                        sum0[n] += va[0] * vb[n+12];
                        sum1[n] += va[1] * vb[n+12];
                        sum2[n] += va[2] * vb[n+12];
                        sum3[n] += va[3] * vb[n+12];
                        va += 4;

                        sum0[n] += va[0] * vb[n+16];
                        sum1[n] += va[1] * vb[n+16];
                        sum2[n] += va[2] * vb[n+16];
                        sum3[n] += va[3] * vb[n+16];
                        va += 4;

                        sum0[n] += va[0] * vb[n+20];
                        sum1[n] += va[1] * vb[n+20];
                        sum2[n] += va[2] * vb[n+20];
                        sum3[n] += va[3] * vb[n+20];
                        va += 4;

                        sum0[n] += va[0] * vb[n+24];
                        sum1[n] += va[1] * vb[n+24];
                        sum2[n] += va[2] * vb[n+24];
                        sum3[n] += va[3] * vb[n+24];
                        va += 4;

                        sum0[n] += va[0] * vb[n+28];
                        sum1[n] += va[1] * vb[n+28];
                        sum2[n] += va[2] * vb[n+28];
                        sum3[n] += va[3] * vb[n+28];
                        va -= 28;
                    }

                    va += 32;
                    vb += 32;
                }

                for (; k<L; k++)
                {
                    for (int n=0; n<4; n++)
                    {
                        sum0[n] += va[0] * vb[n];
                        sum1[n] += va[1] * vb[n];
                        sum2[n] += va[2] * vb[n];
                        sum3[n] += va[3] * vb[n];
                    }
                    
                    va += 4;
                    vb += 4;
                }

                for (int n=0; n<4; n++)
                {
                    output0[n] = sum0[n] + biasptr[0];
                    output1[n] = sum1[n] + biasptr[1];
                    output2[n] = sum2[n] + biasptr[2];
                    output3[n] = sum3[n] + biasptr[3];
                }
#endif // __SSE__
                output0 += 4;
                output1 += 4;
                output2 += 4;
                output3 += 4;
            }

            for (; j<N; j++)
            {                
                const float* vb = bottom_tm.channel(j/4 + j%4);
                const float* va = kernel_tm.channel(i/4);
#if __SSE__
                __m128 _sum0_3 = _mm_loadu_ps(biasptr);
                __m128 _sum0 = _mm_set1_ps(0.0);
                __m128 _sum1 = _mm_set1_ps(0.0);
                __m128 _sum2 = _mm_set1_ps(0.0);
                __m128 _sum3 = _mm_set1_ps(0.0);

                int k=0;
                for (; k+3<L; k=k+4)
                {
                    __m128 _vb0 = _mm_set1_ps(vb[0]);
                    __m128 _vb1 = _mm_set1_ps(vb[1]);
                    __m128 _vb2 = _mm_set1_ps(vb[2]);
                    __m128 _vb3 = _mm_set1_ps(vb[3]);
                    __m128 _va0 = _mm_loadu_ps(va);
                    __m128 _va1 = _mm_loadu_ps(va+4);
                    __m128 _va2 = _mm_loadu_ps(va+8);
                    __m128 _va3 = _mm_loadu_ps(va+12);

                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va0, _vb0));// sum0 += (k00-k30) * a00
                    _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_va1, _vb1));// sum1 += (k01-k31) * a10
                    _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_va2, _vb2));// sum2 += (k02-k32) * a20
                    _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_va3, _vb3));// sum3 += (k03-k33) * a30

                    va += 16;
                    vb += 4;
                }

                _sum0 = _mm_add_ps(_sum0, _sum1);
                _sum2 = _mm_add_ps(_sum2, _sum3);
                _sum0_3 = _mm_add_ps(_sum0_3, _sum0);
                _sum0_3 = _mm_add_ps(_sum0_3, _sum2);

                for (; k<L; k++)
                {
                    __m128 _vb0 = _mm_set1_ps(vb[0]);
                    __m128 _va = _mm_loadu_ps(va); 

                    _sum0_3 = _mm_add_ps(_sum0_3, _mm_mul_ps(_va, _vb0));// sum0 += (k00-k30) * a00

                    va += 4;
                    vb += 1;
                }         
                output0[0] = _sum0_3[0];
                output1[0] = _sum0_3[1];
                output2[0] = _sum0_3[2];
                output3[0] = _sum0_3[3];
#else
                float sum0 = biasptr[0];
                float sum1 = biasptr[1];
                float sum2 = biasptr[2];
                float sum3 = biasptr[3];

                for (int k=0; k<L; k++)
                {
                    sum0 += va[0] * vb[0];
                    sum1 += va[1] * vb[0];
                    sum2 += va[2] * vb[0];
                    sum3 += va[3] * vb[0];

                    va += 4;
                    vb += 1;
                }
                
                output0[0] = sum0;
                output1[0] = sum1;
                output2[0] = sum2;
                output3[0] = sum3;
#endif // __SSE__
                output0++;
                output1++;
                output2++;
                output3++;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i=remain_outch_start; i<outch; i++)
        {
            float* output = top_blob.channel(i);

            const float bias0 = bias ? bias[i] : 0.f;

            int j=0;
            for (; j+3<N; j=j+4)
            {
                const float* vb = bottom_tm.channel(j/4);       
                const float* va = kernel_tm.channel(i/4 + i%4);
#if __SSE__
                __m128 _sum0 = _mm_set1_ps(bias0);

                int k=0;
                for (; k+3<L; k=k+4)
                {
                    // k0
                    __m128 _va0 = _mm_set1_ps(va[0]);
                    __m128 _va1 = _mm_set1_ps(va[1]);
                    __m128 _va2 = _mm_set1_ps(va[2]);
                    __m128 _va3 = _mm_set1_ps(va[3]);
                    __m128 _vb0 = _mm_loadu_ps(vb);
                    __m128 _vb1 = _mm_loadu_ps(vb+4);
                    __m128 _vb2 = _mm_loadu_ps(vb+8);
                    __m128 _vb3 = _mm_loadu_ps(vb+12);

                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb0, _va0));// sum0 = (a00-a03) * k00                
                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb1, _va1));// sum0 += (a10-a13) * k01
                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb2, _va2));// sum0 += (a20-a23) * k02
                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb3, _va3));// sum0 += (a30-a33) * k03
                
                    va += 4;
                    vb += 16;
                }

                for (; k<L; k++)
                {
                    // k0
                    __m128 _va0 = _mm_set1_ps(va[0]);
                    __m128 _vb0 = _mm_loadu_ps(vb);

                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb0, _va0));    // sum0 = (a00-a03) * k00

                    va += 1;
                    vb += 4;
                }
                _mm_storeu_ps(output, _sum0); 
#else                
                float sum[4] = {0};

                int k=0;
                for (; k+3<L; k=k+4)
                {
                    for (int n=0; n<4; n++)
                    {
                        sum[n] += va[0] * vb[n];
                        sum[n] += va[1] * vb[n+4];
                        sum[n] += va[2] * vb[n+8];
                        sum[n] += va[3] * vb[n+12];
                        //sum[n] += va[4] * vb[n+16];
                        //sum[n] += va[5] * vb[n+20];
                        //sum[n] += va[6] * vb[n+24];
                        //sum[n] += va[7] * vb[n+28];
                    }

                    va += 4;
                    vb += 16;
                }

                for (; k<L; k++)
                {
                    for (int n=0; n<4; n++)
                    {
                        sum[n] += va[0] * vb[n];
                    }

                    va += 1;
                    vb += 4;
                }

                for (int n=0; n<4; n++)
                {
                    output[n] = sum[n] + bias0;
                }
#endif // __SSE__
                output += 4;
            }

            for (; j<N; j++)
            {
                const float* vb = bottom_tm.channel(j/4 + j%4);
                const float* va = kernel_tm.channel(i/4 + i%4);

                int k=0;
#if __SSE__
                __m128 _sum0 = _mm_set1_ps(0.f);

                for (; k+3<L; k+=4)
                {
                    __m128 _p0 = _mm_loadu_ps(vb);
                    __m128 _k0 = _mm_loadu_ps(va);
                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_p0, _k0));

                    va += 4;
                    vb += 4;                    
                }
                float sum0 = bias0 + _sum0[0] + _sum0[1] + _sum0[2] + _sum0[3];
#else
                float sum0 = bias0;
#endif // __SSE__
                for (; k<L; k++)
                {
                    sum0 += va[0] * vb[0];

                    va += 1;
                    vb += 1;
                }
                output[0] = sum0;

                output++;
            }
        }
    }   
}
}
