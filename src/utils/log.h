/* Copyright 2020 The Mapnn Team. All Rights Reserved. 
 *                                                                            
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *                                                                            
 *     http://www.apache.org/licenses/LICENSE-2.0
 *                                                                            
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __MAPNN_LOG_H__
#define __MAPNN_LOG_H__

#include <stdio.h>

// for apple
#if defined(__APPLE__)
#define LOGI(...)  fprintf(stdout,__VA_ARGS__)
#define LOGE(...)  fprintf(stderr,__VA_ARGS__)
#ifdef __DEBUG_GRAPH__
#define LOGDG(...)  printf(__VA_ARGS__)
#endif
#ifdef __DEBUG_KERNEL__
#define LOGDK(...)  printf(__VA_ARGS__)
#endif

// for android
#elif defined(ANDROID)
#define LOG_TAG "MapNN"
#include <android/log.h>
#define LOGI(...)  { \
    __android_log_print(ANDROID_LOG_INFO,"MapNN",__VA_ARGS__); \
    fprintf(stdout,__VA_ARGS__); \
}
#define LOGE(...)  { \
    __android_log_print(ANDROID_LOG_ERROR,"MapNN",__VA_ARGS__); \
    fprintf(stderr,__VA_ARGS__); \
}
#ifdef __DEBUG_GRAPH__
#define LOGDG(...)  { \
    __android_log_print(ANDROID_LOG_DEBUG,"MapNN",__VA_ARGS__); \
    printf(__VA_ARGS__); \
}
#endif
#ifdef __DEBUG_KERNEL__
#define LOGDK(...)  { \
    __android_log_print(ANDROID_LOG_DEBUG,"MapNN",__VA_ARGS__); \
    printf(__VA_ARGS__); \
}
#endif
#else
#include <stdio.h>
#define LOGI(...)  fprintf(stdout,__VA_ARGS__)
#define LOGE(...)  fprintf(stderr,__VA_ARGS__)
#ifdef __DEBUG_GRAPH__
#define LOGDG(...) fprintf(stdout,__VA_ARGS__)
#endif
#ifdef __DEBUG_KERNEL__
#define LOGDK(...)  fprintf(stdout,__VA_ARGS__)
#endif

#endif

#ifndef LOGDG
#define LOGDG(...)
#endif

#ifndef LOGDK
#define LOGDK(...)
#endif

#endif // __MAPNN_LOG_H__
