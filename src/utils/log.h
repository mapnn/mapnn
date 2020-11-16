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

#define __DEBUG_OPERATOR__
#define __DEBUG_KERNEL__

#if defined(__APPLE__)
#elif defined(ANDROID)
#include <android/log.h>
#else
#include <stdio.h>

#define LOGI(f, ...)  printf(f, ##__VA_ARGS__)
#define LOGE(f, ...)  printf(f, ##__VA_ARGS__)

#ifdef __DEBUG_OPERATOR__
#define LOGDG(f, ...)  printf(f, ##__VA_ARGS__)
#endif

#ifdef __DEBUG_KERNEL__
#define LOGDK(f, ...)  printf(f, ##__VA_ARGS__)
#endif

#endif

#endif // __MAPNN_LOG_H__
