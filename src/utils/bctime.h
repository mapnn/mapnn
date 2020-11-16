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

#ifndef __MAPNN_BCTIME_H__
#define __MAPNN_BCTIME_H__

#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

class BCTime {
public:
    BCTime();
    BCTime(const char* label);
    ~BCTime();
    BCTime(const BCTime&)  = delete;
    BCTime(const BCTime&&) = delete;
    BCTime& operator=(const BCTime&) = delete;
    BCTime& operator=(const BCTime&&) = delete;
    float get();

private:
    uint64_t time_;
    const char* label_ = NULL;
};

inline BCTime::BCTime() {
    struct timeval Current;
    gettimeofday(&Current, nullptr);
    time_= Current.tv_sec * 1000000 + Current.tv_usec;
}
inline BCTime::BCTime(const char* label) {
    label_ = label;
    struct timeval Current;
    gettimeofday(&Current, nullptr);
    time_= Current.tv_sec * 1000000 + Current.tv_usec;
}
inline BCTime::~BCTime() {
#ifdef __DEBUG__
    if(label_ != NULL) {
        struct timeval Current;
        gettimeofday(&Current, nullptr);
        auto lastBCTime = Current.tv_sec * 1000000 + Current.tv_usec;
        printf("%s, cost time: %f ms\n", label_, (float)(lastBCTime - time_) / 1000.0f);
    }
#endif
}
inline float BCTime::get() {
    struct timeval Current;
    gettimeofday(&Current, nullptr);
    auto lastBCTime = Current.tv_sec * 1000000 + Current.tv_usec;
    return (float)(lastBCTime - time_) / 1000.0f;
}

#endif // __MAPNN_BCTIME_H__
