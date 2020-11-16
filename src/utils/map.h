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

#ifndef __MAPNN_MAP_H__
#define __MAPNN_MAP_H__

#include "operator.h"
#include "graph.h"
#include "node.h"

#define DECLARE_OPTIMAL_MAP(map)                                \
class map: public Map {                                         \
public:                                                         \
    map(bool (*option)(Operator&)):Map(option){}                \
    const char* name() override { return #map; }                \
    const MapStage getStage() override { return STAGE_OPTIMAL; }\
    bool request(Operator& op) override;                        \
    bool run(Graph* graph, Node* node) override;                \
};
#define OPTION(exp) [](Operator& op)->bool{return exp;}

#define DECLARE_FUSION_MAP(map)                                 \
class map: public Map {                                         \
public:                                                         \
    map(bool (*option)(Operator&)):Map(option){}                \
    const char* name() override { return #map; }                \
    const MapStage getStage() override { return STAGE_FUSION; } \
    bool request(Operator& op) override;                        \
    bool run(Graph* graph, Node* node) override;                \
};
#define OPTION(exp) [](Operator& op)->bool{return exp;}

#define DECLARE_KERNEL_MAP(map)                                 \
class map: public Map {                                         \
public:                                                         \
    map(bool (*option)(Operator&)):Map(option){}                \
    const char* name() override { return #map; }                \
    const MapStage getStage() override { return STAGE_KERNEL; } \
    bool request(Operator& op) override;                        \
    bool run(Graph* graph, Node* node) override;                \
};
#define OPTION(exp) [](Operator& op)->bool{return exp;}

class Map {
private:
    bool (*option_)(Operator&) = NULL;
public:
    Map(bool (*option)(Operator&)):option_(option){};
    virtual ~Map() = default;
    virtual const char* name();
    virtual const MapStage getStage();
    virtual bool run(Graph* graph, Node* node);
    virtual bool request(Operator& op);
    virtual bool option(Operator& op);
};
inline const char* Map::name() { 
    return "";
}
inline const MapStage Map::getStage() { 
    return STAGE_NULL;
}
inline bool Map::run(Graph* graph, Node* node) {
    return false;
}
inline bool Map::request(Operator& op) {
    return false;
}
inline bool Map::option(Operator& op) {
    if(option_ == NULL) return true;
    return option_(op);
}

#endif // __MAPNN_MAP_H__
