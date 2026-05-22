add_executable(main_edge_matching main_edge_matching.cpp)
target_link_libraries(main_edge_matching lems_vo_slam 
                      -L/gpfs/data/bkimia/cchien3/yaml-cpp/bin/lib64 yaml-cpp
                      ${THIRD_PARTY_LIBS}
)