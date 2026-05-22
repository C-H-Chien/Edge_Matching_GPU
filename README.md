# GPU-Based Edge Feature Matching
## Research @ LEMS, Brown University
This repository is home to the GPU implementation that constructs edge correspondences from _(i)_ a stereo image pair and _(ii)_ from temporal stereo images to arrive at quadtiplet edge correspondences (or "quads"), aiming to address the super slow computation time from the CPU implementation. This is one of the building blocks of [QuadEdge-VO](https://github.com/Brown-LEMS/Edge_Based_Visual_Odometry), a visual odometry framework using edge quads. Thus, in our temporal edge matching, we often say it construct edge correspondences between a "key (stereo) frame" and the "current (stereo) frame".

Note that this code **does not** require a rectified stereo image pair, nor needs to have the same image resolution for left and right cameras. This is the case in the ETH3D-SLAM dataset, and we use is as a demonstration here.

## Dependencies
Running on both CPU and GPU requires the following libraries/packages:
* Eigen 3.X or beyond
* OpenCV 4.X (with contrib is optional)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp) (used to parse data from the dataset config file)
* CUDA (tested on version 12.9)
* glog or boost (Optional)

## How to use the code
### Build and Compile
Follow the standard build and compilation, namely, 
```bash
mkdir build && cd build
cmake ..
make -j
```
You shall find executive files under the ``bin`` folder. Note that CUDA sources are compiled when `ENABLE_CUDA` is on (`src/CMakeLists.txt`), linked against `CUDA::cudart` with separable compilation.

For Brown University users, a shell script `load_oscars_modules.sh` is provided so a simple command `$ source ./load_oscars_modules.sh` would automatically load all the necessary modules to build the code.

### Execute
Run the execution file `main_VO` with a specified YAML configuration file:
```bash
./bin/main_VO --config_file ./your-config-file.yaml
```
where the configuration file ``your-config-file.yaml`` can be customized depending on the dataset in use. See ``.yaml`` files under the ``config`` folder for more information. <br />

## Overview of GPU Implemented Edge Correspondences

Basically, there are four sequental stages:
| Stage | Role |
|-------|------|
| **Third-Order Edge Detector** | Detect accurate third-order edges (oriented points) on the stereo image pair |
| **Preparation** | Set up geometry and images on the GPU for the current stereo pair |
| **Stereo** | Match left edges to right edges within one stereo frame |
| **Temporal** | Match keyframe stereo mates to current-frame stereo mates across time |

The third-order edge detector is borrowed from my [another GitHub repository](https://github.com/C-H-Chien/Third-Order-Edge-Detector). In the process of stereo and temporal edge matching, several GPU parallel strategies are adopted:
- Undistroted stereo images are cached as CUDA textures for fast, subpixel intensity sampling. The same texutres are reused over the time when applicable.
- The geometric filters, photometric filters, and refinement as well as consolidation are basically one kernel per filter/step, if otherwise specified. In each kernel, one thread is assigned either per left edge, or per candidate mate, depending on the process. For example, in epipolar rasterization for geometric filtering, one thread is allocated per left edge, while in the NCC/SIFT filter for photometric filtering, one thread is allocated per one candidate match. 
- *Thrust* handles on-device sorting, prefix scans, and compaction, _e.g._, grouping edges by spatial-grid cell (`sort_by_key`), segmenting refined hypotheses per left edge before merge/clustering, and picking one final stereo mate per left edge.

Optional host-side export (`retrieve_stereo_mates`, `retrieve_final_matches`, output files) is used for logging and visualization; the hot path can stay entirely on the device between stereo and temporal steps.

## Example
TODO


## Author
Chiang-Heng Chien (chiang-heng_chien@brown.edu)