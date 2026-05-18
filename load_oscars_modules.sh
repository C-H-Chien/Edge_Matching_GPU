#!/bin/bash
#clear

module_loaded() {
    module list 2>&1 | grep -q "$1"
}

module_loaded cmake/3.29.6-ocf3 || module load cmake/3.29.6-ocf3
module_loaded eigen/3.4.0-hggg || module load eigen/3.4.0-hggg
module_loaded boost/1.88.0-6ijj || module load boost/1.88.0-6ijj
module_loaded opencv/4.6.0s-22w7 || module load opencv/4.6.0s-22w7
module_loaded cuda/12.9.0-cinr || module load cuda/12.9.0-cinr