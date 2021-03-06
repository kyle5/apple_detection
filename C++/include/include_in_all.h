#ifndef __INCLUDE_IN_ALL__
#define __INCLUDE_IN_ALL__
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <algorithm>
#include <vector>
#include <stdexcept>
#include <iostream>

#include <stdio.h>
#include <dirent.h>
#include <iostream>     // std::cout
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include <climits>

#include <sys/types.h>
#include <sys/stat.h>

#define PI 3.14159265

extern "C" {
    #include "vl/generic.h"
    #include "vl/slic.h"
}

enum orchard_classification { APPLE=1, LEAF=2, TREE_TRUNK=3, TOO_DARK=4, VERY_TOP=5, TOP=6, BOTTOM=7, VERY_BOTTOM=8 };
#endif // __INCLUDE_IN_ALL__
