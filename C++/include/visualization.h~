// visualization

#ifndef __KYLE_VISUALIZATION__
#define __KYLE_VISUALIZATION__
#include "include_in_all.h"
using namespace std;
using namespace cv;
Mat create_red_img_from_segmentation_mat( Mat segmentations_cv_16U );
Mat draw_grid_points_on_image( Mat valid_img_16U, vector<vector<long> > grid_keypoints_train );
Mat display_superpixel_classifications( vector<orchard_classification> cur_superpixel_classifications, Mat input_img_rgb_uchar, Mat superpixel_img, vector<double> superpixel_dominant_angle, string img_name );
Mat draw_classifications( const Mat superpixel_image, Mat classifications );
Mat draw_apple_pixels( const Mat superpixel_image, const Mat raw_image, Mat classifications );
Mat draw_classification_probability( Mat superpixel_image, Mat percentages_by_type, int idx = 0 );
Mat draw_only_apple_classifications( Mat superpixel_image, Mat classifications );
#endif // __KYLE_VISUALIZATION__
