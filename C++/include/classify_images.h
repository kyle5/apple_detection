#include "include_in_all.h"
#include "setup_feature_sets.h"

#ifndef __KYLE_CLASSIFICATION__
#define __KYLE_CLASSIFICATION__

using namespace std;
using namespace cv;
struct classification_result {
  Mat superpixel_image;
  vector<vector<uchar> > classifications;
};

void filter_features_by_valid_superpixels( Mat &texture_features, Mat &color_features, vector<vector<long> > &grid_indices, const vector<Mat> &superpixels_to_check_vec );
vector<vector<uchar> > query_superpixel_features( const Mat query_features, const classifier_features classification_struct, const vector< vector<long> > &grid_indices );
struct classification_result classify_image( struct classifier_features cf_struct, std::string raw_img_path, struct features_full_images features_cur, string feature_type, vector<Mat> &superpixels_to_check_vec, int i = -1 );
cv::Mat calculate_percentage_by_type( const cv::Mat superpixel_image, std::vector<std::vector<uchar> > classifications );
int orchard_class_to_int( orchard_classification cur_class );
int get_total_indices( vector< vector< long > > grid_indices );
#endif // __KYLE_CLASSIFICATION__
