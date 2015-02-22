#ifndef __KYLE_SETUP_FEATURE_SETS__
#define __KYLE_SETUP_FEATURE_SETS__

#include <sys/stat.h>

using namespace cv;
using namespace std;

// complete_color_features: 
struct supervised_feature_set {
	vector<Mat> complete_texture_features;
	vector<Mat> complete_color_features;
  vector<Mat> complete_labels;
  supervised_feature_set( int length_features ) {
    complete_texture_features = vector< Mat >(length_features, Mat(0, 0, CV_8U));
    complete_color_features = vector< Mat >(length_features, Mat(0, 0, CV_8U));
    complete_labels = vector< Mat >(length_features, Mat(0, 0, CV_8U));
  }
};

// need: feature space, labels, pca transformation, 
// one for each LOO set, the full classifier
struct classifier_features {
  Mat feature_space;
  PCA pca_transform;
  Mat combined_labels;
  vector<Mat> norm_factors_color, norm_factors_texture;
  // could have different types of classifiers in here as well
};

struct features_computed {
  vector<vector<long> > grid_indices;
  vector<KeyPoint> grid_keypoints;
  Mat superpixel_image;
  Mat texture_features;
  Mat color_features;
  Mat superpixels_value_is_valid_8U;
};

struct features_full_images {
  // TODO: This will also contain data on the images that have black backgrounds, features pertaining to the trunks of trees, etc...
	vector<vector<vector<long> > > indices;
	vector<Mat> all_texture_features;
	vector<Mat> all_color_features;
	vector<Mat> superpixel_imgs;
  vector<Mat> valid_imgs;
  vector<vector<KeyPoint> > keypoints;
  vector<Mat> valid_superpixels;
  features_full_images(  ) { }
  features_full_images( int length_features ) {
    indices = vector<vector<vector<long> > >(length_features);
    all_texture_features = vector<Mat>(length_features);
    all_color_features = vector<Mat>(length_features);
    superpixel_imgs = vector<Mat>(length_features);
    valid_superpixels = vector<Mat>(length_features);
    valid_imgs = vector<Mat>(length_features);
    keypoints = vector<vector<KeyPoint> >(length_features);
  }
};



struct classifier_features setup_classifier_features( struct supervised_feature_set feature_set_input, int image_loo = -1 );
struct supervised_feature_set setup_supervised_feature_set( vector<string> raw_img_paths, vector<string> mask_img_paths, int compute_invalid_features, struct filepaths_apple_project fpaths, string root_saving_string, float resize_factor );
struct features_computed compute_features_single_image( Mat input_img_rgb_uchar_train, string feature_type );
struct features_full_images setup_features_full_images( vector<string> raw_img_paths, float resize_factor, string feature_type, struct filepaths_apple_project fpaths );
Mat transform_indices_to_mat( vector<vector<long> > indices, int num_superpixels );
Mat transform_keypoints_to_mat( vector<KeyPoint> keypoints, int num_superpixels );
vector<vector<long> > transform_mat_to_indices( Mat indices_mat, int num_superpixels );
int save_features_full_images( string path_save_dir, struct features_full_images features_input, struct filepaths_apple_project fpaths );
struct features_full_images load_features_full_images( string path_save_dir, struct filepaths_apple_project fpaths );
#endif // __KYLE_SETUP_FEATURE_SETS__
