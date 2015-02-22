#ifndef __KYLE_MAIN__
#define __KYLE_MAIN__

#define PI 3.14159265

using namespace std;
using namespace cv;

// this gets the pixels that are valid in the training mask
Mat compute_valid_pixels( Mat input_img_rgb_uchar_train_mask ); c
// this gets the grid points from a superpixels image
vector<vector<long> > compute_grid_points_over_superpixels( Mat input_img ); c
// this gets the features at specified indices from an image
Mat compute_features_over_image( Mat input_img_rgb_uchar, vector<vector<long> > grid_keypoints, string feature_type ); c
// this creates the machine learning structure used to classify the features that were computed from the image
// KD_tree_kyle create_kd_tree( vector<Mat> features_over_image );
// this computes the superpixels throughout the image
Mat compute_superpixels( Mat input_img_rgb_uchar );
vector<orchard_classification> query_superpixel_features( vector<orchard_classification> combined_labels, Mat cur_features, const Mat combined_descriptors_double );
Mat multiply_matrix_by_255( Mat valid_img_8U );
Mat find_equal_16U( Mat superpixel_img, int compare_number_input );
Mat fill_binary_image( Mat valid_img_8U );
Mat get_mask_of_parts( Mat input_img_8UC3 );

//  
string type2str(int type);
Mat create_red_img_from_segmentation_mat( Mat segmentations_cv_16U );
Mat create_mat_from_array( const Mat input_img, vl_uint32* segmentation );
vector<KeyPoint> convert_vector_of_indices_to_keypoints( vector<vector<long>> all_keypoints_indices, const Mat mat )
Mat draw_grid_points_on_image( Mat valid_img_16U, vector<vector<long> > grid_keypoints_train )
Mat compute_invalid_pixels( Mat valid_img_16U, Mat input_img_rgb_uchar_train )
Mat display_superpixel_classifications( vector<orchard_classification> cur_superpixel_classifications, Mat input_img_rgb_uchar, Mat superpixel_img, vector<double> superpixel_dominant_angle, string img_name )
vector< Mat > split_features( Mat descriptors_object_8U, vector< vector< long > > grid_keypoints )
uchar average_elements(int *elements, int total_elements)
uchar std_elements(int *elements, int total_elements, uchar cur_average)
double average_elements_angle( int *elements, int total_elements)
double std_elements_angle( int *elements, int total_elements, double cur_average_angle )
Mat get_average_and_standard_over_patch( Mat input_img_rgb_uchar, vector< KeyPoint > grid_keypoints_train, vector<bool> is_angular_dimension, int compute_std )
Mat compute_color_features_over_image( Mat input_img_rgb_uchar, vector< KeyPoint > grid_keypoints_train )
Mat compute_std_by_row( Mat combined_descriptors, Mat row_mean )
vector<Mat> compute_mean_and_std( const Mat combined_descriptors_input )
vector<string> get_files_with_extension ( string dir_path, string extension  )
Mat normalize_mat( Mat input_mat, vector<Mat> norm_factors_mean_and_std )
Mat convert_grid_locations_to_mat( vector< vector< long > > grid_keypoints_train )
vector<vector<long> > convert_mat_to_grid_keypoints_vector( Mat grid_keypoints_train_combined )
vector<vector<long> > prune_keypoints( vector<vector<long> > grid_keypoints_invalid, int num_remaining )

#ifndef __KYLE_MAIN__
