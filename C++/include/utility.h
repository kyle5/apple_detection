//  
#include "include_in_all.h"
using namespace std;
using namespace cv;
#ifndef __UTILITY__
#define __UTILITY__

vector<Mat> calculate_superpixel_indices( Mat superpixel_image );
Mat calculate_maximal_classifications( Mat percentages_by_type, float min_probability );
void combine_features_vertically_ref( cv::Mat &full, cv::Mat add_on );
void combine_features_horizontally_ref( cv::Mat &full, cv::Mat add_on );
std::string type2str(int type);
int compute_max_value( cv::Mat input_mat );
cv::Mat create_mat_from_array( const cv::Mat input_img, vl_uint32* segmentation );
std::vector<cv::KeyPoint> convert_vector_of_indices_to_keypoints( std::vector<std::vector<long> > all_keypoints_indices, const cv::Mat mat );
cv::Mat compute_invalid_pixels( cv::Mat valid_img_16U, cv::Mat input_img_rgb_uchar_train );
cv::Mat multiply_matrix_by_255( cv::Mat valid_img_8U );
cv::Mat find_equal_16U( cv::Mat superpixel_img, int compare_number_input );
cv::Mat fill_binary_image( cv::Mat valid_img_8U );
cv::Mat get_mask_of_parts( cv::Mat input_img_8UC3 );
std::vector<std::string> get_files_with_extension( std::string dir_path, std::string extension  );
long get_total_keypoints( std::vector<std::vector<long> > grid_keypoints_invalid );
cv::Mat convert_grid_locations_to_mat( std::vector< std::vector< long > > grid_keypoints_train );
void dilate_and_fill_image_ref( cv::Mat &input_img_8U_ref );
void increment_superpixel_img_ref( Mat &superpixel_img, int compute_max_value, int &max_value );
Mat read_image_and_flip_90( string raw_filepath, float resize_factor );
Mat count_labels( Mat cur_labels );
int count_two_depth_vector( const vector<vector<long> > input_vector );
void copy_mat_to_10_chunks( Mat combined_descriptors, vector<Mat> &descriptor_train_chunks );
vector<string> correlate_image_paths( vector<int> match_numbers, vector<int> to_be_matched_numbers, vector<string> to_be_matched_paths );
void verify_type( int type_input, uchar type_check_val );
void check_for_nan( Mat input_mat_64F );
template <typename T> Mat draw_grid_superpixels( Mat superpixel_image2, const vector< vector< T > > &grid_indices );
#endif
