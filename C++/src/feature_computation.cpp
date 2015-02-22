#include "utility.h"
#include "feature_computation.h"
#include "include_in_all.h"


using namespace std;
using namespace cv;

Mat threshold_uchar_above_250( Mat src_gray ) {
  int threshold_value = 250, max_BINARY_value = 255;
	Mat pixels_at_255;
	threshold( src_gray, pixels_at_255, threshold_value, max_BINARY_value, THRESH_BINARY );
  return pixels_at_255;
}

// feature computation
Mat compute_valid_pixels( Mat input_img_uchar_train_mask ) {
  // ret: 8U image : 1 == true
	Mat dst;
	Mat src_gray;
  int type_cur = input_img_uchar_train_mask.type();
  uchar depth = type_cur & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type_cur >> CV_CN_SHIFT);
  if ( depth == CV_8U && chans == 1 ) {
    src_gray = input_img_uchar_train_mask;
	} else if ( depth == CV_8U && chans == 3 ) {
    cvtColor( input_img_uchar_train_mask, src_gray, CV_BGR2GRAY );
  } else {
    throw runtime_error("valid pixels invalid type!");
  }
  Mat pixels_segmentation = threshold_uchar_above_250( src_gray );
	uchar * ptr_temp = (uchar *) pixels_segmentation.ptr();
	for ( int i = 0; i < pixels_segmentation.rows*pixels_segmentation.cols; i++ ) {
		ptr_temp[i] = ptr_temp[i] > 240 ? 1 : 0;
	}
	int erosion_size = 4;
	int erosion_type = MORPH_RECT;
	Mat element = getStructuringElement( erosion_type, Size( 2*erosion_size + 1, 2*erosion_size+1 ), Point( erosion_size, erosion_size ) );
  erode( pixels_segmentation, pixels_segmentation, element );
	pixels_segmentation.convertTo( pixels_segmentation, CV_8U );
	return pixels_segmentation;
}

vector<vector<long> > compute_grid_points_over_superpixels( Mat input_img_16U, int superpixel_img, int max_value ) {
  // TODO: ex: should set this up to output a Mat with < superpixel idx, x value, y value >
	vector< vector<long > > grid_points_over_superpixels( max_value, vector<long>() );
	int edge_buffer = 40;
	int step_size = 15;
	for( int i = edge_buffer; i < input_img_16U.rows - edge_buffer; i += step_size ) {
		for( int j = edge_buffer; j < input_img_16U.cols - edge_buffer; j += step_size ) {
      long pixel_index = j * ((long) input_img_16U.rows) + i;
      int superpixel_image_point_value = (int) input_img_16U.at<ushort>(i, j);
      if ( superpixel_image_point_value > 0 ) {
        int superpixel_idx_real = superpixel_image_point_value-1;
        grid_points_over_superpixels[superpixel_idx_real].push_back( pixel_index );
      }
		}
	}
	return grid_points_over_superpixels;
}

Mat compute_features_over_image( const Mat input_img_rgb_uchar, vector<KeyPoint> points_for_feature_computation, string feature_type ) {
	SurfDescriptorExtractor extractor;
	// FREAK extractor;
	Mat input_img_gray_uchar;
	cvtColor( input_img_rgb_uchar, input_img_gray_uchar, CV_RGB2GRAY );
	vector<Mat> all_superpixel_features;
	//
	fprintf( stderr, "feature points to compute:  %d\n", (int) points_for_feature_computation.size() );
  Mat descriptors_object_8U;
	extractor.compute( input_img_gray_uchar, points_for_feature_computation, descriptors_object_8U );
	fprintf( stderr, "after extracting features\n" );
	if ( descriptors_object_8U.rows != (int) points_for_feature_computation.size() ) {
		throw runtime_error("not as many descriptors as points");
	}
	return descriptors_object_8U;
}

uchar average_elements(int *elements, int total_elements) {
	int total = 0;
	for( int i = 0; i < total_elements; i++ ) {
		total += elements[i];
	}
	return (uchar) (total/total_elements);
}

uchar std_elements(int *elements, int total_elements, uchar cur_average) {
	double total_diff_sq = 0;
	double cad = ((double)cur_average);
	for( int i = 0; i < total_elements; i++ ) {
		double cur_diff = ((double)elements[i]) - cad;
		int cd_sq = pow( cur_diff, 2);
		total_diff_sq += cd_sq;
	}
	double temp = total_diff_sq / (double) total_elements;
	temp = sqrt(temp);
	if ( temp > 255 || temp < 0) throw runtime_error( "std computation has errors?!" );
	return ((uchar) temp);
}

double average_elements_angle( int *elements, int total_elements) {
	double total_x = 0, total_y = 0;
	for( int i = 0; i < total_elements; i++ ) {
		double cur_angle = ( 2 * ( (double) elements[i] ) ) * (PI/180);
		total_y += sin( cur_angle );
		total_x += cos( cur_angle );
	}
	double avg_y = total_y / ( (double) total_elements );
	double avg_x = total_x / ( (double) total_elements );
	double avg_angle = atan2( avg_y, avg_x );
	return avg_angle;
}

double std_elements_angle( int *elements, int total_elements, double cur_average_angle ) {
	if ( cur_average_angle > PI || cur_average_angle < (-1*PI) ) throw runtime_error( "A : Angle computations have an error" );
	double total_delta_angle_sq = 0;
	for( int i = 0; i < total_elements; i++ ) {
		double cur_angle = ( 2 * ( (double) elements[i] ) ) * (PI/180);
		if ( cur_angle > PI ) cur_angle -= 2*PI;
		if ( cur_angle > PI || cur_angle < (-1*PI) ) throw runtime_error( "B: Angle computations have an error" );
		double diff = abs(cur_angle - cur_average_angle);
		if ( diff > PI ) diff = (2*PI) - diff;
		total_delta_angle_sq += pow( diff, 2 );
	}
	total_delta_angle_sq /= total_elements;
	total_delta_angle_sq = sqrt( total_delta_angle_sq );
	return total_delta_angle_sq;
}

Mat get_average_and_standard_over_patch( Mat input_img_rgb_uchar, vector< KeyPoint > grid_keypoints_train, vector<bool> is_angular_dimension, int compute_std ) {
	int min_dim = -7;
	int max_dim = 7;
	int side_length = ( max_dim - min_dim + 1 );
	int total_elements = side_length * side_length;
	int total_angular_dimensions = 0;
	for (int i = 0; i < (int) is_angular_dimension.size(); i++) {
		if(is_angular_dimension[i]) total_angular_dimensions++;
	}
	Mat cur_mean_and_std;
	if ( compute_std > 0) {
		cur_mean_and_std = Mat::zeros( (int) grid_keypoints_train.size(), 6+total_angular_dimensions, CV_8U );
	} else {
		cur_mean_and_std = Mat::zeros( (int) grid_keypoints_train.size(), 3+total_angular_dimensions, CV_8U );
	}
	int elements[total_elements];
	int angular_methods_used = 0;
	int counter = 0;
	for( int a = 0; a < 3; a++ ) {
		for( int i = 0; i < (int) grid_keypoints_train.size(); i++ ) {
			counter = 0;
			for( int j = min_dim; j <= max_dim; j++ ) {
				for( int k = min_dim; k <= max_dim; k++ ) {
					Vec3b temp2 = input_img_rgb_uchar.at<Vec3b>( grid_keypoints_train[i].pt.y + j, grid_keypoints_train[i].pt.x + k );
					elements[counter++] = (int) temp2[a];
				}
			}
			uchar cur_average, cur_std;
			if ( is_angular_dimension[a] ) {
				double cur_average_angle = average_elements_angle(elements, total_elements);
				double cur_std_double = std_elements_angle( elements, total_elements, cur_average_angle );
				double cur_std_uchar_prep = cur_std_double * 255/(PI);
				if ( cur_std_uchar_prep > 255 || cur_std_uchar_prep < 0 ) throw runtime_error( "My idea of angles is wrong" );
				cur_std = (uchar) cur_std_uchar_prep;
				double cur_average_x_prep = cos( cur_average_angle );
				double cur_average_y_prep = sin( cur_average_angle );
				cur_mean_and_std.at<uchar>( i, a + angular_methods_used ) = (uchar) ((cur_average_x_prep+1) * (255/2.1));
				cur_mean_and_std.at<uchar>( i, a + 1 + angular_methods_used ) = (uchar) ((cur_average_y_prep+1) * (255/2.1));
				if ( compute_std > 0) {
					cur_mean_and_std.at<uchar>( i, a + (3 + total_angular_dimensions) ) = cur_std;
				}
			} else {
				cur_average = average_elements(elements, total_elements);
				cur_std = std_elements(elements, total_elements, cur_average);
				cur_mean_and_std.at<uchar>( i, a + angular_methods_used ) = cur_average;
				if ( compute_std > 0) {
					cur_mean_and_std.at<uchar>( i, a + (3 + total_angular_dimensions) ) = cur_std;
				}
			}
		}
		if ( is_angular_dimension[a] ) {
			angular_methods_used++;
		}
	}
	return cur_mean_and_std;
}

Mat compute_color_features_over_image( Mat input_img_rgb_uchar, vector< KeyPoint > grid_keypoints_train ) {
	// rgb = 1; HSV = 2;
  int color_feature_type(2);	
	int rgb_int(1);
	int hsv_int(2);
	Mat all_mask_color_features_train_combined;
	// 1. the standard deviation of color
	// 2. perhaps 3-scale sift descriptors for each color dimension
	vector<bool> is_angular_dimension(3, false);
	Mat average_color = get_average_and_standard_over_patch( input_img_rgb_uchar, grid_keypoints_train, is_angular_dimension, 0 );
	if ( color_feature_type == rgb_int ) {
		all_mask_color_features_train_combined = average_color;
	} else if ( color_feature_type == hsv_int ) {
		Mat img_hsv;
		cvtColor( input_img_rgb_uchar, img_hsv, CV_RGB2HSV );
		is_angular_dimension[0] = true;
		Mat average_hsv = get_average_and_standard_over_patch( input_img_rgb_uchar, grid_keypoints_train, is_angular_dimension, 0 );
    fprintf( stderr, "a\n" );
		hconcat( average_color, average_hsv, all_mask_color_features_train_combined );
    fprintf( stderr, "average_color: %d/%d, average_hsv: %d/%d, all_mask_color_features_train_combined: %d/%d\n", average_color.rows, average_color.cols, average_hsv.rows, average_hsv.cols, all_mask_color_features_train_combined.rows, all_mask_color_features_train_combined.cols );
		fprintf( stderr, "type of hsv: %s\n", type2str( img_hsv.type() ).c_str() );
	}
	return all_mask_color_features_train_combined;
}

Mat compute_std_by_row( Mat combined_descriptors, Mat row_mean ) {
  verify_type( combined_descriptors.type(), CV_64F );
	Mat std_by_dimension(1, combined_descriptors.cols, CV_64F, Scalar(1));
	for( int i = 0; i < combined_descriptors.rows; i++ ) {
		for( int j = 0; j < combined_descriptors.cols; j++ ) {
			double cur_diff = ((double)combined_descriptors.at<double>(i,j)) - row_mean.at<double>(0, j);
      double temp_pow = pow(cur_diff, 2);
			std_by_dimension.at<double>( 0, j ) += temp_pow;
		}
	}
	Mat rows_mat( 1, combined_descriptors.cols, CV_64F, combined_descriptors.rows );
	divide( std_by_dimension, rows_mat, std_by_dimension );
  sqrt( std_by_dimension, std_by_dimension );
	std_by_dimension.convertTo(std_by_dimension, CV_64F);

  cerr << "row mean:   " << std_by_dimension << endl;
  cerr << "std_by_dimension:   " << std_by_dimension << endl;
	return std_by_dimension;
}

vector<Mat> compute_mean_and_std( const Mat combined_descriptors ) {
  verify_type( combined_descriptors.type(), CV_64F );
	Mat row_mean;
	reduce( combined_descriptors, row_mean, 0, CV_REDUCE_AVG );
	row_mean.convertTo(row_mean, CV_64F);
  check_for_nan( row_mean );
	Mat std_by_dimension = compute_std_by_row( combined_descriptors, row_mean );
  check_for_nan( std_by_dimension );
	vector<Mat> mean_and_std;
	mean_and_std.push_back( row_mean );
	mean_and_std.push_back( std_by_dimension );
	return mean_and_std;
}

Mat normalize_mat( const Mat input_mat, vector<Mat> norm_factors_mean_and_std ) {
  verify_type( input_mat.type(), CV_64F );
	fprintf( stderr, "input_mat: rows: %d: cols: %d\n", input_mat.rows, input_mat.cols );
	fprintf(stderr, "norm_factors_mean_and_std[0]: rows: %d: cols: %d\n", norm_factors_mean_and_std[0].rows, norm_factors_mean_and_std[0].cols );
	fprintf(stderr, "norm_factors_mean_and_std[1]: rows: %d: cols: %d\n", norm_factors_mean_and_std[1].rows, norm_factors_mean_and_std[1].cols );
	fprintf( stderr, "norm_factors_mean_and_std.size(): %d\n", (int) norm_factors_mean_and_std.size() );
	input_mat.convertTo( input_mat, CV_64F );
	Mat normalized;
	Mat subtract_mat = Mat::zeros( input_mat.rows, input_mat.cols, CV_64F );
	Mat divide_mat = Mat::ones( input_mat.rows, input_mat.cols, CV_64F );
	fprintf( stderr, "subtract_mat: rows: %d: cols: %d\n", subtract_mat.rows, subtract_mat.cols );
	fprintf( stderr, "divide_mat: rows: %d: cols: %d\n", divide_mat.rows, divide_mat.cols );
	Mat mean_mat = norm_factors_mean_and_std[0];
	mean_mat.convertTo(mean_mat, CV_64F);
	Mat std_mat = norm_factors_mean_and_std[1];
	std_mat.convertTo(std_mat, CV_64F);
  fprintf( stderr, "input_mat : r: %d : c: %d\n", input_mat.rows, input_mat.cols );
  fprintf( stderr, "mean_mat : r: %d : c: %d\n", mean_mat.rows, mean_mat.cols );
  fprintf( stderr, "std_mat : r: %d : c: %d\n", std_mat.rows, std_mat.cols );
  if ( input_mat.cols != mean_mat.cols || input_mat.cols != std_mat.cols ) throw runtime_error("normalization size problem...");
	for(int i = 0; i < input_mat.rows; i++ ) {
		for(int j = 0; j < input_mat.cols; j++ ) {
			subtract_mat.at<double>(i, j) = ( mean_mat.at<double>(0, j) );
      double cur_std = std_mat.at<double>(0, j);
      if ( cur_std != 0) {
        divide_mat.at<double>(i, j) = cur_std;
      }
    }
	}
	subtract(input_mat, subtract_mat, normalized);
	divide(normalized, divide_mat, normalized);
	return normalized;
}

vector<vector<long> > convert_mat_to_grid_keypoints_vector( Mat grid_keypoints_train_combined ) {
	int count_temp = 1;
	for( int j = 0; j < grid_keypoints_train_combined.rows; j++ ) {
		long cur_el = (long) grid_keypoints_train_combined.at< int >( j, 0 );
		if( j < grid_keypoints_train_combined.rows-1 && cur_el == -1 ) {
			count_temp += 1;
		}
	}
	int idx = 0;
	vector< vector<long> > grid_keypoints_train( count_temp, vector<long>() );
	for( int j = 0; j < grid_keypoints_train_combined.rows; j++ ) {
		long cur_el = (long) grid_keypoints_train_combined.at< int >( j, 0 );
		if( j < grid_keypoints_train_combined.rows-1 && cur_el == -1 ) {
			idx++;
		} else {
			grid_keypoints_train[idx].push_back( cur_el );
		}
	}
	return grid_keypoints_train;
}

void filter_indices_by_valid_superpixels( const Mat &rgb_image_8U, vector<vector<long> > &grid_indices, const Mat &superpixels_to_check) {
  vector<int> superpixels_to_keep( (int) grid_indices.size(), 1 );
  if ( superpixels_to_check.rows != (int) grid_indices.size() ) throw runtime_error( "( superpixels_to_check.rows != (int) grid_indices.size() )" );
  verify_type( superpixels_to_check.type(), CV_8U );
  for ( int j = 0; j < (int) superpixels_to_check.rows; j++ ) { // for each superpixel
    uchar check_superpixel_b = superpixels_to_check.at<uchar>(j, 0);
    if ( !( (int) check_superpixel_b ) ) {
      grid_indices[j] = vector<long>();
    }
  }
}

void prune_keypoints( vector<vector<long> > &grid_keypoints_invalid, int num_remaining ) {
	// compute the total number of features in the vector of vectors list
	int total_keypoints = get_total_keypoints( grid_keypoints_invalid );
  int step = 1;
  if ( total_keypoints >= num_remaining ) {
	  step = total_keypoints / num_remaining;
  } else {
    step = 1;
  }
	int total_indices_final = 0, removed = 0;
	for( int i = 0; i < total_keypoints; i+=step ) total_indices_final++;
	// get random numbers between 0 and (total_keypoints-1)
	long total_counter = 0;
	for (int i = 0; i < (int) grid_keypoints_invalid.size(); i++ ) {
		for ( int j = (int) grid_keypoints_invalid[i].size()-1; j >= (int) 0; j-- ) {
			if ( total_counter % step != 0 ) {
        grid_keypoints_invalid[i].erase(grid_keypoints_invalid[i].begin()+j);
        removed++;
        if ( (total_keypoints-removed) == num_remaining) return;
			}
      total_counter++;
		}
	}
}

Mat get_superpixels_with_high_value( Mat input_img_rgb_uchar_train_temp, const vector<vector<long> > grid_indices, Mat superpixel_image_input ) {
  float resize_factor(1);
  float limitting_value(80);
  Mat img_gray;
  cvtColor(input_img_rgb_uchar_train_temp, img_gray, CV_BGR2GRAY );
  int rows = input_img_rgb_uchar_train_temp.rows;
  
  int num_superpixels = compute_max_value( superpixel_image_input );
  Mat superpixels_value_is_valid( (int) grid_indices.size(), 1, CV_8U, Scalar(0) );
  int cur_idx = -1;
  vector<int> count_pixels( (int) grid_indices.size(), 0 );
  vector<int> count_valid( (int) grid_indices.size(), 0 );
  uchar *gray_img_ptr = (uchar *) img_gray.ptr();
  ushort *si_ptr = (ushort *) superpixel_image_input.ptr();
  for ( int i = 0; i < img_gray.rows * img_gray.cols; i++  ) {
    ushort index_cur = si_ptr[i]-1;
    count_pixels[index_cur]++;
    float gray_value = (float) gray_img_ptr[i];
    if ( gray_value > limitting_value ) {
      count_valid[index_cur]++;
    }
  }
  for ( int i = 0; i < (int) count_valid.size(); i++ ) {
    float percent_valid = ( (float) count_valid[i] ) / ( (float) count_pixels[i] );
    if ( percent_valid > 0.8 ) {
      superpixels_value_is_valid.at<uchar>( i, 0 ) = 1;
    }
  }
  return superpixels_value_is_valid;
}
