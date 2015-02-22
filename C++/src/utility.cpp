#include "utility.h"
#include "include_in_all.h"

using namespace std;
using namespace cv;

string type2str(int type) {
  string r;
  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);
  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }
  r += "C";
  r += (chans+'0');
  return r;
}

void check_for_nan( Mat input_mat_64F ) {
  verify_type( input_mat_64F.type(), CV_64F );
  for ( int i = 0; i < input_mat_64F.rows; i++ ) {
    for ( int j = 0; j < input_mat_64F.cols; j++ ) {
      double row_mean_cur = input_mat_64F.at<double>( i, j );
      if ( row_mean_cur != row_mean_cur ) {
        cerr << "There is an NaN number! " << row_mean_cur << " " << endl;
        throw runtime_error("NaN number!");
      }
    }
  }
}

void verify_type( int type_input, uchar type_check_val ) {
  uchar depth = type_input & CV_MAT_DEPTH_MASK;
  if ( depth != type_check_val ) {
    cerr << "The actual type is: " << type2str( type_input ) << endl;
    throw runtime_error( "The types do not match!" );
  }
}

void combine_features_horizontally_ref( Mat &full, Mat add_on ) {
  if (full.rows == 0 || full.cols == 0 ) full = add_on;
  else {
    Mat temp;
    hconcat( full, add_on, temp );
    full = temp;
  }
}
void combine_features_vertically_ref( Mat &full, Mat add_on ) {
  if (full.rows == 0 || full.cols == 0 ) full = add_on;
  else if ( add_on.rows == 0 || add_on.cols == 0 ) return;
  else {
    Mat temp;
    vconcat( full, add_on, temp );
    full = temp;
  }
}

void increment_superpixel_img_ref( Mat &superpixel_img, int compute_max_value, int &max_value ) {
  if ( CV_16U != (superpixel_img.type() & CV_MAT_DEPTH_MASK) ) throw runtime_error( "The superpixel image is not ushort ?!" );
  ushort *si_ptr = (ushort *) superpixel_img.ptr();
  for ( int i = 0; i < superpixel_img.rows*superpixel_img.cols; i++ ) {
    si_ptr[i]++;
    if ( compute_max_value != -1 && si_ptr[i] > max_value ) max_value = si_ptr[i];
  }
}

Mat create_mat_from_array( const Mat input_img, vl_uint32* segmentation ) {
	fprintf( stderr, "start of create_mat_from_array\n" );
	Mat segmentations_cv_16U = Mat::zeros( input_img.rows, input_img.cols, CV_16U );
  for (int i = 0; i < input_img.rows; ++i) {
    for (int j = 0; j < input_img.cols; ++j) {
      segmentations_cv_16U.at<unsigned short>(i, j) = ( unsigned short ) segmentation[j + input_img.cols*i];
    }
  }
	fprintf( stderr, "end of create_mat_from_array\n" );
  return segmentations_cv_16U;
}

template <typename T> Mat draw_grid_superpixels( Mat superpixel_image2, const vector< vector< T > > &grid_indices ) {
  Mat visible_superpixels( superpixel_image2.rows, superpixel_image2.cols, CV_8U, Scalar(0) );
  verify_type( superpixel_image2.type(), CV_16U );
  for ( int ii = 0; ii < superpixel_image2.rows; ii++ ) {
    for ( int jj = 0; jj < superpixel_image2.cols; jj++ ) {
      int sup_idx = ((int)superpixel_image2.at<ushort>( ii, jj )) - 1;
      if ( sup_idx >= grid_indices.size() ) throw runtime_error("indexation error");
      if ( grid_indices[sup_idx].size() > 0 ) {
        visible_superpixels.at<uchar>(ii, jj) = 255;
      }
    }
  }
  return visible_superpixels;
}

template Mat draw_grid_superpixels<long>( Mat superpixel_image2, const vector< vector< long > > &grid_indices );
template Mat draw_grid_superpixels<uchar>( Mat superpixel_image2, const vector< vector< uchar > > &grid_indices );

vector<KeyPoint> convert_vector_of_indices_to_keypoints( vector<vector<long> > all_keypoints_indices, const Mat mat ) {
	int r = mat.rows;
	vector<KeyPoint> vector_keypoints;
	for ( long i = 0; i < (long) all_keypoints_indices.size(); i++ ) {
		for ( long j = 0; j < (long) all_keypoints_indices[i].size(); j++) {
			long ci = all_keypoints_indices[i][j];
			KeyPoint cur_keypoint( ci / r, ci % r, 10 );
			vector_keypoints.push_back( cur_keypoint );
		}
	}
	return vector_keypoints;
}

Mat read_image_and_flip_90(string raw_filepath, float resize_factor ) {
  // load image
  Mat input_img_rgb_uchar_train = imread( raw_filepath );
  // resize image
  resize( input_img_rgb_uchar_train, input_img_rgb_uchar_train, Size(), resize_factor, resize_factor, INTER_NEAREST );
  // turn to upright
  transpose( input_img_rgb_uchar_train, input_img_rgb_uchar_train );
  flip( input_img_rgb_uchar_train, input_img_rgb_uchar_train, 0 );
  return input_img_rgb_uchar_train;
}

Mat compute_invalid_pixels( Mat valid_img_16U, Mat input_img_rgb_uchar_train ) {
	Mat input_img_gray_uchar_train;
	cvtColor( input_img_rgb_uchar_train, input_img_gray_uchar_train, CV_RGB2GRAY );
	Mat invalid_img_16U = Mat::zeros( valid_img_16U.rows, valid_img_16U.cols, CV_16U );
	for ( int i = 0; i < valid_img_16U.rows; i++ )
	for ( int j = 0; j < valid_img_16U.cols; j++ )
	{
		bool condition_1 = valid_img_16U.at<ushort>(i, j) < 1;
		bool condition_2 = input_img_gray_uchar_train.at<uchar>(i, j) > 25;
		if ( condition_1 && condition_2 ) {
			invalid_img_16U.at<ushort>(i, j) = 1;
		}
	}
	Mat invalid_img_8U;
	invalid_img_16U.convertTo(invalid_img_8U, CV_8U);
	Mat invalid_img_8U_temp = multiply_matrix_by_255( invalid_img_8U );
	
	int erosion_size = 10;
	int erosion_type = MORPH_RECT; 
	Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
	Mat invalid_img_8U_eroded, invalid_img_16U_eroded;
  erode( invalid_img_8U, invalid_img_8U_eroded, element );
	invalid_img_8U_temp = multiply_matrix_by_255( invalid_img_8U_eroded );
	invalid_img_8U_eroded.convertTo(invalid_img_16U_eroded, CV_16U);
	return invalid_img_16U_eroded;
}

Mat multiply_matrix_by_255( Mat valid_img_8U ) {	
	Mat multiplier( valid_img_8U.rows, valid_img_8U.cols, CV_8U, Scalar(255) );
	Mat output_8U;
	multiply( valid_img_8U, multiplier, output_8U );
	return output_8U;
}

Mat find_equal_16U( Mat superpixel_img, int compare_number_input ) {
	Mat valid_this_superpixel_16U;
	Mat this_superpixel_number( superpixel_img.rows, superpixel_img.cols, CV_16U, compare_number_input );
	compare( superpixel_img, this_superpixel_number, valid_this_superpixel_16U, CMP_EQ );
	valid_this_superpixel_16U.convertTo( valid_this_superpixel_16U, CV_16U );
	return valid_this_superpixel_16U;
}

vector<string> get_files_with_extension( string dir_path, string extension  ) {
	vector< string > mask_filepaths;
	DIR* dirFile = opendir( dir_path.c_str() );
	if ( dirFile ) {
		struct dirent* hFile;
		while (( hFile = readdir( dirFile )) != NULL )  {
			if ( !strcmp( hFile->d_name, "."  )) continue;
			if ( !strcmp( hFile->d_name, ".." )) continue;
			if ( ( hFile->d_name[0] == '.' )) continue;
      cerr << " cur d_name : " << string ( hFile->d_name ) << endl;
  		if ( strstr( hFile->d_name, extension.c_str() ) ) {
				string cur_file( hFile->d_name );
				mask_filepaths.push_back( cur_file );
			}
		}
		closedir( dirFile );
	}
	return mask_filepaths;
}

Mat convert_grid_locations_to_mat( vector< vector< long > > grid_keypoints_train ) {
	int total_size_temp = 0;
	for( int j = 0; j < grid_keypoints_train.size(); j++ ) {
		total_size_temp += (j == grid_keypoints_train.size()) ? (int) grid_keypoints_train[j].size() : (int) grid_keypoints_train[j].size() + 1;
	}
	Mat grid_keypoints_train_combined( total_size_temp, 1, CV_32S );
	long cur_place = 0;
	for( int j = 0; j < grid_keypoints_train.size(); j++ ) {
		for( int k = 0; k < grid_keypoints_train[j].size(); k++ ) {
			grid_keypoints_train_combined.at< int >( cur_place, 0 ) = (int) grid_keypoints_train[j][k];
			cur_place++;
		}
		if (j < (int)grid_keypoints_train.size()-1) grid_keypoints_train_combined.at< int >(cur_place, 0) = -1;
	}
	return grid_keypoints_train_combined;
}

long get_total_keypoints( vector<vector<long> > grid_keypoints_invalid ) {
	long total_keypoints = 0;
	for ( long i = 0; i < (long) grid_keypoints_invalid.size(); i++ ) {
		total_keypoints += (long) grid_keypoints_invalid[i].size();
	}
	return total_keypoints;
}

Mat get_mask_of_parts( Mat input_img_8UC3 ) {
  imwrite( "input_img_parts.ppm", input_img_8UC3 );
  Mat mask_of_parts = Mat::zeros( input_img_8UC3.rows, input_img_8UC3.cols, CV_8U );
  Mat very_top = mask_of_parts.clone();
  Mat top = mask_of_parts.clone();
  Mat bottom = mask_of_parts.clone();
  Mat very_bottom = mask_of_parts.clone();
  for ( int i = 0; i < input_img_8UC3.rows; i++ ) {
    for ( int j = 0; j < input_img_8UC3.cols; j++ ) {
      Vec3b cv = input_img_8UC3.at<Vec3b>( i, j );
      uchar b = cv[0]; uchar g = cv[1]; uchar r = cv[2];
      if( b > 250 && g < 5 && r < 5 ) {
        // blue
        very_bottom.at<uchar>(i, j) = 255;
      } else if( b < 1 && g > 254 && r < 1 ) {
        // green
        bottom.at<uchar>(i, j) = 255;
      } else if( b < 1 && g < 1 && r > 253 ) {
        // red
        top.at<uchar>(i, j) = 255;
      } else if( b < 5 && g > 250 && r > 250 ) {
        // yellow
        very_top.at<uchar>(i, j) = 255;
      }
    }
  }
  very_bottom = fill_binary_image( very_bottom );
  bottom = fill_binary_image( bottom );
  top = fill_binary_image( top );
  very_top = fill_binary_image( very_top );
  // Note : Other methods should use the values of 5 - 8 as well to mark specific parts of the apples
  for ( int i = 0; i < input_img_8UC3.rows; i++ )
  for ( int j = 0; j < input_img_8UC3.cols; j++ )
  {
    if( very_bottom.at<uchar>(i, j) == 1 ) {
      mask_of_parts.at<uchar>(i, j) = 8;
    } else if( bottom.at<uchar>(i, j) == 1 ) {
      mask_of_parts.at<uchar>(i, j) = 7;
    } else if( top.at<uchar>(i, j) == 1 ) {
      mask_of_parts.at<uchar>(i, j) = 6;
    } else if( very_top.at<uchar>(i, j) == 1 ) {
      mask_of_parts.at<uchar>(i, j) = 5;
    }
  }
  imwrite( "very_bottom.png", very_bottom );
  imwrite( "bottom.png", bottom );
  imwrite( "top.png", top );
  imwrite( "very_top.png", very_top );
  return mask_of_parts;
}

void dilate_and_fill_image_ref( Mat &input_img_8U_ref ) {
  // note output is 255|0
  int dilation_size = 5;
  int dilation_type = MORPH_RECT;
  Mat element = getStructuringElement( dilation_type, Size( 2*dilation_size + 1, 2*dilation_size+1 ), Point( dilation_size, dilation_size ) );
  cerr << "The type before is : " << type2str( input_img_8U_ref.type() ) << endl;
  dilate( input_img_8U_ref, input_img_8U_ref, element );
  cerr << "The type after is : " << type2str( input_img_8U_ref.type() ) << endl;
  input_img_8U_ref = fill_binary_image( input_img_8U_ref );
}

Mat fill_binary_image( Mat valid_img_8U ) {
  // note output is 255|0
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  Mat valid_img_8U_temp = valid_img_8U.clone();
  findContours( valid_img_8U_temp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE );
  Mat singleLevelHoles = Mat::zeros(valid_img_8U.size(), valid_img_8U.type());
  for(vector<Vec4i>::size_type idx=0; idx<hierarchy.size(); ++idx) {
    if(hierarchy[idx][3] != -1) {
      drawContours(singleLevelHoles, contours, idx, Scalar::all(255), CV_FILLED, 8, hierarchy);
    }
  }
	for( int i = 0; i < valid_img_8U_temp.rows; i++  ) {
    for( int j = 0; j < valid_img_8U_temp.cols; j++  ) {
      uchar org_val = valid_img_8U_temp.at<uchar>(i, j);
      uchar new_val = singleLevelHoles.at<uchar>(i, j);
      if( org_val > 0 || new_val > 0 ) valid_img_8U_temp.at<uchar>(i, j) = 1;
    }
	}
  return valid_img_8U_temp;
}

int compute_max_value( const Mat input_mat ) {
  if ( (input_mat.type() & CV_MAT_DEPTH_MASK) != CV_16U ) throw runtime_error("max value only works for unsigned short images!");
  ushort max_value = 0;
  ushort *input_mat_ptr = (ushort *) input_mat.ptr();
  for ( int i = 0; i < input_mat.rows*input_mat.cols; i++ )
    if ( input_mat_ptr[ i ] > max_value )
      max_value = input_mat_ptr[ i ];
  return (int) max_value;
}

vector<Mat> calculate_superpixel_indices( Mat superpixel_image ) {
  int max_value = compute_max_value( superpixel_image );
  vector<Mat> all_indices( max_value );
  for ( int i = 0; i < max_value; i++ ) {
    Mat indices_cur = superpixel_image == i+1;
    all_indices[i] = indices_cur;
  }
  return all_indices;
}

Mat calculate_maximal_classifications( Mat percentages_by_type, float min_probability ) {
  // for each superpixel
  Mat classifications = Mat( percentages_by_type.rows, 1, CV_8U, Scalar(4) );
  int i, j;
  cout << "The type is : percentages_by_type.type: " << type2str( percentages_by_type.type() ) << endl;
  for ( i = 0; i < percentages_by_type.rows; i++ ) {
    // get the last idx : iow calculate the classification of the superpixel
    uchar cur_classification_idx = 0;
    float max_value = -1;
    for ( j = 0; j < percentages_by_type.cols; j++ ) {
      float cur_value = percentages_by_type.at<float>( i, j );
      if ( cur_value > min_probability && cur_value > max_value ) {
        max_value = cur_value;
        cur_classification_idx = (uchar) j + 1;
      }
    }
    if ( cur_classification_idx == 0 ) classifications.at<uchar>(i, 0) = 4;
    else classifications.at<uchar>(i, 0) = cur_classification_idx;
  }
  return classifications;
}

Mat count_labels( Mat cur_labels ) {
  Mat output(1, 8, CV_32S, Scalar(0));
  for ( int ii = 0; ii < cur_labels.rows; ii++ ) {
    uchar temp = cur_labels.at<uchar>(ii, 0);
    int idx_cur = ((int) temp) - 1;
    if ( idx_cur < 0 || idx_cur > 7 ) {
      throw runtime_error( "( idx_cur < 0 || idx_cur > 7 )" );
    }
    output.at<int>( 0, idx_cur )++;
  }
  cerr << "cur_labels computed : " << output << endl;
  return output;
}

int count_two_depth_vector( const vector<vector<long> > input_vector ) {
  int total = 0;
  for ( int i = 0; i < (int) input_vector.size(); i++ )
    total+=(int) input_vector[i].size();
  return total;
}

void copy_mat_to_10_chunks( Mat combined_descriptors, vector<Mat> &descriptor_train_chunks ) {
  int steps = combined_descriptors.rows / 10;
  cout << "copy_mat_to_10_chunks: steps: " << steps << endl;
  for ( int i = 0; i < steps; i++ ) {
    int start = i*10;
    int end = (i+1)*10-1;
    Rect face( 0, start, combined_descriptors.cols, end-start+1 );
    Mat temp;
    combined_descriptors(face).copyTo( temp );
    descriptor_train_chunks.push_back( temp );
  }
}
