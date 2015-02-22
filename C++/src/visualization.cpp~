#include "visualization.h"
#include "utility.h"

Mat create_red_img_from_segmentation_mat( Mat segmentations_cv_16U ) {
	fprintf( stderr, "start of create_red_img_from_segmentation_mat\n" );
	int max_value = compute_max_value( segmentations_cv_16U );
	unsigned short * ptr_to_data = (unsigned short *) segmentations_cv_16U.ptr();
	int number_per_shade = (max_value+1) / 255;
	double shade_multiplier = 255 / (double) (max_value+1);
	Mat segmentation_in_red = Mat::zeros( segmentations_cv_16U.rows, segmentations_cv_16U.cols, CV_8U );
	for ( int i = 0; i < segmentations_cv_16U.rows * segmentations_cv_16U.cols; i++ ) {
		unsigned short cur = ptr_to_data[i];
		int r = i%segmentations_cv_16U.rows;
		int c = i/segmentations_cv_16U.rows;
		segmentation_in_red.at<uchar>(r,c) = ( uchar ) (((double) cur ) * shade_multiplier);
	}
	fprintf( stderr, "end of create_red_img_from_segmentation_mat\n" );
	return segmentation_in_red;
}

Mat draw_grid_points_on_image( Mat valid_img_16U, vector<vector<long> > grid_keypoints_train ) {
	fprintf(stderr, "aaaa\n");
	Mat grid_keypoints_8U = Mat::zeros( valid_img_16U.rows, valid_img_16U.cols, CV_8U );
	for ( int i = 0; i < (int) grid_keypoints_train.size(); i++) {
		vector<long> cur_indices = grid_keypoints_train[i];
		for ( int j = 0; j < (int) cur_indices.size(); j++ ) {
			int cur_idx = (int) cur_indices[j];
			int r = cur_idx % valid_img_16U.rows;
			int c = cur_idx / valid_img_16U.rows;
			grid_keypoints_8U.at<uchar>(r, c) = 1;
		}
	}
	fprintf(stderr, "aaab\n");
	int dilation_size = 1;
	int dilation_type = MORPH_RECT;
	Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
	int total_keypoints = 0;
	for ( int i = 0; i < grid_keypoints_8U.rows; i++ ) {
		for ( int j = 0; j < grid_keypoints_8U.cols; j++ ) {
			int cur_value = grid_keypoints_8U.at<uchar>(i, j) > 0 ? 1 : 0;
			total_keypoints += cur_value;
		}
	}
	Mat grid_keypoints_dilated_8U;
  dilate( grid_keypoints_8U, grid_keypoints_dilated_8U, element );
	fprintf(stderr, "aaac\n");
	Mat valid_points_drawn = Mat::zeros( valid_img_16U.rows, valid_img_16U.cols, CV_8UC3 );
	for ( int i = 0; i < valid_points_drawn.rows; i++ ) {
		for ( int j = 0; j < valid_points_drawn.cols; j++ ) {
			if ( grid_keypoints_dilated_8U.at<uchar>(i, j) == 1 ) {
				valid_points_drawn.at<cv::Vec3b>(i, j)[0] = 0;
				valid_points_drawn.at<cv::Vec3b>(i, j)[1] = 255;
				valid_points_drawn.at<cv::Vec3b>(i, j)[2] = 0;
			} else if ( valid_img_16U.at<ushort>(i, j) > 0 ) {
				valid_points_drawn.at<cv::Vec3b>(i, j)[0] = 255;
				valid_points_drawn.at<cv::Vec3b>(i, j)[1] = 255;
				valid_points_drawn.at<cv::Vec3b>(i, j)[2] = 255;
			}
		}
	}
	fprintf(stderr, "aaad\n");
	return valid_points_drawn;
}

Mat display_superpixel_classifications( vector<orchard_classification> cur_superpixel_classifications, Mat input_img_rgb_uchar, Mat superpixel_img, vector<double> superpixel_dominant_angle, string img_name ) {
	Mat valid_points_drawn = Mat::zeros( input_img_rgb_uchar.rows, input_img_rgb_uchar.cols, CV_8UC3 );
	Mat valid_points_drawn_gradient = Mat::zeros( input_img_rgb_uchar.rows, input_img_rgb_uchar.cols, CV_8UC3 );
	for ( int i = 0; i < valid_points_drawn.rows; i++ ) {
		for ( int j = 0; j < valid_points_drawn.cols; j++ ) {
			int cur_superpixel_idx = superpixel_img.at<ushort>( i, j );
			int r = 0, g=0, b=0;
      Vec3b red(255, 0, 0); // right
      Vec3b green(0, 255, 0); // left
      Vec3b blue(0, 0, 255); // down
      Vec3b white(255, 255, 255); // up
			if (cur_superpixel_classifications[cur_superpixel_idx] == APPLE) {
				valid_points_drawn.at<cv::Vec3b>(i, j)[0] = input_img_rgb_uchar.at<cv::Vec3b>(i, j)[0];
				valid_points_drawn.at<cv::Vec3b>(i, j)[1] = input_img_rgb_uchar.at<cv::Vec3b>(i, j)[1];
				valid_points_drawn.at<cv::Vec3b>(i, j)[2] = input_img_rgb_uchar.at<cv::Vec3b>(i, j)[2];
        double ca = superpixel_dominant_angle[cur_superpixel_idx];
        if ( ca < (PI/4) && ca > (-1*PI/4) ) { // right facing angle: red 
          valid_points_drawn_gradient.at<cv::Vec3b>(i, j) = red;
        } else if ( ca < (-1*PI/4) && ca > (-3*PI/4) ) { // left facing angle: green 
          valid_points_drawn_gradient.at<cv::Vec3b>(i, j) = green;
        } else if (ca  < (-3*PI/4) || ca > (3*PI/4)) { // downward facing angle: blue
          valid_points_drawn_gradient.at<cv::Vec3b>(i, j) = blue;
        } else if ( ca > (1*PI/4) && ca < (3*PI/4) ) { // upwards facing angle: white
          valid_points_drawn_gradient.at<cv::Vec3b>(i, j) = white;
        }
			} else {
				valid_points_drawn.at<cv::Vec3b>(i, j)[0] = r;
				valid_points_drawn.at<cv::Vec3b>(i, j)[1] = g;
				valid_points_drawn.at<cv::Vec3b>(i, j)[2] = b;
			}
		}
	}
  imwrite( img_name, valid_points_drawn_gradient );
	return valid_points_drawn;
}

Mat draw_classifications( const Mat superpixel_image, Mat classifications ) {
  cv::Vec3b array_colors[8] = {
    cv::Vec3b(255, 0, 0),
    cv::Vec3b(0, 255, 0),
    cv::Vec3b(0, 0, 255),
    cv::Vec3b(255, 255, 0),
    cv::Vec3b(0, 255, 255),
    cv::Vec3b(255, 0, 255),
    cv::Vec3b(255, 255, 255),
    cv::Vec3b(150, 150, 150)
  };
  cout << "The superpixel image type is: " << type2str( superpixel_image.type() ) << endl;
  int i, j;
  Mat ret = Mat::zeros( superpixel_image.rows, superpixel_image.cols, CV_8UC3 );

  for ( i = 0; i < superpixel_image.rows; i++ )
  for ( j = 0; j < superpixel_image.cols; j++ )
  { 
    ushort cur_superpixel_idx = superpixel_image.at<ushort>(i, j);
    uchar cur_classification = classifications.at<uchar>( (int) cur_superpixel_idx, 0 );
    if ( cur_classification > 0 ) {
      ret.at<cv::Vec3b>( i, j ) = array_colors[ (int) cur_classification - 1 ];
    }
  }
  return ret;
}

Mat draw_apple_pixels( const Mat superpixel_image, const Mat raw_image, Mat classifications ) {
  if ( superpixel_image.rows != raw_image.rows ) throw runtime_error( "image resizing error: kyle 51" );
  cv::Vec3b array_colors[8] = {
    cv::Vec3b(255, 0, 0),
    cv::Vec3b(0, 255, 0),
    cv::Vec3b(0, 0, 255),
    cv::Vec3b(255, 255, 0),
    cv::Vec3b(0, 255, 255),
    cv::Vec3b(255, 0, 255),
    cv::Vec3b(255, 255, 255),
    cv::Vec3b(150, 150, 150)
  };
  cout << "The superpixel image type is: " << type2str( superpixel_image.type() ) << endl;
  int i, j;
  Mat ret = Mat::zeros( superpixel_image.rows, superpixel_image.cols, CV_8UC3 );

  for ( i = 0; i < superpixel_image.rows; i++ )
  for ( j = 0; j < superpixel_image.cols; j++ )
  {
    ushort cur_superpixel_idx = superpixel_image.at<ushort>(i, j) - 1;
    uchar cur_classification = classifications.at<uchar>( (int) cur_superpixel_idx, 0 );
    if ( cur_classification > 0 ) {
      int cur_idx = (int) cur_classification - 1;
      if ( cur_idx == 0 ) ret.at<cv::Vec3b>( i, j ) = raw_image.at<cv::Vec3b>( i, j );
      else ret.at<cv::Vec3b>( i, j ) = array_colors[ cur_idx ];
    }
  }
  return ret;
}

// TODO ex: finish this function
Mat draw_classification_probability( Mat superpixel_image, Mat percentages_by_type, int idx ) {
  // go through each pixel
  // simply write (255 * probability) that corresponds to that superpixel
  Mat classifications_probability_8U = Mat( superpixel_image.rows, superpixel_image.cols, CV_8U, Scalar(0) );
  for ( int i = 0; i < superpixel_image.rows; i++ ) {
    for ( int j = 0; j < superpixel_image.cols; j++ ) {
      int superpixel_idx = ((int)superpixel_image.at<ushort>(i, j))-1;
      float cur_prob = percentages_by_type.at<float>( superpixel_idx, idx );
      if ( cur_prob > 1 ) throw runtime_error( "Percentages_by_type[ x ] greater than 1?" );
      classifications_probability_8U.at<uchar>( i, j ) = (uchar) ( 255.0 * cur_prob );
    }
  }
  return classifications_probability_8U;
}

// TODO ex: finish this function
Mat draw_only_apple_classifications( Mat superpixel_image, Mat classifications ) {
  // go through each pixel in superpixel image
  // if classification is 1 : apple : set classification image = 255
  Mat  classifications_apple = Mat( superpixel_image.rows, superpixel_image.cols, CV_8U, Scalar(0) );
  for ( int i = 0; i < superpixel_image.rows; i++ ) {
    for ( int j = 0; j < superpixel_image.cols; j++ ) {
      int superpixel_idx = ((int) superpixel_image.at<ushort>(i, j) ) - 1;
      int classification = classifications.at<uchar>(superpixel_idx, 0);
      int classification_idx = classification - 1;
      if ( classification_idx == 0 ) {
        classifications_apple.at<uchar>( i, j ) = 255;
      }
    }
  }
  return classifications_apple;
}
