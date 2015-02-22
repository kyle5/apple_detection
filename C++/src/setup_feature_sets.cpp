#include "utility.h"
#include "include_in_all.h"
#include "setup_correct_apple_paths.h"
#include "setup_feature_sets.h"
#include "superpixel_computation.h"
#include "feature_computation.h"

// inputs: supervised feature set and an image number
// outputs: classifier_features object
struct classifier_features setup_classifier_features( struct supervised_feature_set feature_set_input, int image_loo ) {
  struct classifier_features ret;
  Mat full_texture_features, full_color_features, full_labels;

  for ( int i = 0; i < (int) feature_set_input.complete_labels.size(); i++ ) {
    bool no_mask_image = feature_set_input.complete_texture_features[i].rows == 0 || feature_set_input.complete_texture_features[i].cols == 0;
    bool cur_image_processing = i == image_loo;
    if( no_mask_image || cur_image_processing ) continue;
    combine_features_vertically_ref( full_texture_features, feature_set_input.complete_texture_features[i] );
    combine_features_vertically_ref( full_color_features, feature_set_input.complete_color_features[i] );
    combine_features_vertically_ref( full_labels, feature_set_input.complete_labels[i] );
  }
  
  if ( full_texture_features.rows == 0 || full_texture_features.cols == 0) throw runtime_error("There are no images to create a classifier?!?!");
  Mat empty, pca_descriptors_train, combined_c_and_t_features_64F;
  full_texture_features.convertTo( full_texture_features, CV_64F );
  full_color_features.convertTo( full_color_features, CV_64F );
  
  ret.norm_factors_texture = compute_mean_and_std( full_texture_features );
  ret.norm_factors_color = compute_mean_and_std( full_color_features );
  
  Mat normalized_texture_descriptors_train = normalize_mat( full_texture_features, ret.norm_factors_texture );
  Mat normalized_color_descriptors_train = normalize_mat( full_color_features, ret.norm_factors_color );
  
  // apply PCA trans to texture feat
  PCA pca( normalized_texture_descriptors_train, empty, CV_PCA_DATA_AS_ROW, 6 );
  pca.project( normalized_texture_descriptors_train, pca_descriptors_train );
  ret.pca_transform = pca;
  cerr << "type of pca_descriptors_train:   " << type2str( pca_descriptors_train.type() ) << endl;
  pca_descriptors_train.convertTo( combined_c_and_t_features_64F, CV_64F );
  // Everything should be converted to 64F. This was the way that it was done before .
  Mat normalized_color_descriptors_train_64F;
  normalized_color_descriptors_train.convertTo( normalized_color_descriptors_train_64F, CV_64F );
  combine_features_horizontally_ref( combined_c_and_t_features_64F, normalized_color_descriptors_train_64F );
  // return structure
  ret.feature_space = combined_c_and_t_features_64F;
  ret.combined_labels = full_labels;

  return ret;
}

Mat transform_indices_to_mat( vector<vector<long> > indices ) {
  int i, j, total_len = 0;
  for (i = 0; i < indices.size(); i++) total_len += (int) indices[i].size();
  Mat ret = Mat::zeros( total_len, 2, CV_32S );
  int cur_idx = -1;
  for (i = 0; i < indices.size(); i++) {
    for (j = 0; j < indices[i].size(); j++) {
      cur_idx++;
      if (j == 0) ret.at<int>(cur_idx, 0) = i;
      ret.at<int>(cur_idx, 1) = indices[i][j];
    }
  }
  return ret;
}

Mat transform_keypoints_to_mat( vector<KeyPoint> keypoints, int num_superpixels ) {
  int i, j, total_len = (int) keypoints.size(), cur_len = 0;
  Mat ret = Mat::zeros( total_len, 2, CV_32S );
  int cur_idx = 0;
  for (i = 0; i < keypoints.size(); i++) {
    ret.at<int>(cur_idx, 0) = keypoints[i].pt.x;
    ret.at<int>(cur_idx, 1) = keypoints[i].pt.y;
  }
  return ret;
}

vector<vector<long> > transform_mat_to_indices( Mat indices_mat, int num_superpixels ) {
  vector<vector<long> > ret(num_superpixels, vector<long>() );
  int i, j, idx = -1;
  for (i = 0; i < indices_mat.rows; i++) {
    int idx_check = indices_mat.at<int>(i, 0);
    if (idx_check > 0) idx = idx_check;
    if ( idx < 0 ) throw runtime_error( "index is less than zero?!" );
    ret[idx].push_back( indices_mat.at<int>(i, 1) );
  }
  return ret;
}

vector<KeyPoint> transform_mat_to_keypoints( Mat indices_mat, int num_superpixels ) {
  vector<KeyPoint> ret;
  int i;
  for (i = 0; i < indices_mat.rows; i++) {
    int x = indices_mat.at<int>(i, 0);
    int y = indices_mat.at<int>(i, 1);
    KeyPoint kp( x, y, 10 );
    ret.push_back( kp );
  }
  return ret;
}

int save_features_full_images( string path_save_dir, struct features_full_images features_input, struct filepaths_apple_project fpaths ) {
  string system_command = string("mkdir " ) + path_save_dir;
  system( system_command.c_str() );
  char temp[256];
  sprintf( temp, "%s/processing_general_info.txt", path_save_dir.c_str() );
  FILE *fid = fopen( temp, "w");
  fprintf( fid, "%d", (int) features_input.all_texture_features.size() );
  fclose( fid );
  int i;
  for ( i = 0; i < (int) features_input.all_texture_features.size(); i++ ) {
    Mat indices_mat = transform_indices_to_mat( features_input.indices[i] );
    Mat keypoints_mat = transform_keypoints_to_mat( features_input.keypoints[i], -1 );
    ostringstream temp;
    temp << path_save_dir << string("/features_full_images_") << fpaths.all_types_image_numbers[i] << string(".yml");
//    string features_full_images_path = path_save_dir + string("/features_full_images_") + string(i) + string(".yml");
    string features_full_images_path = temp.str();
    cout << "features_full_images_path: " << features_full_images_path << endl;
		FileStorage storage_features_full_images( features_full_images_path, cv::FileStorage::WRITE );
    storage_features_full_images << "texture_features" << features_input.all_texture_features[i];
    storage_features_full_images << "color_features" << features_input.all_color_features[i];
    storage_features_full_images << "superpixel_image" << features_input.superpixel_imgs[i];
    storage_features_full_images << "valid_image" << features_input.valid_imgs[i];
    storage_features_full_images << "indices_mat" << indices_mat;
    storage_features_full_images << "keypoints_mat" << keypoints_mat;
    storage_features_full_images.release();
  }
  return 0;
}
#include <sys/stat.h>

struct features_full_images load_features_full_images( string path_save_dir, struct filepaths_apple_project fpaths ) {
  char temp[256];
  sprintf( temp, "%s/processing_general_info.txt", path_save_dir.c_str() );
  FILE *fid = fopen(temp, "r");
  int num_superpixels;
  fscanf( fid, "%d", &num_superpixels );
  fclose( fid );
  
  struct features_full_images features_output;
  int i;
  for ( i = 0; i < num_superpixels; i++ ) {
    ostringstream temp;
    temp << path_save_dir << string("/features_full_images_") << fpaths.all_types_image_numbers[i] << string(".yml");
    string features_full_images_path =  temp.str();
    
    // check if the file exists first
    struct stat buffer;
    if ( !( stat( features_full_images_path.c_str(), &buffer ) == 0 ) ) {
      throw runtime_error( "features have changed?" );
    }
    
		FileStorage storage_features_full_images( features_full_images_path, cv::FileStorage::READ );
    Mat texture_features, color_features, superpixel_image, valid_image, indices_mat, keypoints_mat;
    storage_features_full_images["texture_features"] >> texture_features;
    storage_features_full_images["color_features"] >> color_features;
    storage_features_full_images["superpixel_image"] >> superpixel_image;
    storage_features_full_images["valid_image"] >> valid_image;
    storage_features_full_images["indices_mat"] >> indices_mat;
    storage_features_full_images["keypoints_mat"] >> keypoints_mat;
    storage_features_full_images.release();
    
    int max_value = compute_max_value( superpixel_image );
    vector< vector< long > > indices = transform_mat_to_indices( indices_mat, max_value );
    vector< KeyPoint > keypoints = transform_mat_to_keypoints( keypoints_mat, max_value );
    
    features_output.all_texture_features.push_back( texture_features );
    features_output.all_color_features.push_back( color_features );
    features_output.superpixel_imgs.push_back( superpixel_image );
    features_output.keypoints.push_back( keypoints );
    features_output.indices.push_back( indices );
    features_output.valid_imgs.push_back( valid_image );
  }
  return features_output;
}

Mat up_over_1_to_255( Mat img_classes_8U ) {
  Mat img_classes_8U_disp = img_classes_8U.clone();
  for ( int j = 0; j < img_classes_8U_disp.rows*img_classes_8U_disp.cols; j++ ) {
    uchar cur_val = img_classes_8U_disp.data[j];
    if ( cur_val > 0) img_classes_8U_disp.data[j] = 255;
  }
  return img_classes_8U_disp;
}

// TODO EX: function to get supervised feature sets
struct supervised_feature_set setup_supervised_feature_set( vector<string> raw_img_paths, vector<string> mask_img_paths, int compute_invalid_features, struct filepaths_apple_project fpaths, string root_saving_string, float resize_factor = 1 ) {
  // TODO : Need some way to distinguish between mask images with 0|1 only and images with 1-5, etc... 
  int num_images = (int) raw_img_paths.size();
  struct supervised_feature_set ret(num_images);
  // TODO : for loop over image numbers
  for ( int i = 0; i < (int) raw_img_paths.size(); i++ ) {
    if ( mask_img_paths[i].length() == 0 ) { // there is no mask image that matches the raw image at this index...
      continue;
    }
    ostringstream temp;
    string path_start("green");
    // TODO ex : save the features according to the image number labels instead of the indices
    temp << path_start << string( "/" ) << root_saving_string << string("_") << fpaths.all_types_image_numbers[i] << string( ".yml" );
    string features_path = temp.str();
    Mat cur_color_features, cur_texture_features, cur_labels;
    int recompute = 0;
    struct stat buffer;
    if ( stat ( features_path.c_str(), &buffer ) != 0 || recompute ) {
      // load image
      // get features
      // TODO : setup raw path from raw path format and image number
      string cur_raw_path = raw_img_paths[i];
      Mat img_classes_8U, input_img_rgb_uchar_train, input_img_rgb_uchar_train_mask, input_img_gray_uchar_train_mask, valid_img_8U, img_classes_16U, img_classes_8U_disp;
      input_img_rgb_uchar_train = read_image_and_flip_90( cur_raw_path, resize_factor );
      Mat temp_mat = input_img_rgb_uchar_train.clone();
      resize( input_img_rgb_uchar_train, temp_mat, Size(), 0.5, 0.5 );
      imwrite( "train_img_raw_sample.ppm", temp_mat );
		  // TODO : setup mask path from mask path format and image number
      string mask_filepath = mask_img_paths[i];
      // read mask image
      input_img_rgb_uchar_train_mask = imread( mask_filepath );
      // compute pixels that are valid
      // TODO : method below for both valid images and superpixel images
      // method to setup : segmentation
      if (compute_invalid_features) {
        img_classes_8U = compute_valid_pixels( input_img_rgb_uchar_train_mask );
        dilate_and_fill_image_ref( img_classes_8U );
      } else {
        img_classes_8U = get_mask_of_parts( input_img_rgb_uchar_train_mask );
        if ( i == 0 ) {
          Mat img_classes_8U_temp;
          resize( img_classes_8U, img_classes_8U_temp, Size(), resize_factor, resize_factor, INTER_NEAREST );
		      Mat img_classes_8U_disp = up_over_1_to_255( img_classes_8U_temp );
          imwrite( "mask_of_parts_valid.pgm", img_classes_8U_disp );
        }
      }
      int type_cur = img_classes_8U.type();
      uchar depth_cur = type_cur & CV_MAT_DEPTH_MASK;
      if ( depth_cur != CV_8U ) throw runtime_error( "mask of parts function is likely wrong?!" );

      double resize_factor = 1;
      resize( img_classes_8U, img_classes_8U, Size(), resize_factor, resize_factor, INTER_NEAREST );
		  if ( i == 0 ) {
        Mat img_classes_8U_disp = up_over_1_to_255( img_classes_8U );
        imwrite( "temp_valid.pgm", img_classes_8U_disp );
      }
      img_classes_8U.convertTo( img_classes_16U, CV_16U );
      int max_value = compute_max_value( img_classes_16U );
      cout << "max_value segmentation binary image: " << max_value << endl;
		  vector<vector<long> > grid_indices = compute_grid_points_over_superpixels( img_classes_16U, 0, max_value );
		  vector<KeyPoint> grid_keypoints = convert_vector_of_indices_to_keypoints( grid_indices, input_img_rgb_uchar_train );
      string feature_type = "SURF";
		  cur_texture_features = compute_features_over_image( input_img_rgb_uchar_train, grid_keypoints, feature_type );
		  cur_color_features = compute_color_features_over_image( input_img_rgb_uchar_train, grid_keypoints );
      if ( (int) grid_keypoints.size() != cur_color_features.rows || (int) grid_keypoints.size() != cur_texture_features.rows ) throw runtime_error( "Texture or Color features do not match keypoints length! Need to somehow update for the keypoints that could not be computed..." );
      cur_labels = Mat( (int) grid_keypoints.size(), 1, CV_8U, Scalar(1) );
      for ( int j = 0; j < (int) grid_keypoints.size(); j++ ) {
        cur_labels.at<uchar>(j, 0) = img_classes_8U.at<uchar>( grid_keypoints[j].pt.y, grid_keypoints[j].pt.x );
      }
      if ( compute_invalid_features ) {
        Mat invalid_img_classes_16U_eroded = compute_invalid_pixels( img_classes_16U, input_img_rgb_uchar_train );
		    Mat invalid_img_classes_8U_eroded;
		    invalid_img_classes_16U_eroded.convertTo( invalid_img_classes_8U_eroded, CV_8U );
		    vector<vector<long> > grid_indices_invalid = compute_grid_points_over_superpixels( invalid_img_classes_16U_eroded, 0, 1 );
        cerr << "pruning: start : " << (int) count_two_depth_vector( grid_indices_invalid ) << endl;
        cerr << "pruning: valid : start : " << (int) count_two_depth_vector( grid_indices ) << endl;
        cerr << "prune to : " << (int) grid_keypoints.size() << endl;
		    prune_keypoints( grid_indices_invalid, (int) grid_keypoints.size() );
        cerr << "pruning: end : " << (int) count_two_depth_vector( grid_indices_invalid ) << endl;
		    vector<KeyPoint> grid_keypoints_invalid = convert_vector_of_indices_to_keypoints( grid_indices_invalid, input_img_rgb_uchar_train );
		    fprintf(stderr, "comp feat invalid training\n");
		    Mat features_train_invalid = compute_features_over_image( input_img_rgb_uchar_train, grid_keypoints_invalid, feature_type );
		    Mat color_features_train_invalid = compute_color_features_over_image( input_img_rgb_uchar_train, grid_keypoints_invalid );
        combine_features_vertically_ref(cur_texture_features, features_train_invalid);
        combine_features_vertically_ref(cur_color_features, color_features_train_invalid);
        combine_features_vertically_ref(cur_labels, Mat(features_train_invalid.rows, 1, CV_8U, Scalar(2) ));
      }
      // save the features
		  FileStorage storage_features_full_images( features_path, cv::FileStorage::WRITE );
      storage_features_full_images << "cur_color_features" << cur_color_features;
      storage_features_full_images << "cur_texture_features" << cur_texture_features;
      storage_features_full_images << "cur_labels" << cur_labels;
      storage_features_full_images.release();
    } else { // load the features
      FileStorage storage_features_full_images( features_path, cv::FileStorage::READ );
      storage_features_full_images["cur_color_features"] >> cur_color_features;
      storage_features_full_images["cur_texture_features"] >> cur_texture_features;
      storage_features_full_images["cur_labels"] >> cur_labels;
      storage_features_full_images.release();
    }
    ret.complete_color_features[i] = cur_color_features;
    ret.complete_texture_features[i] = cur_texture_features;
    ret.complete_labels[i] = cur_labels;
  }
  return ret;
}

struct features_computed compute_features_single_image( Mat input_img_rgb_uchar_train, string feature_type ) {
  struct features_computed ret;
  Mat superpixel_img = compute_superpixels( input_img_rgb_uchar_train );
  int num_superpixels;
  increment_superpixel_img_ref( superpixel_img, 1, num_superpixels );
	vector<vector<long> > grid_keypoints_processing = compute_grid_points_over_superpixels( superpixel_img, 1, num_superpixels );
  // compute the valid intensity pixels
  Mat superpixels_value_is_valid_8U = get_superpixels_with_high_value( input_img_rgb_uchar_train, grid_keypoints_processing, superpixel_img );
  filter_indices_by_valid_superpixels( input_img_rgb_uchar_train, grid_keypoints_processing, superpixels_value_is_valid_8U );

  if ( num_superpixels != superpixels_value_is_valid_8U.rows ) throw runtime_error( "num superpixels does not match valid count!" ); 
  else cout << "num_superpixels == superpixels_value_is_valid_8U.rows !" << endl;
	vector<KeyPoint> grid_keypoints_processing_feature_computation = convert_vector_of_indices_to_keypoints( grid_keypoints_processing, input_img_rgb_uchar_train );
	Mat all_superpixel_features_combined = compute_features_over_image( input_img_rgb_uchar_train, grid_keypoints_processing_feature_computation, feature_type );
	Mat all_superpixel_color_features_combined = compute_color_features_over_image( input_img_rgb_uchar_train, grid_keypoints_processing_feature_computation );
  ret.grid_indices = grid_keypoints_processing;
  ret.grid_keypoints = grid_keypoints_processing_feature_computation;
  ret.superpixel_image = superpixel_img;
  ret.texture_features = all_superpixel_features_combined;
  ret.color_features = all_superpixel_color_features_combined;
  ret.superpixels_value_is_valid_8U = superpixels_value_is_valid_8U;
  // save the valid intensity superpixels
  return ret;
}

// TODO EX: function to get feature sets over full images
// TODO : signal when results have already been computed
struct features_full_images setup_features_full_images( vector<string> raw_img_paths, float resize_factor, string feature_type, struct filepaths_apple_project fpaths ) {
  struct features_full_images ret( (int) raw_img_paths.size() );
  for ( int i = 0; i < (int) raw_img_paths.size(); i++ ) {
    ostringstream temp;
    string path_save_dir = string( "green/" );
    temp << path_save_dir << string("features_full_images_") << fpaths.all_types_image_numbers[i] << string(".yml");
    string features_full_images_path = temp.str();
    int recompute = 1;
    struct stat   buffer;
    if ( recompute || stat ( features_full_images_path.c_str(), &buffer ) != 0 ) {
      cout << "filename : ` " << features_full_images_path << " ` does not exist :{" << endl;
      string cur_raw_img_path = raw_img_paths[i];
      Mat input_img_rgb_uchar_train;
      input_img_rgb_uchar_train = read_image_and_flip_90( cur_raw_img_path, resize_factor );
      struct features_computed features_cur = compute_features_single_image( input_img_rgb_uchar_train, feature_type );
      Mat indices_mat = transform_indices_to_mat( features_cur.grid_indices );
      Mat keypoints_mat = transform_keypoints_to_mat( features_cur.grid_keypoints, -1 );
      cout << "features_full_images_path: " << features_full_images_path << endl;
		  FileStorage storage_features_full_images( features_full_images_path, cv::FileStorage::WRITE );

      storage_features_full_images << "superpixels_value_is_valid_8U" << features_cur.superpixels_value_is_valid_8U;
      storage_features_full_images << "texture_features" << features_cur.texture_features;
      storage_features_full_images << "color_features" << features_cur.color_features;
      storage_features_full_images << "superpixel_image" << features_cur.superpixel_image;
      storage_features_full_images << "indices_mat" << indices_mat;
      storage_features_full_images << "keypoints_mat" << keypoints_mat;
      storage_features_full_images.release();

      ret.superpixel_imgs[i] = features_cur.superpixel_image;
      ret.valid_superpixels[i] = features_cur.superpixels_value_is_valid_8U;
      ret.all_texture_features[i] = features_cur.texture_features;
      ret.all_color_features[i] = features_cur.color_features;
      ret.keypoints[i] = features_cur.grid_keypoints;
      ret.indices[i] = features_cur.grid_indices;
      cerr << "temp " << endl;
    } else {
      cout << "filename : ` " << features_full_images_path << " ` does exist!" << endl;
		  FileStorage storage_features_full_images( features_full_images_path, cv::FileStorage::READ );
      Mat texture_features, color_features, superpixel_image, indices_mat, keypoints_mat, superpixels_value_is_valid_8U;
      storage_features_full_images["texture_features"] >> texture_features;
      storage_features_full_images["color_features"] >> color_features;
      storage_features_full_images["superpixel_image"] >> superpixel_image;
      storage_features_full_images["indices_mat"] >> indices_mat;
      storage_features_full_images["keypoints_mat"] >> keypoints_mat;
      storage_features_full_images["superpixels_value_is_valid_8U"] >> superpixels_value_is_valid_8U;
      storage_features_full_images.release();

      int max_value = compute_max_value( superpixel_image );
      vector<vector<long> > grid_indices = transform_mat_to_indices( indices_mat, max_value );
      vector< KeyPoint > grid_keypoints = transform_mat_to_keypoints( keypoints_mat, max_value );

      if ( (int) grid_indices.size() != max_value )
        throw runtime_error( "classifier does not correlate superpixel image to the feature indices" );
      else 
        cerr << "features seem to be alright" << endl;
      ret.superpixel_imgs[i] = superpixel_image;
      ret.all_texture_features[i] = texture_features;
      ret.all_color_features[i] = color_features;
      ret.keypoints[i] = grid_keypoints;
      ret.indices[i] = grid_indices;
      ret.valid_superpixels[i] = superpixels_value_is_valid_8U;
      cout << "temp " << endl;
    }
  }
  // TODO EX: go through each raw image:
  //   // TODO EX:get the superpixels, indices of grid points, texture features, color features, ....?
  return ret;
}
