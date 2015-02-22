  float downsample_size = 1;
  if (argc > 1) {
    downsample_size = atof(argv[1]);
  }
  float resize_factor = 1/downsample_size;
	for( int i = 0; i < (int) fpaths.raw_paths_match_mask.size(); i++ ) {
		string raw_filepath = fpaths.raw_paths_match_mask[i];
		string mask_filepath = fpaths.mask_paths[i];
    string mask_of_parts_filepath = fpaths.mask_of_shape_paths[i];
		int img_number_real = fpaths.mask_paths_image_numbers[i];
    
    std::ostringstream stringStream;
    stringStream << setw(6) << setfill('0') << img_number_real;
    string img_number = stringStream.str();
    
		string apple_type_and_img_number = apple_type + string("/") + img_number;
    
		char cur_image_features_and_data_computed_path[300];
		sprintf( cur_image_features_and_data_computed_path, "/home/kyle/undergraduate_thesis/my_code/C++/data/%s/one_descriptor.yml", apple_type_and_img_number.c_str() );
    
		vector<vector<long> > grid_keypoints_processing;
		Mat all_mask_features_train_combined, all_mask_features_train_invalid_combined;
		Mat all_mask_color_features_train_combined, all_mask_color_features_train_invalid_combined;
		Mat all_superpixel_features_combined, all_superpixel_color_features_combined;
		Mat superpixel_img;
		Mat valid_img_8U, valid_img_16U;
    Mat input_img_rgb_uchar_train, input_img_rgb_uchar_train_mask, input_img_gray_uchar_train_mask;
    
/*          setup supervised training set...
		input_img_rgb_uchar_train = imread( raw_filepath );
    resize( input_img_rgb_uchar_train, input_img_rgb_uchar_train, Size(), resize_factor, resize_factor, INTER_NEAREST );
		transpose( input_img_rgb_uchar_train, input_img_rgb_uchar_train );
		flip( input_img_rgb_uchar_train, input_img_rgb_uchar_train, 0 );
		imwrite( "temp_raw.ppm", input_img_rgb_uchar_train );
		vector<orchard_classification> combined_labels;
		input_img_rgb_uchar_train_mask = imread( mask_filepath );
    input_img_gray_uchar_train_mask;
    cvtColor( input_img_rgb_uchar_train_mask, input_img_gray_uchar_train_mask, CV_BGR2GRAY );
    valid_img_8U = compute_valid_pixels( input_img_gray_uchar_train_mask );
    dilate_and_fill_image_ref( valid_img_8U );
    resize( valid_img_8U, valid_img_8U, Size(), resize_factor, resize_factor, INTER_NEAREST );
		if ( i == 0 ) { imwrite( "temp.pgm", input_img_gray_uchar_train_mask ); imwrite( "temp_valid.pgm", valid_img_8U ); }
    valid_img_8U = threshold_uchar_above_250( valid_img_8U );
    valid_img_8U.convertTo( valid_img_16U, CV_16U );
		vector<vector<long> > grid_keypoints_train = compute_grid_points_over_superpixels( valid_img_16U );
    
		vector<KeyPoint> grid_keypoints_train_feature_computation = convert_vector_of_indices_to_keypoints( grid_keypoints_train, input_img_rgb_uchar_train );
		fprintf(stderr, "comp feat valid training\n");
		all_mask_features_train_combined = compute_features_over_image( input_img_rgb_uchar_train, grid_keypoints_train_feature_computation, feature_type );
		all_mask_color_features_train_combined = compute_color_features_over_image( input_img_rgb_uchar_train, grid_keypoints_train_feature_computation );
    Mat invalid_img_16U_eroded = compute_invalid_pixels( valid_img_16U, input_img_rgb_uchar_train );
		Mat invalid_img_8U_eroded;
		invalid_img_16U_eroded.convertTo( invalid_img_8U_eroded, CV_8U );
		vector<vector<long> > grid_keypoints_invalid = compute_grid_points_over_superpixels( invalid_img_8U_eroded );
    
    int draw_keypoints = 0;
    if ( draw_keypoints == 1 ) {
		  Mat output_temp = draw_grid_points_on_image( valid_img_16U, grid_keypoints_train );
		  Mat output_temp_8U;
		  output_temp.convertTo( output_temp_8U, CV_8U );
      if ( output_temp_8U.rows == input_img_rgb_uchar_train.rows ) resize( output_temp_8U, output_temp_8U, Size(), 0.5, 0.5, INTER_NEAREST );
      string output_img_path = apple_type + string("/") + img_number + string("_grid_over_valid_pixels.ppm");
      imwrite( output_img_path, output_temp_8U );
    }

		Mat output_temp_invalid_points_drawn = draw_grid_points_on_image( invalid_img_16U_eroded, grid_keypoints_invalid );
		Mat output_temp_invalid_points_drawn_8U;
		output_temp_invalid_points_drawn.convertTo( output_temp_invalid_points_drawn_8U, CV_8U );
		// prune the invalid keypoints down to the size of the valid features
    // TODO : keep track of the total true and false keypoints before the pruning is completed
		prune_keypoints( grid_keypoints_invalid, (int) grid_keypoints_train_feature_computation.size() );

    
    cerr << "grid_keypoints_invalid.size(): " << grid_keypoints_invalid.size() << endl << "grid_keypoints_train_feature_computation.size(): " << grid_keypoints_train_feature_computation.size() << endl;
    char invalid_drawn_img_path[256];
		vector<KeyPoint> grid_keypoints_invalid_feature_computation = convert_vector_of_indices_to_keypoints( grid_keypoints_invalid, input_img_rgb_uchar_train );
		fprintf(stderr, "comp feat invalid training\n");
		all_mask_features_train_invalid_combined = compute_features_over_image( input_img_rgb_uchar_train, grid_keypoints_invalid_feature_computation, feature_type );
		all_mask_color_features_train_invalid_combined = compute_color_features_over_image( input_img_rgb_uchar_train, grid_keypoints_invalid_feature_computation );

		// compute the features for the labelled apple parts
		Mat input_img_rgb_uchar_train_mask_of_parts = imread( mask_of_parts_filepath );
    string rgb_type = type2str( input_img_rgb_uchar_train_mask_of_parts.type() );
    cerr << "rgb_type: " << rgb_type << endl;
    // mask of parts returns : Very top: 1, top: 2, bottom: 3, very bottom: 4
    Mat mask_of_parts = get_mask_of_parts( input_img_rgb_uchar_train_mask_of_parts );
		resize( mask_of_parts, mask_of_parts, Size(), resize_factor, resize_factor, INTER_NEAREST );
    
		// setup valid image from mask of parts : extract color and texture features in the same way as was done previously
		// get max pixel value in mask_of_parts
		int max_value_in_mask_of_parts = 4;
		for ( int j = 1; j <= max_value_in_mask_of_parts; j++ ) {
		// for i = 1:max_...
			// get image in same format
				// change get valid features?
			Mat valid_img_16U = compute_valid_pixels( input_img_rgb_uchar_train_mask_of_parts );
			// extract features : texture and color
				// hopefully in the same way as the training features
			// extend both the features and the labels by the amount currently computed
		}
*/
/*
		// compute superpixels - check
		fprintf( stderr, "before computing superpixels\n" );
		char superpixel_img_mat_path[512];
		sprintf( superpixel_img_mat_path, "%s/superpixel_img_%s.yml", apple_type.c_str(), img_number.c_str() );
		struct stat buffer2;
    //
		if ( stat ( superpixel_img_mat_path, &buffer2 ) == 0 ) {
			cv::FileStorage storage_superpixels( superpixel_img_mat_path, cv::FileStorage::READ );				
			storage_superpixels["superpixel_img"] >> superpixel_img;
			storage_superpixels.release();
		} else {
    //
			cv::FileStorage storage_superpixels( superpixel_img_mat_path, cv::FileStorage::WRITE );
      // TODO: Change the size of the superpixels to account for the downscaling of the image
			superpixel_img = compute_superpixels( input_img_rgb_uchar_train );
			storage_superpixels << "superpixel_img" << superpixel_img;
			storage_superpixels.release();
		}
		fprintf(stderr, "superpixel image: %d: %d\n", superpixel_img.rows, superpixel_img.cols );
		if (superpixel_img.rows * superpixel_img.cols == 0) throw runtime_error( "no superpixel image? the superpixel image size is 0?" );
    
		grid_keypoints_processing = compute_grid_points_over_superpixels( superpixel_img );
    
		vector<KeyPoint> grid_keypoints_processing_feature_computation = convert_vector_of_indices_to_keypoints( grid_keypoints_processing, input_img_rgb_uchar_train );
		all_superpixel_features_combined = compute_features_over_image( input_img_rgb_uchar_train, grid_keypoints_processing_feature_computation, feature_type );
    
		all_superpixel_color_features_combined = compute_color_features_over_image( input_img_rgb_uchar_train, grid_keypoints_processing_feature_computation );
    fprintf( stderr, "full image features: texture: r: %dc: %d\ncolor: r: %dc: %d\n", all_superpixel_features_combined.rows, all_superpixel_features_combined.cols, all_superpixel_color_features_combined.rows, all_superpixel_color_features_combined.cols );
    fprintf( stderr, "mask valid: texture: r: %d c: %d\ncolor: r: %d c: %d\n", all_mask_features_train_combined.rows, all_mask_features_train_combined.cols, all_mask_color_features_train_combined.rows, all_mask_color_features_train_combined.cols );
    fprintf( stderr, "mask invalid: texture: r: %d c: %d\ncolor: r: %d c: %d\n", all_mask_features_train_invalid_combined.rows, all_mask_features_train_invalid_combined.cols, all_mask_color_features_train_invalid_combined.rows, all_mask_color_features_train_invalid_combined.cols );


    if ( 0 ) {
		  cv::FileStorage storage( cur_image_features_and_data_computed_path, cv::FileStorage::WRITE );
		  storage << "all_mask_features_train_combined" << all_mask_features_train_combined;
		  storage << "all_mask_features_train_invalid_combined" << all_mask_features_train_invalid_combined;
		  storage << "all_mask_color_features_train_combined" << all_mask_color_features_train_combined;
		  storage << "all_mask_color_features_train_invalid_combined" << all_mask_color_features_train_invalid_combined;
		  storage << "all_superpixel_features_combined" << all_superpixel_features_combined;
		  storage << "all_superpixel_color_features_combined" << all_superpixel_color_features_combined;
		  Mat grid_keypoints_processing_superpixels;
		  grid_keypoints_processing_superpixels = convert_grid_locations_to_mat( grid_keypoints_processing );
		  storage << "grid_keypoints_train_combined" << grid_keypoints_processing_superpixels;
		  storage << "superpixel_img" << superpixel_img;
		  storage << "valid_img_8U" << valid_img_8U;
		  storage.release();
    }


    vector<orchard_classification> cur_valid_labels( all_mask_features_train_combined.rows, APPLE );
    vector<orchard_classification> cur_invalid_labels( all_mask_features_train_invalid_combined.rows, LEAF );
		fprintf(stderr, "finished initial feature computation\n");
		all_superpixel_grid_keypoints_processing.push_back( grid_keypoints_processing );
		// feature texture valid
		all_valid_texture_features.push_back( all_mask_features_train_combined );
		// features texture invalid
		all_invalid_texture_features.push_back( all_mask_features_train_invalid_combined );
		// features color valid
		all_valid_color_features.push_back( all_mask_color_features_train_combined );
		// features color invalid
		all_invalid_color_features.push_back( all_mask_color_features_train_invalid_combined );
		// texture features for superpixels
		all_superpixel_features_vec.push_back( all_superpixel_features_combined );
		// color features for superpixels
		all_superpixel_color_features_vec.push_back( all_superpixel_color_features_combined );
		// keypoints for current superpixel
		superpixel_imgs.push_back( superpixel_img );
		valid_imgs.push_back( valid_img_8U );
*/
	}
