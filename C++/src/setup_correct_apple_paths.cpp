#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include "setup_correct_apple_paths.h"
#include "utility.h"
#include <utility>

using namespace std;
using namespace cv;

string get_filename( string filename ) {
  int idx0 = filename.rfind("/");
  int idx1 = filename.rfind(".");
  string path = filename.substr(0,idx0+1);
  string ext  = filename.substr(idx1);
  return filename.substr(idx0+1,idx1-idx0-1);
}

vector<int> get_image_numbers( vector<string> image_paths, string root, string ext ) {
  vector<int> image_numbers( (int) image_paths.size(), -1 );
  for ( int i = 0; i < (int) image_paths.size(); i++ ) {
    string cur_mask_path = image_paths[i];
    // fileparts to get the filename by itself
    string cur_filename = get_filename( cur_mask_path );
    cerr << "cur_filename: " << cur_filename << " " << endl;
    // take off "image_name_root"
    // convert the rest to a real number
    cur_filename.erase( 0, root.length() );
    cur_filename.erase( cur_filename.end()-ext.length(), cur_filename.end() );
    int num = atoi( cur_filename.c_str() );
    image_numbers[i] = num;
    cerr << "one string: " << cur_filename << " cur number: " << num << endl;
  }
  return image_numbers;
}

vector<string> setup_filepaths( string filepath_root, string filepath_root_fname, string filepath_ext, vector<int> corresponding_numbers ) {
  vector<string> filepaths_made;
  for ( int i = 0; i < (int) corresponding_numbers.size(); i++ ) {
    ostringstream os_temp;
    os_temp << setw(4) << setfill('0') << corresponding_numbers[i];
    string filepath_cur = filepath_root + filepath_root_fname + os_temp.str() + filepath_ext;
    filepaths_made.push_back( filepath_cur );
  }
  return filepaths_made;
}

vector<string> create_paths( string path_str, vector<string> fnames ) {
  string temp("");
  vector<string> paths( (int) fnames.size(), "" );
  for ( int i = 0; i < (int) fnames.size(); i++ ) {
    string cur_fname = fnames[i];
    string path_cur = path_str + cur_fname;
    paths[i] = path_cur;
  }
  return paths;
}

void remove_extraneous_paths( struct filepaths_apple_project &fpaths ) {

  for ( int i = 0; i < (int) fpaths.all_types_image_numbers.size(); i++ ) {
    int raw_number = fpaths.all_types_image_numbers[i];
    int raw_image_needed = 0;
    for ( int j = 0; j < (int) fpaths.mask_image_numbers.size(); j++ ) {
      if ( fpaths.all_types_image_numbers[i] == fpaths.mask_image_numbers[j] ) {
        raw_image_needed = 1;
      }
    }
    for ( int j = 0; j < (int) fpaths.mos_image_numbers.size(); j++ ) {
      if ( fpaths.all_types_image_numbers[i] == fpaths.mos_image_numbers[j] ) {
        raw_image_needed = 1;
      }
    }
    if ( !raw_image_needed ) {
      fpaths.raw_training_paths[i] = string("");
    }
  }
  vector<string> raw_training_paths_new, mask_paths_new, mos_paths_new;
  vector<int> all_types_image_numbers_new, mos_image_numbers_new, mask_image_numbers_new;
  for ( int i = 0; i < (int) fpaths.raw_training_paths.size(); i++ ) {
    if ( fpaths.raw_training_paths[i].length() == 0 ) {
      continue;
    }
    raw_training_paths_new.push_back(fpaths.raw_training_paths[i]);
    mask_paths_new.push_back(fpaths.mask_paths[i]);
    mos_paths_new.push_back(fpaths.mos_paths[i]);
    all_types_image_numbers_new.push_back(fpaths.all_types_image_numbers[i]);
    mask_image_numbers_new.push_back(fpaths.mask_image_numbers[i]);
    mos_image_numbers_new.push_back(fpaths.mos_image_numbers[i]);
  }
  fpaths.raw_training_paths = raw_training_paths_new;
  fpaths.mask_paths = mask_paths_new;
  fpaths.mos_paths = mos_paths_new;
  fpaths.all_types_image_numbers = all_types_image_numbers_new;
  fpaths.mask_image_numbers = mask_image_numbers_new;
  fpaths.mos_image_numbers = mos_image_numbers_new;
}

vector<string> correlate_image_paths( vector<int> match_numbers, vector<int> to_be_matched_numbers, vector<string> to_be_matched_paths ) {
  vector<string> correlated_paths( (int) match_numbers.size(), string("") );
  for ( int i = 0; i < (int) match_numbers.size(); i++ ) {
    string matched_string("");
    for ( int j = 0; j < (int) to_be_matched_numbers.size(); j++ ) {
      if ( match_numbers[i] == to_be_matched_numbers[j] ) {
        matched_string = to_be_matched_paths[j];
        break;
      }
    }
    correlated_paths[i] = matched_string;
  }
  // to be matched should be the same length as the path names
  return correlated_paths;
}

vector< std::pair<int, int> > sort_vector_with_indices( vector<int> vector_to_sort ) {
  vector<pair<int, int> > vect;
  for ( size_t i = 0; i < vector_to_sort.size(); i++ ) {
    pair<int, int> my_pair = make_pair(vector_to_sort[i], (int) i);
    vect.push_back( my_pair );
  }
  sort( vect.begin(), vect.end() );
  return vect;
}

void prune_image_paths_and_image_numbers( vector< string > &testing_paths_preliminary, vector< int > &numbers_testing_images, int skip_freq, int starting_image_number ) {
  if (starting_image_number < 0) starting_image_number = 0;
  if ( (int) testing_paths_preliminary.size() != (int) numbers_testing_images.size() ) throw runtime_error( "aaaa" );
  vector< string > testing_paths_preliminary_new;
  vector< int > numbers_testing_images_new;
  int processing = 0;
  for ( int i = 0; i < (int) testing_paths_preliminary.size(); i++ ) {
    if ( numbers_testing_images[i] == starting_image_number ) {
      processing = 1;
    }
    if (!processing) {
      continue;
    }
    if ( skip_freq <= 0 ||  ( i - starting_image_number ) % skip_freq == 0 ) {
      testing_paths_preliminary_new.push_back( testing_paths_preliminary[i] );
      numbers_testing_images_new.push_back( numbers_testing_images[i] );
    }
  }
  testing_paths_preliminary = testing_paths_preliminary_new;
  numbers_testing_images = numbers_testing_images_new;
}

void sort_images_and_image_numbers( vector< string > &testing_paths_preliminary, vector< int > &numbers_testing_images ) {
  // sort the image order by the image number parsed
  // 
  vector< string > testing_paths_preliminary_new( (int) testing_paths_preliminary.size() );
  vector< int > numbers_testing_images_new( (int) numbers_testing_images.size() );
  vector<pair<int, int> > vect = sort_vector_with_indices( numbers_testing_images );
  if ( (int) testing_paths_preliminary.size() != (int) numbers_testing_images.size() || (int) testing_paths_preliminary.size() != (int) vect.size() ) {
    throw runtime_error( "The testing path lengths do not match?" );
  }
  for (int i = 0; i < (int) vect.size(); i++ ) {
    int idx_sorted = vect[i].second;
    numbers_testing_images_new[i] = numbers_testing_images[idx_sorted];
    testing_paths_preliminary_new[i] = testing_paths_preliminary[idx_sorted];
  }
  numbers_testing_images = numbers_testing_images_new;
  testing_paths_preliminary = testing_paths_preliminary_new;
}

struct filepaths_apple_project setup_correct_apple_paths( string dataset_path ) {
  struct filepaths_apple_project fpaths;
  
  string mask_path_str = dataset_path + string( "/train/mask/" );
  string raw_path_str = dataset_path + string( "/train/raw/" );
  string mask_of_shape_path_str = dataset_path + string( "/train/mask_of_shape/" );
  
  string row_path = dataset_path + string("/rows/row001_Eside_S2N/cam0_images/");
  
  string extension_mask("_mask.png");
  string extension_mask_of_shape("_mask_of_tops_and_bottoms.png");
  string extension_testing(".jpg");
  
  // raw
  vector<string> raw_training_fnames = get_files_with_extension( raw_path_str, extension_testing );
  vector<string> raw_training_paths = create_paths( raw_path_str, raw_training_fnames );
  string ext_before("camA_");
  string ext_after("");
  vector< int > numbers_raw_images = get_image_numbers( raw_training_paths, ext_before, ext_after );
  // mask
  vector<string> mask_fnames = get_files_with_extension( mask_path_str, extension_mask );
  vector<string> mask_paths_preliminary = create_paths( mask_path_str, mask_fnames );
  string ext_before_mask("camA_");
  string ext_after_mask("_mask");
  vector< int > numbers_mask_images = get_image_numbers( mask_paths_preliminary, ext_before_mask, ext_after_mask );
  vector<string> mask_paths_correlated = correlate_image_paths( numbers_raw_images, numbers_mask_images, mask_paths_preliminary );
  // mos
  vector<string> mos_fnames = get_files_with_extension( mask_of_shape_path_str, extension_mask_of_shape );
  vector<string> mos_paths_preliminary = create_paths( mask_of_shape_path_str, mos_fnames );
  string ext_before_mos("camA_");
  string ext_after_mos("_mask_of_tops_and_bottoms");
  vector< int > numbers_mos_images = get_image_numbers( mos_paths_preliminary, ext_before_mos, ext_after_mos );
  vector<string> mos_paths_correlated = correlate_image_paths( numbers_raw_images, numbers_mos_images, mos_paths_preliminary );
  
  // testing images
  vector<string> testing_fnames = get_files_with_extension( row_path, extension_testing );
  vector<string> testing_paths_preliminary = create_paths( row_path, testing_fnames );
  string ext_before_testing("camA_");
  string ext_after_testing("");
  vector< int > numbers_testing_images = get_image_numbers( testing_paths_preliminary, ext_before_testing, ext_after_testing );
  
  // sort the image order by the image number parsed
  sort_images_and_image_numbers( testing_paths_preliminary, numbers_testing_images );
  // for now, skip the first 40 images
  int skip_freq = 0;
  int starting_number = 60;
  prune_image_paths_and_image_numbers( testing_paths_preliminary, numbers_testing_images, skip_freq, starting_number );
  
  // really should update the image numbers to be the same size 
  fpaths.raw_training_paths = raw_training_paths;
  fpaths.mask_paths = mask_paths_correlated;
  fpaths.mos_paths = mos_paths_correlated;
  fpaths.all_types_image_numbers = numbers_raw_images;
  fpaths.mask_image_numbers = numbers_mask_images;
  fpaths.mos_image_numbers = numbers_mos_images;
  fpaths.testing_image_paths = testing_paths_preliminary;
  fpaths.testing_image_numbers = numbers_testing_images;
  
  for ( int i = 0; i < (int) fpaths.raw_training_paths.size(); i++ ) {
    cout << "before: fpaths.raw_training_paths:  " << fpaths.raw_training_paths[i] << endl;
    if ( i < 3 ) cout << "before: fpaths.mask_paths:  " << fpaths.mask_paths[i] << endl;
    if ( i < 3 ) cout << "before: fpaths.mos_paths:  " << fpaths.mos_paths[i] << endl;

    cout << "before: fpaths.all_types_image_numbers:  " << fpaths.all_types_image_numbers[i] << endl;
    cout << "before: fpaths.mask_image_numbers:  " << fpaths.mask_image_numbers[i] << endl;
    cout << "before: fpaths.mos_image_numbers:  " << fpaths.mos_image_numbers[i] << endl;
    cout <<  endl;
  }
  remove_extraneous_paths( fpaths );
  for ( int i = 0; i < (int) fpaths.raw_training_paths.size(); i++ ) {
    cout << "fpaths.raw_training_paths:  " << fpaths.raw_training_paths[i] << endl;
    if ( i < 3 )  cout << "fpaths.mask_paths:  " << fpaths.mask_paths[i] << endl;
    if ( i < 3 )  cout << "fpaths.mos_paths:  " << fpaths.mos_paths[i] << endl << endl;
  }
  return fpaths;
}
