usr_path = `pkg-config  opencv  --cflags`
vl_feat_dir = /home/kyle/Downloads/vlfeat-0.9.19
include_local = -I include/ -I $(vl_feat_dir)

link_path = /usr/local/lib/libopencv_calib3d.so /usr/local/lib/libopencv_contrib.so /usr/local/lib/libopencv_core.so /usr/local/lib/libopencv_features2d.so /usr/local/lib/libopencv_flann.so /usr/local/lib/libopencv_gpu.so /usr/local/lib/libopencv_highgui.so /usr/local/lib/libopencv_imgproc.so /usr/local/lib/libopencv_legacy.so /usr/local/lib/libopencv_ml.so /usr/local/lib/libopencv_nonfree.so /usr/local/lib/libopencv_objdetect.so /usr/local/lib/libopencv_ocl.so /usr/local/lib/libopencv_photo.so /usr/local/lib/libopencv_stitching.so /usr/local/lib/libopencv_superres.so /usr/local/lib/libopencv_ts.a /usr/local/lib/libopencv_video.so /usr/local/lib/libopencv_videostab.so

src_dir = src/
std = -std=c++0x
vl_link = -L $(vl_feat_dir)/bin/glnxa64/ -l vl

all : build/feature_computation.o build/utility.o build/setup_feature_sets.o build/superpixel_computation.o build/visualization.o build/classify_images.o build/routine.o build/setup_correct_apple_paths.o
	gcc -g -I $(usr_path) $(include_local) -Wl,-rpath=/usr/local/lib,-rpath=${vl_feat_dir}/bin/glnxa64/ build/setup_correct_apple_paths.o build/feature_computation.o build/utility.o build/setup_feature_sets.o build/superpixel_computation.o build/visualization.o build/classify_images.o build/routine.o -o build/main.exe $(link_path) $(vl_link)

build/setup_correct_apple_paths.o : src/setup_correct_apple_paths.cpp
	gcc -g -c -I $(usr_path) $(include_local) $(src_dir)/setup_correct_apple_paths.cpp -o build/setup_correct_apple_paths.o

build/routine.o : src/routine_superpixel_texture_and_color_majority.cpp
	gcc -g -c -I $(usr_path) $(include_local) $(src_dir)/routine_superpixel_texture_and_color_majority.cpp -o build/routine.o

build/feature_computation.o : $(src_dir)/feature_computation.cpp
	gcc -g -c -I $(usr_path) $(include_local) $(src_dir)/feature_computation.cpp -o build/feature_computation.o

build/utility.o : $(src_dir)/utility.cpp
	gcc -g -c -I $(usr_path) $(include_local) $(src_dir)/utility.cpp -o build/utility.o

build/setup_feature_sets.o : $(src_dir)/setup_feature_sets.cpp 
	gcc -g -c -I $(usr_path) $(include_local) $(src_dir)/setup_feature_sets.cpp -o build/setup_feature_sets.o

build/superpixel_computation.o : $(src_dir)/superpixel_computation.cpp 
	gcc -g -c -I $(usr_path) $(include_local) $(src_dir)/superpixel_computation.cpp -o build/superpixel_computation.o

build/visualization.o : $(src_dir)/visualization.cpp
	gcc -g -c -I $(usr_path) $(include_local) $(src_dir)/visualization.cpp -o build/visualization.o

build/classify_images.o : $(src_dir)/classify_images.cpp 
	gcc -g -c -I $(usr_path) $(include_local) $(src_dir)/classify_images.cpp -o build/classify_images.o

clean :
	rm build/*.o build/*.exe 
