function [] = project_predicted_positions()
  row_path = '/media/YIELDEST_KY/apple/2011_09_15_sunrise_9b/rows/row001_Eside_S2N/';
  
  % path to all raw images
  path_raw_imgs = sprintf( '%s/cam0_images/', row_path );
  % path to the detections
  path_detection_results = '/home/kyle/undergraduate_thesis/my_code/C++/build/green/';
  % full list of raw images
  raw_imgs_regex = sprintf( '%s/*.jpg', path_raw_imgs );
  raw_imgs_paths = cell_array_of_paths( raw_imgs_regex );
  [raw_imgs_paths_new, raw_imgs_numbers] = sort_images_by_numbers(raw_imgs_paths, 'camA_', '');
  
  % full list of detection result superpixel images
  superpixel_imgs_regex = sprintf( '%s/superpixel_image_*.png', path_detection_results );
  superpixel_imgs_paths = cell_array_of_paths( superpixel_imgs_regex );
  [superpixel_imgs_paths, superpixel_imgs_numbers] = sort_images_by_numbers(superpixel_imgs_paths, 'superpixel_image_', '');
  
  % full list of detection result probabilities
  probabilities_regex = sprintf( '%s/apple_probabilities_*.pgm', path_detection_results );
  probability_imgs_paths = cell_array_of_paths( probabilities_regex );
  [probability_imgs_paths_new, probability_imgs_numbers] = sort_images_by_numbers(probability_imgs_paths, 'apple_probabilities_', '');
  
  images_raw = false( 1000, 1 );
  images_raw( raw_imgs_numbers+1 ) = true;
  images_sup = false( 1000, 1 );
  images_sup( superpixel_imgs_numbers+1 ) = true;
  images_prob = false( 1000, 1 );
  images_prob( probability_imgs_numbers+1 ) = true;
  
  all = images_raw & images_sup & images_prob;
  all_image_types_found = find(all);
  
  % load all of the vehicle pose data
  vehicle_pose_row_path = sprintf( '%s/vehicle_pose_row.txt', row_path );
  vpr = load( vehicle_pose_row_path );
  
  % parameters for projecting the detections onto the images
  distance_to_vine = 2;
  max_distance_gaussian = 0.25;
  
  for i = 1:numel( all_image_types_found )
    cur_image_num = all_image_types_found(i);
    next_image_num = cur_image_num+1;
    
    raw_imgs_find = find( (cur_image_num-1) == raw_imgs_numbers );
    cur_raw_img_path = sprintf('%s/%s', path_raw_imgs, raw_imgs_paths_new{ raw_imgs_find(1) });
    next_raw_imgs_find = find( (next_image_num-1) == raw_imgs_numbers );
    next_raw_img_path = sprintf('%s/%s', path_raw_imgs, raw_imgs_paths_new{ next_raw_imgs_find(1) });

    sp_imgs_find = find( (cur_image_num-1) == superpixel_imgs_numbers );
    cur_sp_img_path = sprintf('%s/%s', path_detection_results, superpixel_imgs_paths{ sp_imgs_find(1) });
    next_sp_imgs_find = find( (next_image_num-1) == superpixel_imgs_numbers);
    next_sp_img_path = sprintf('%s/%s', path_detection_results, superpixel_imgs_paths{ next_sp_imgs_find(1) });
    
    prob_imgs_find = find( (cur_image_num-1) == probability_imgs_numbers );
    cur_probability_img_path = sprintf('%s/%s', path_detection_results, probability_imgs_paths_new{ prob_imgs_find(1) });
    next_prob_imgs_find = find( (next_image_num-1) == probability_imgs_numbers );
    next_probability_img_path = sprintf('%s/%s', path_detection_results, probability_imgs_paths_new{ next_prob_imgs_find(1) });
    
    vpr_cur = vpr(cur_image_num, :);
    vpr_cur_x = vpr_cur(1);
    vpr_next = vpr(next_image_num, :);
    vpr_next_x = vpr_next(1);
    
    cur_raw_img = imread(cur_raw_img_path);
    cur_raw_img = rot90( cur_raw_img );
    next_raw_img = imread(next_raw_img_path);
    next_raw_img = rot90( next_raw_img );
    cur_probability_img = imread( cur_probability_img_path );
    next_probability_img = imread( next_probability_img_path );
    cur_sp_img = imread( cur_sp_img_path );
    next_sp_img = imread( next_sp_img_path );
    
    
    C = [ 2042.68, 0, 1461.91; 0, 2032.71, 2149.38; 0, 0, 1 ];
    visualize = 0;
    if visualize == 1
      % Test that I can project a point in 2 sequential images
      pt_first_image = [2000; 3000; 1];
      normalized_vec = project_onto_fruitwall( pt_first_image, C );
      projected_X = normalized_vec*vpr_cur(2);
      offset_X = zeros(size(projected_X));
      offset_X(1) = vpr_next(1) - vpr_cur(1);
      predicted_X = projected_X - offset_X;
      predicted_X = predicted_X / vpr_cur(2);
      predicted_X = [predicted_X'; 1];
      pt_second_image = backproject_from_fruitwall( predicted_X, C );

      % show both pt_first_image and pt_second_image 
      figure, imshow(cur_raw_img), hold on, plot( pt_first_image(1), pt_first_image(2), 'g*' );
      figure, imshow(next_raw_img), hold on, plot( pt_second_image(1), pt_second_image(2), 'g*' );
      keyboard;
    end
    %%%%%%% projection of the superpixels onto the images %%%%%%% 
    total_pix = numel(cur_sp_img);
    total_sp = max(double(cur_sp_img(:)));
    avg_size = total_pix / total_sp;
    avg_r = (avg_size/pi)^0.5;
    
    % for each superpixel:
    %   compute the projection onto the next image
    %   calculate the radius based on (prob**2)*max_radius
    should_check = cur_probability_img > 150;
    unique_superpixel_values_sorted = sort( unique( cur_sp_img( should_check ) ), 'ascend');
    
    save_centroids = 1;
    centroids_path = 'save_file.mat';
    if save_centroids == 1
      all_centroids = [];
      probabilities = [];
      cur_idx = 1;
      for j = 1:total_sp
        if mod( j, 100) == 0; disp(j); end
        if cur_idx > numel( unique_superpixel_values_sorted ) continue; end
        if ( unique_superpixel_values_sorted(cur_idx) == j )
          cur_idx = cur_idx + 1;
        else
          continue;
        end
        % disp(j);
        cur_sp_b = j == cur_sp_img;
        % get the average superpixel certainty for the image
        valid_probability_values = cur_probability_img(cur_sp_b);
        avg_valid_probability_values =  mean( double(valid_probability_values(:) ) ) / 255.0;

        s = regionprops( cur_sp_b, 'centroid' );
        % if ( abs(centroids(1) - centroids(2)) < 40 ) continue; end
        % keyboard;
        all_centroids = [all_centroids; round([s(1).Centroid(1), s(1).Centroid(2)])];
        probabilities = [probabilities; avg_valid_probability_values];
      end
      save( centroids_path, 'all_centroids', 'probabilities' );
    else
      load( centroids_path, 'all_centroids', 'probabilities' );
    end
    
    % draw the centroids
    % translate all of the centroids to the next image
    % save the translated centroids if they look good
    translated_centroids = zeros(size(all_centroids));
    for j = 1:size(all_centroids, 1)
      pt_first_image = [all_centroids(j, 1); all_centroids(j, 2); 1];
      normalized_vec = project_onto_fruitwall( pt_first_image, C );
      projected_X = normalized_vec*vpr_cur(2);
      offset_X = zeros(size(projected_X));
      offset_X(1) = vpr_next(1) - vpr_cur(1);
      predicted_X = projected_X - offset_X;
      predicted_X = predicted_X / vpr_cur(2);
      predicted_X = [predicted_X'; 1];
      pt_second_image = backproject_from_fruitwall( predicted_X, C );
      translated_centroids(j, 1) = pt_second_image(1); translated_centroids(j, 2) = pt_second_image(2);
    end
    
    visualize2 = 1;
    if visualize2
      figure, imshow(cur_raw_img), hold on, plot( all_centroids(:,1), all_centroids(:,2), 'g*' );
      figure, imshow(next_raw_img), hold on, plot( translated_centroids(:,1), translated_centroids(:,2), 'g*' );
      keyboard;
    end
    
    translated_cords_mat = 'tcm.mat';
    save_cur = 1;
    if save_cur
      save( translated_cords_mat, 'translated_centroids' );
    else
      load( translated_cords_mat, 'translated_centroids' );
    end
    
    superpixel_predictions = zeros(size(cur_sp_img));
    disp( ['have ', num2str( numel( probabilities ) ) , ' probabilities to draw...'] );
    for j = 1:numel( probabilities )
      radius_value = 30 * avg_r * probabilities(j)^2;
      superpixel_predictions = draw_centroids_as_gaussians( superpixel_predictions, translated_centroids(j, :), radius_value );
      
    end
    figure, imagesc( superpixel_predictions );
    figure, imshow( cur_probability_img );
    keyboard;
  end
end

function [X] = project_onto_fruitwall( pts, C )
  % outputs the normalized vector to the detection
  x_vals = pts(1, :);
  y_vals = pts(2, :);
  X_x = (x_vals-C(1, 3))/C(1, 1);
  X_y = (y_vals-C(2, 3))/C(2, 2);
  X = [X_x, X_y];
end

function [pts] = backproject_from_fruitwall( X, C )
  % input needs to be the normalized vector detection
  pts = C * X;
end

function [superpixel_predictions] = draw_centroids_as_gaussians( superpixel_predictions, centroids, radius_size )
  % get centroid x and y
  radius_size2 = radius_size;
  xx = round(centroids(1));
  yy = round(centroids(2));
  if (xx < radius_size2 || xx > size(superpixel_predictions, 2)-radius_size2 || yy < radius_size2 || yy > size(superpixel_predictions, 1)-radius_size2)
    return;
  end
  patch = gaussian_second( radius_size );
  % gaussian2d( radius_size, 2.5 );
  % draw the gaussian filter onto the image
  start_x = xx - size(patch, 2);
  start_y = yy - size(patch, 1);
  superpixel_predictions( start_y:(start_y+size(patch, 1)-1), start_x:(start_x+size(patch, 2)-1) ) = superpixel_predictions( start_y:(start_y+size(patch, 1)-1), start_x:(start_x+size(patch, 2)-1) ) + patch;
end

function f=gaussian2d(N,sigma)
  % N is grid size, sigma speaks for itself
  [x y]=meshgrid(round(-N/2):round(N/2), round(-N/2):round(N/2));
  f=exp(-x.^2/(2*sigma^2)-y.^2/(2*sigma^2));
  f=f./sum(f(:));
end

function z = gaussian_second(N)
  x=linspace(-3,3,N);
  y=x';
  [X,Y]=meshgrid(x,y);
  z=exp(-(X.^2+Y.^2)/2);
end

function [superpixel_imgs_paths] = cell_array_of_paths( superpixel_imgs_regex )
  superpixel_imgs_dir = dir( superpixel_imgs_regex );
  superpixel_imgs_paths = cell(size(superpixel_imgs_dir, 1), 1);
  for i = 1:size( superpixel_imgs_dir, 1 )
    superpixel_imgs_paths{i, 1} = superpixel_imgs_dir(i).name;
  end
end

function [image_numbers] = get_image_numbers( image_paths_new, str_before, str_after )
  image_numbers = zeros( numel(image_paths_new), 1 );
  for i = 1:numel(image_paths_new)
    image_path_cur =  image_paths_new{i};
    [~, n, ~] = fileparts( image_path_cur );
    chopped_str = n(numel(str_before)+1:end);
    num = str2double(chopped_str);
    image_numbers(i) = num;
  end
end

function [image_paths_new, image_numbers] = sort_images_by_numbers(image_paths, str_before, str_after)
  [image_numbers] = get_image_numbers( image_paths, str_before, str_after );
  % indices are all of the old positions to make the new sorted array
  [vals, indices] = sort(image_numbers, 'ascend');
  image_paths_new = cell(size(image_paths, 1), 1);
  [image_paths_new{:}] = image_paths{indices};
  image_numbers = image_numbers(indices);
end