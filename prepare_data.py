import os
import shutil
import re

# Function to count the number of files with a specific extension in a folder
def count_files(folder_path, extension):
    count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(extension):
            count += 1
    return count

# Main function to iterate through the main folder and subfolders
def count_images_in_subfolders(main_folder):
    for i in range(8):  # Assuming you have subfolders iqs_dv_0 to iqs_dv_8
        subfolder_path = os.path.join(main_folder, f'iqs_dv_0{i+1}')
        
        if os.path.exists(subfolder_path):
            print(f"Subfolder 'iqs_dv_0{i+1}':",subfolder_path)
            
            for subfolder_name in ['images', 'masks', 'zeiss_annotations']:
                subfolder_full_path = os.path.join(subfolder_path, subfolder_name)
                if os.path.exists(subfolder_full_path):
                    if subfolder_name == "zeiss_annotations":
                        file_count = count_files(subfolder_full_path, '.json')
                    else:   
                        file_count = count_files(subfolder_full_path, '.h5')  # Change the extension as needed
                    print(f"  {subfolder_name}: {file_count} images",subfolder_full_path)
                else:
                    print(f"  {subfolder_name}: Folder not found",subfolder_full_path)
        else:
            print(f"Subfolder 'iqs_dv_0{i+1}': Folder not found",subfolder_path)
        print()




def move_unlabeled_images(main_folder):
    for i in range(8): 
        subfolder_path = os.path.join(main_folder, f'iqs_dv_0{i+1}')
        unlabeled_images_path = os.path.join(subfolder_path, 'unlabeled_images')
        unlabeled_annotations_path = os.path.join(subfolder_path, 'unlabeled_annotations')
        
        if os.path.exists(subfolder_path):
            os.makedirs(unlabeled_images_path, exist_ok=True)
            os.makedirs(unlabeled_annotations_path, exist_ok=True)

            for image_name in os.listdir(os.path.join(subfolder_path, 'images')):
                image_path = os.path.join(subfolder_path, 'images', image_name)
                match = re.search(r'_dataset_(\d+)\.h5$', image_name)
                dataset_number = match.group(1)

                mask_filename = re.sub(r'_dataset_(\d+\.h5)$', fr'_data_data_manual_dataset_{dataset_number}.h5', image_name)
                mask_path = os.path.join(subfolder_path, 'masks', mask_filename)  
                mask_filename_2 = re.sub(r'_dataset_(\d+\.h5)$', fr'_data_data_generated_dataset_{dataset_number}.h5', image_name)
                mask_path_2 = os.path.join(subfolder_path, 'masks', mask_filename_2) 

                if not os.path.exists(mask_path) and not os.path.exists(mask_path_2) :
                    # Move image to unlabeled_images
                    shutil.move(image_path, os.path.join(unlabeled_images_path, image_name))
                    annotation_filename = re.sub(r'_dataset_(\d+\.h5)$', fr'_data_data_dataset_{dataset_number}.json', image_name)
                    
                    annotation_path = os.path.join(subfolder_path, 'zeiss_annotations', annotation_filename )
                    
                    if os.path.exists(annotation_path):
                        # Move annotation to unlabeled_annotations
                        shutil.move(annotation_path, os.path.join(unlabeled_annotations_path, annotation_filename))
        else:
            print(f"Subfolder 'iqs_dv_0{i+1}': Folder not found")
        

import os
import shutil

def merge_train_val_test(main_folder):
    
    final_train_path = os.path.join(main_folder, 'train')
    final_val_path = os.path.join(main_folder, 'val')
    final_test_path = os.path.join(main_folder, 'test')

    os.makedirs(final_train_path, exist_ok=True)
    os.makedirs(final_val_path, exist_ok=True)
    os.makedirs(final_test_path, exist_ok=True)
    for i in range(8):  
        subfolder_path = os.path.join(main_folder, f'iqs_dv_{i+1}')
        for dataset_type in ['train', 'val', 'test']:
            dataset_path = os.path.join(subfolder_path, f'iqs_dv_{i+1}_{dataset_type}')

            if dataset_type == 'train':
                final_destination = final_train_path
            elif dataset_type == 'val':
                final_destination = final_val_path
            elif dataset_type == 'test':
                final_destination = final_test_path

            if os.path.exists(dataset_path):
                # Move all files from the dataset_path to the final_destination
                for file_name in os.listdir(dataset_path):
                    source_file_path = os.path.join(dataset_path, file_name)
                    destination_file_path = os.path.join(final_destination, file_name)
                    shutil.move(source_file_path, destination_file_path)
                
                # Remove the empty dataset folder
                os.rmdir(dataset_path)
    

def add_dataset_number_to_filenames(main_folder):
    for i in range(8):  
        subfolder_path = os.path.join(main_folder, f'iqs_dv_0{i+1}')

        if os.path.exists(subfolder_path):
            
            for folder_name in ['masks', 'images', 'zeiss_annotations']:
                folder_path = os.path.join(subfolder_path, folder_name)
                if os.path.exists(folder_path):
                    print(folder_path)
                    for file_name in os.listdir(folder_path):
                        # Extract file extension
                        file_base, file_extension = os.path.splitext(file_name)

                        # Add dataset number to the file name
                        new_file_name = f"{file_base}_dataset_{i+1}{file_extension}"

                        # Construct full paths
                        source_file_path = os.path.join(folder_path, file_name)
                        destination_file_path = os.path.join(folder_path, new_file_name)

                        # Rename the file
                        os.rename(source_file_path, destination_file_path)

        else:
            print(f"Subfolder 'iqs_dv_0{i+1}': Folder not found")

        #dataset_index = int(filename.split('_')[-1].replace("dataset", "").replace(".h5", "")) 
# Replace 'path_to_your_folder' with the actual path to your main folder
main_folder_path = '/home/zozchaab/data/deepvision/deepvision'
os.chdir(main_folder_path)
#count_images_in_subfolders(os.getcwd())
print(os.getcwd())
#add_dataset_number_to_filenames(main_folder_path) 
#move_unlabeled_images(main_folder_path)
"""
cp -r /home/zozchaab/data/deepvision/deepvision/iqs_dv_01/* /home/zozchaab/data/deepvision/deepvision/iqs_dv_val/
cp -r -u /home/zozchaab/data/deepvision/deepvision/iqs_dv_02/* /home/zozchaab/data/deepvision/deepvision/iqs_dv_val/
cp -r /home/zozchaab/data/deepvision/deepvision/iqs_dv_03/* /home/zozchaab/data/deepvision/deepvision/iqs_dv_test/
cp -r /home/zozchaab/data/deepvision/deepvision/iqs_dv_04/* /home/zozchaab/data/deepvision/deepvision/iqs_dv_train/
cp -r /home/zozchaab/data/deepvision/deepvision/iqs_dv_05/* /home/zozchaab/data/deepvision/deepvision/iqs_dv_test/
cp -r /home/zozchaab/data/deepvision/deepvision/iqs_dv_06/* /home/zozchaab/data/deepvision/deepvision/iqs_dv_test/
cp -r /home/zozchaab/data/deepvision/deepvision/iqs_dv_07/* /home/zozchaab/data/deepvision/deepvision/iqs_dv_train/
cp -r /home/zozchaab/data/deepvision/deepvision/iqs_dv_08/* /home/zozchaab/data/deepvision/deepvision/iqs_dv_train/
""" 
import os
import h5py


def list_objects(group, indent=0):
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            print(f"{' ' * indent}Group: {name}")
            list_objects(item, indent + 2)
        elif isinstance(item, h5py.Dataset):
            print(f"{' ' * indent}Dataset: {name} (Shape: {item.shape}, Dtype: {item.dtype})")
        else:
            print(f"{' ' * indent}Unknown object type: {name}")

def read_h5_file(file_path):
    with h5py.File(file_path, 'r') as file:
        print(f"Objects in {file_path}:")
        list_objects(file)

def iterate_h5_files(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a directory.")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and file_path.lower().endswith('.h5'):
            read_h5_file(file_path)

# Specify the folder containing the .h5 files
folder_path = '/home/zozchaab/data/deepvision/deepvision/iqs_dv_test/masks'

# Call the function to iterate through .h5 files in the folder
iterate_h5_files(folder_path)
