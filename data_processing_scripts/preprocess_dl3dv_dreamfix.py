import os
import json
import shutil
import numpy as np
import subprocess
import glob

ids = ['2385549d398bdfb55ed17b547c9846f0d01d571b4e050f90df707b577084fb55',
       'cd9c981eeb4a9091547af19181b382698e9d9eee0a838c7c9783a8a268af6aee',
       '54bf355ca7e08ed1bc86f5772e564ac0f92981ca25dab24d86b694e915fc4c43',
       '2f3e1c0f688c84cec67f9a1ea219c54c14ffabf31a046e620dacc690cac2f1bd',
       'ceb252f5d4419510655cf9ed7afbf3e8e688825f798d80414c7715dc8ace153a',
       'd3812aad538261e7f73c75762ff55f23b468bcc76f376d52ac86ca6cf3c44b4b',
       '35872363e17af5d173b6a0b09fcf5de94627ad5dc5f8a9ad4c579f3e70b4797a',
       '50c46cf8b8b22c8d2ffdef8964b05ddbceaef312c9a9ff331d1ecebfd223f72a',
       'adf35184a12d4cfa3f4248b87aa5adb4f39f179df460d6d76136e13d37299a2a',
       '032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7',
       'eb4cf52988f805e6fce11d1b239fa9de32eb157364cff06ebac0aa50e0a46567',
       '493816813d2d6d248eb3c2b0b77b63e54235266e9a06e270fd0d282f13960493',
       '7a9f97660be8f2421b37dd413c898360532f17c6447c11515e8aded2d80eb99d',
       'fb3b73f1d3fe9d192f21f55f5100fd258887aef345f778e0a64fc0587930a6f9',
       '15ff83e2531668d27c92091c97d31401ce323e24ee7c844cb32d5109ab9335f7',
       '3b7529dcccf7097cba31a989536035595e214b0f3775867fd3a62eb186870090',
       '918c8dad730c3b804306c5da8486124be4aa0612e85fb825338fd350c912e1b0',
       'd4fbeba0168af8fddb2fc695881787aedcd62f477c7dcec9ebca7b8594bbd95b',
       'c076929db6501cf7ebe386c70e6d77ea3af844a745e794f2ec17c981c465a69b',
       '8b9fb9d9f10e8c64d5034be69809465753b8ba88ef12da82afd47d38ee789934',
       '280abf7bd93b81b077af1db638229dbb09869052fa0b7b57c81c94d2db893829',
       '8fdc5130f0731360cc053f59049bfe6db509567457c5471c970292966cb34f17',
       '5c8dafad7d782c76ffad8c14e9e1244ce2b83aa12324c54a3cc10176964acf04',
       '35317e621976e87f0c143e66fc61fb8cddb4ff134304da7a00e32ac1983105b4',
       'a401469cb0fa576b93e836a4a6d712394f1a55a0a53a42250286980bd078e669',
       'a62c330f5403e2e41a82a74c4e865b705c5706843b992fae2fe2e538b122d984',
       '946f49be73928469000baa5ca04d2573137c5ee6a66362bcf8d130354dca8924',
       'e8ce51b6abfe05bf8dca47e29c8be6c1e6de27a8c9fece7a121400b931b2ca0f',
       '70eac6ff18a1daae4eeccc5eb18723eaf7e029a77e79c79d9573f7bac59ba92b',]
ids = ids[:1]

num_views = 6

for scene_id in ids:

    print('scene_id', scene_id, '-'*50)
    # Source and destination paths
    source_base_path = "/share/phoenix/nfs04/S7/yc2463/Difix3D/dataset/DL3DV-10K-Benchmark/" + scene_id + "/nerfstudio/"
    destination_base_path = f"/share/phoenix/nfs04/S7/yc2463/Difix3D_fork/dataset/DL3DV-Benchmark-Dreamfix-Split/{scene_id}"

    # Ensure the destination directory exists
    if not os.path.exists(destination_base_path):
        os.makedirs(destination_base_path)
        print(f"Created destination directory: {destination_base_path}")

    # Load the transforms.json file
    transforms_path = os.path.join(source_base_path, "transforms.json")
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded transforms.json from {transforms_path}")

    # Load the sampled indices json file
    sampled_indices_file_path = f"/share/phoenix/nfs04/S7/yc2463/Difix3D_fork/train_test_split/DL3DV-Benchmark-Dreamfix-Split/{scene_id}/half_covisibility_sampled_indices_0.json"
    with open(sampled_indices_file_path, 'r') as fin:
        sampled_indices = json.load(fin)
    train_ids = sampled_indices[:num_views]
    print(f"Loaded sampled indices from {sampled_indices_file_path}")

    # Prepare the images directory in the destination path

    source_images_dir = os.path.join(source_base_path, "images")
    for fd_name in ["images_4", "images"]:
        destination_images_dir = os.path.join(destination_base_path, fd_name)

        if not os.path.exists(destination_images_dir):
            os.makedirs(destination_images_dir)
            print(f"Create images directory at: {destination_images_dir}")

    # Update frames with new file paths and copy images
    for idx, frame in enumerate(data["frames"]):
        original_file_path = frame["file_path"] # e.g., "./images/frame_XXXX.png"
        image_name = os.path.basename(original_file_path)

        # Determine new file name based on cluster
        if idx in train_ids:
            new_image_name = image_name.replace("frame_", "frame_train_")
        else:
            new_image_name = image_name.replace("frame_", "frame_eval_")

        # Update the file_path in the JSON data
        new_file_path = "./images/" + new_image_name
        frame["file_path"] = new_file_path

        # Copy and rename the image file
        src_image_path = os.path.join(source_base_path, original_file_path.strip("./"))
        dest_image_path = os.path.join(destination_base_path, new_file_path.strip("./"))


        # Ensure the destination directory exists
        dest_image_dir = os.path.dirname(dest_image_path)
        if not os.path.exists(dest_image_dir):
            os.makedirs(dest_image_dir)

        # shutil.copy(src_image_path, dest_image_path)
        # for fd_name in ["images_2", "images_4", "images_8", "images"]:
        for fd_name in ["images_4", "images"]:
            subprocess.run(['cp', src_image_path.replace('/images/', '/' + fd_name + '/'),
                            dest_image_path.replace('/images/', '/' + fd_name + '/')], check=True)

        # print(f"Cluster: {cluster_label}, Original Image: {src_image_path}, New Image: {dest_image_path}")

    # Save the updated transforms.json
    updated_transforms_path = os.path.join(destination_base_path, "transforms.json")
    with open(updated_transforms_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Updated transforms.json saved to {updated_transforms_path}")

    # Copy other folders if necessary (colmap, images_2, images_4, images_8)
    folders_to_copy = ["colmap"]
    for folder_name in folders_to_copy:
        src_folder = os.path.join(source_base_path, folder_name)
        dest_folder = os.path.join(destination_base_path, folder_name)
        if os.path.exists(src_folder):
            #shutil.copytree(src_folder, dest_folder)
            subprocess.run(['cp', '-r', src_folder, dest_folder], check=True)
            print(f"Copied {folder_name} to destination.")
        else:
            print(f"Folder {folder_name} does not exist in source.")
    print("Processing completed.")
