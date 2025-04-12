import os
import cv2

def print_directory_tree(startpath, indent=''):
    for item in sorted(os.listdir(startpath)):
        full_path = os.path.join(startpath, item)
        if os.path.isdir(full_path):
            print(f"{indent}📁 {item}/")
            print_directory_tree(full_path, indent + '    ')
        else:
            print(f"{indent}📄 {item}")

def convert_videos_to_grayscale(input_directory, output_base):
    # Create the output base directory if it doesn't exist
    os.makedirs(output_base, exist_ok=True)
    
    # Get the name of the original dataset directory
    original_dir_name = os.path.basename(os.path.normpath(input_directory))
    
    # Create the same structure in the output directory
    output_directory = os.path.join(output_base, original_dir_name)
    os.makedirs(output_directory, exist_ok=True)
    
    for root, dirs, files in os.walk(input_directory):
        # Create all subdirectories in output
        for dir_name in dirs:
            input_subdir_path = os.path.join(root, dir_name)
            # Get relative path from input_directory
            rel_path = os.path.relpath(input_subdir_path, input_directory)
            # Create corresponding directory in output
            output_subdir_path = os.path.join(output_directory, rel_path)
            os.makedirs(output_subdir_path, exist_ok=True)
            
        # Process video files
        for file in files:
            if file.lower().endswith('.mp4'):
                input_path = os.path.join(root, file)
                
                # Get relative path from input_directory
                rel_dir = os.path.relpath(root, input_directory)
                # Create corresponding output path
                output_subdir = os.path.join(output_directory, rel_dir)
                os.makedirs(output_subdir, exist_ok=True)
                
                # Set output path maintaining original file name format
                output_path = os.path.join(output_subdir, f"bw_{file}")

                print(f"Processing: {input_path}")

                # Open the video file
                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    print(f"❌ Failed to open {input_path}")
                    continue

                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change to 'XVID' or 'avc1'
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    out.write(gray_frame)

                cap.release()
                out.release()
                print(f"✅ Saved grayscale video to: {output_path}\n")