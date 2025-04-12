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

def convert_video_to_grayscale(file_directory: str, file_name: str) -> None:
    if file_name.lower().endswith('.mp4'):
        input_path = os.path.join(file_directory, file_name)
        output_path = os.path.join(file_directory, f"bw_{file_name}")

        if os.path.exists(output_path):
            print(f"⚠️ Output file already exists: {output_path}")
            return

        print(f"Processing: {input_path}")

        # Open the video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ Failed to open {input_path}")
            return

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
        return