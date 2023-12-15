import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from Barrelprojection import interp2, barrelBackProjection, evaluatePixel_Front

def extract_frames(video_path, save_path, skip_frames=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            frames.append(frame)
            frame_filename = f"{save_path}/frame_{frame_count}.jpg"
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()
    return frames

def load_images_from_folder(folder, color_mode=cv2.IMREAD_UNCHANGED):
    images = []
    for filename in os.listdir(folder):
        # Check for image extension
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            # Read and add the image to the list
            img = cv2.imread(img_path, color_mode)
            if img is not None:
                images.append(img)
    return images

def detect_features_sift(frames, roi):
    # Create a SIFT object
    sift = cv2.SIFT_create()
    keypoints_and_descriptors = []
    for frame in frames:
        # Apply the ROI to the frame
        roi_frame = frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        # Detect features in the ROI using SIFT
        kp, des = sift.detectAndCompute(roi_frame, None)
        # Adjust keypoint coordinates to match the original frame coordinates
        for k in kp:
            k.pt = (k.pt[0] + roi[0], k.pt[1] + roi[1])
        keypoints_and_descriptors.append((frame, kp, des))
    return keypoints_and_descriptors

def match_features_sift(keypoints_and_descriptors1, keypoints_and_descriptors2):
    # Use BFMatcher with appropriate norm type (cv2.NORM_L2 for SIFT)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matched_keypoints = []
    for (frame1, kp1, des1), (frame2, kp2, des2) in zip(keypoints_and_descriptors1, keypoints_and_descriptors2):
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography using RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # Filter the matched keypoints based on the mask
        filtered_src_pts = src_pts[mask.ravel() == 1]
        filtered_dst_pts = dst_pts[mask.ravel() == 1]

        matched_keypoints.append((filtered_src_pts, filtered_dst_pts))
    return matched_keypoints

def find_average_homography(frames1, frames2, matched_keypoints):
    # Check if the number of frames and number of matched keypoints are consistent
    # if not (len(frames1) == len(frames2) == len(matched_keypoints)):
    #     raise ValueError("The length of frames and matched keypoints must be the same.")

    # Initialize sum of homography matrices
    H_sum = np.zeros((2, 3), dtype=np.float64)

    # Number of pairs with valid homographies
    valid_pairs = 0

    # Iterate over the pairs of frames and their matched keypoints
    for (frame1, frame2, (src_pts, dst_pts)) in zip(frames1, frames2, matched_keypoints):
        # Estimate the homography matrix between each pair of frames
        # H, status = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        H, status = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold = 3.0)
        # H = np.vstack((H,np.array([0, 0, 1])))

        # Ensure that a valid homography was found
        if H is not None:
            H_sum += H
            valid_pairs += 1

    # Compute the average homography if at least one valid homography was found
    if valid_pairs > 0:
        average_H = H_sum / valid_pairs
        return average_H
    else:
        raise ValueError("No valid homography matrices were found.")

def create_panorama(frames1, frames2, matched_keypoints, H):
    panoramas = []
    for (frame1, frame2, (src_pts, dst_pts)) in zip(frames1, frames2, matched_keypoints):
        # Estimate the homography matrix.
        # H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        H = H
        # H, status = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold = 3.0)

        # Warp the second image to the perspective of the first
        warp_frame2 = cv2.warpAffine(frame2, H, (frame1.shape[1] + frame2.shape[1], max(frame1.shape[0], frame2.shape[0])))
        # display_image('panorama warp frame', warp_frame2)

        # Initialize a canvas large enough to hold the panorama
        panorama = np.zeros((max(frame1.shape[0], warp_frame2.shape[0]), frame1.shape[1] + frame2.shape[1], 3), dtype=np.uint8)

        # Place the first image on the canvas
        panorama[:frame1.shape[0], :frame1.shape[1]] = frame1

        # display_image('panorama first frame', panorama)

        # Create a mask for the pixels that are not black (image exists)
        

        # Create a masked array that includes the width of both images
        warp_masked = np.zeros_like(panorama)
        # print(panorama.shape)
        # print(warp_masked.shape)
        warp_masked[:warp_frame2.shape[0], :frame1.shape[1] + frame2.shape[1]] = warp_frame2[:warp_frame2.shape[0], :frame1.shape[1] + frame2.shape[1]]
        # display_image('panorama warp_mask frame', warp_masked)
        mask = (warp_masked > 0).any(axis=2)

        # Use the mask to blend the warped frame onto the panorama
        # display_image('panorama frame', panorama)
        # panorama = np.where(mask[:, :, None], np.add(panorama, warp_masked), panorama)
        panorama = np.where(mask[:, :, None], warp_masked, panorama)

        panoramas.append(panorama)

    return panoramas

if __name__ == "__main__":

    video_path_1 = './final_project/data_campus/4/vid1.mp4'
    video_path_2 = './final_project/data_campus/4/vid2.mp4'
    video_path_3 = './final_project/data_campus/4/vid3.mp4'
    video_path_4 = './final_project/data_campus/4/vid4.mp4'

    save_path_1 = './final_project/campus_video/frames/1'
    save_path_2 = './final_project/campus_video/frames/2'
    save_path_3 = './final_project/campus_video/frames/3'
    save_path_4 = './final_project/campus_video/frames/4'

    print("Extracting frames")
    frames_1 = extract_frames(video_path_1, save_path_1, skip_frames=1)
    frames_2 = extract_frames(video_path_2, save_path_2, skip_frames=1)
    frames_3 = extract_frames(video_path_3, save_path_3, skip_frames=1)
    # frames_4 = extract_frames(video_path_4, save_path_4, skip_frames=1)

    mapping = evaluatePixel_Front(np.array([1920,1080]))

    print("Undistorting frames")
    undistorted_frames_1 = [barrelBackProjection(frame, 1920, 1080, mapping) for frame in frames_1]
    del frames_1
    print("frame1 finished")
    undistorted_frames_2 = [barrelBackProjection(frame, 1920, 1080, mapping) for frame in frames_2]
    del frames_2
    print("frame2 finished")
    undistorted_frames_3 = [barrelBackProjection(frame, 1920, 1080, mapping) for frame in frames_3]
    del frames_3
    print("frame3 finished")
    # undistorted_frames_4 = [barrelBackProjection(frame, 1920, 1080, mapping) for frame in frames_4]
    # del frames_4
    # print("frame4 finished")

    panorama_generated = undistorted_frames_1
    for i in range(2):
        print("CREATING PANORAMA "+str(i+1))
        undistorted_frames_L = panorama_generated

        if i == 3:
            undistorted_frames_R = locals()['undistorted_frames_1']
        else:
            undistorted_frames_R = locals()['undistorted_frames_'+str(i+2)]

        sample_image_L = undistorted_frames_L[0]
        sample_image_R = undistorted_frames_R[0]

        height, width = sample_image_L.shape[:2]
        r_height, r_width = sample_image_R.shape[:2]

        roi_left = (2 * width // 3, 0, width // 3, height)
        roi_right = (0, 0, r_width // 3, r_height)             # Left third of the right image

        # Detect features
        keypoints_and_descriptors_L = detect_features_sift(undistorted_frames_L, roi_left)
        keypoints_and_descriptors_R = detect_features_sift(undistorted_frames_R, roi_right)
        undistorted_frame_L, kp_L, _ = keypoints_and_descriptors_L[0]
        undistorted_frame_R, kp_R, _ = keypoints_and_descriptors_R[0]

        matched_keypoints = match_features_sift(keypoints_and_descriptors_L, keypoints_and_descriptors_R)

        src_pts, dst_pts = matched_keypoints[0]
        matched_frame = np.hstack((undistorted_frame_L, undistorted_frame_R))

        homography = find_average_homography(undistorted_frames_L, undistorted_frames_R, matched_keypoints)
        panorama_frames = create_panorama(undistorted_frames_L, undistorted_frames_R, matched_keypoints, homography)

        panorama_generated = panorama_frames

    # del undistorted_frames_4
    del undistorted_frames_3
    del undistorted_frames_2
    del undistorted_frames_1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID' if you face issues with 'mp4v'
    frame_rate = 30  # Or the frame rate of your original video
    panorama_height, panorama_width = panorama_frames[0].shape[:2]  # Assuming all panoramas have the same shape
    out = cv2.VideoWriter('panorama_video.mp4', fourcc, 30, (panorama_width, panorama_height))

    # Write each frame to the video
    for panorama in panorama_frames:
        normalized_panorama = cv2.normalize(panorama, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        uint8_panorama = normalized_panorama.astype(np.uint8)

        # Write the normalized and converted frame to the video
        out.write(uint8_panorama)

    # Release the VideoWriter object
    out.release()
    print("The video was successfully saved as 'panorama_video.mp4'")
