import cv2
assert cv2.__version__[0] == '4', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob
from PIL import Image
def interp2(v, xq, yq):
	dim_input = 1
	if len(xq.shape) == 2 or len(yq.shape) == 2:
		dim_input = 2
		q_h = xq.shape[0]
		q_w = xq.shape[1]
		xq = xq.flatten()
		yq = yq.flatten()

	h = v.shape[0]
	w = v.shape[1]
	if xq.shape != yq.shape:
		raise('query coordinates Xq Yq should have same shape')

	x_floor = np.floor(xq).astype(np.int32)
	y_floor = np.floor(yq).astype(np.int32)
	x_ceil = np.ceil(xq).astype(np.int32)
	y_ceil = np.ceil(yq).astype(np.int32)

	x_floor[x_floor < 0] = 0
	y_floor[y_floor < 0] = 0
	x_ceil[x_ceil < 0] = 0
	y_ceil[y_ceil < 0] = 0

	x_floor[x_floor >= w-1] = w-1
	y_floor[y_floor >= h-1] = h-1
	x_ceil[x_ceil >= w-1] = w-1
	y_ceil[y_ceil >= h-1] = h-1

	v1 = v[y_floor, x_floor]
	v2 = v[y_floor, x_ceil]
	v3 = v[y_ceil, x_floor]
	v4 = v[y_ceil, x_ceil]

	lh = yq - y_floor
	lw = xq - x_floor
	hh = 1 - lh
	hw = 1 - lw

	w1 = hh * hw
	w2 = hh * lw
	w3 = lh * hw
	w4 = lh * lw

	interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

	if dim_input == 2:
		return interp_val.reshape(q_h, q_w)
	return interp_val

def barrelBackProjection(image, size_W, size_H, map):

	generated_pic = np.zeros((size_H, size_W, 3), dtype=np.uint8)
	for i in range(3):
		generated_pic[:, :, i] = interp2(image[:,:,i], map[0], map[1]).reshape(size_H,size_W)
  
	return generated_pic


def evaluatePixel_Front(srcSize):
	xx, yy = np.meshgrid(np.arange(0,srcSize[0]), np.arange(0,srcSize[1]))
	xx = xx.flatten()
	yy = yy.flatten()

	# convert outcoords to radians (180 = pi, so half a sphere)
	o_x = 1.0 * xx / srcSize[0]
	o_y = 1.0 * yy / srcSize[1]
	theta = ((59.0 / 180 + 0.5) - o_x * 118.0 / 180)* np.pi
	phi = (o_y  * 69.0 / 180 + 0.5 - 34.5 / 180 ) * np.pi

	# Convert outcoords to spherical (x,y,z on unisphere)
	x_sph = np.cos(theta) * np.sin(phi)
	y_sph = np.sin(theta) * np.sin(phi)
	z_sph = np.cos(phi)

	# Convert spherical to input coordinates...
	theta2 = np.arctan2(-z_sph, x_sph)
	phi2_over_pi = np.arccos(y_sph) / (np.pi * 133/ 180)

	x_incentered = (18.36 / 16.0   *(phi2_over_pi * np.cos(theta2)) + 0.5) * srcSize[0]
	y_incentered = (18.36/ 9.0  *(phi2_over_pi * np.sin(theta2)) + 0.5) * srcSize[1]

	# x_incentered = ((phi2_over_pi * np.cos(theta2)) + 0.5) * srcSize[0]
	# y_incentered = ((phi2_over_pi * np.sin(theta2)) + 0.5) * srcSize[1]

	return np.array([x_incentered,y_incentered])


# You should replace these 3 lines with the output in calibration step
DIM=(1920, 1080)
K=np.array([[854.2478846266547, 0.0, 960.5956653046294], [0.0, 639.2599759908496, 539.6973814104975], [0.0, 0.0, 1.0]])
D=np.array([[0.04346967920883741], [0.038850164968710264], [-0.03369272932737338], [0.009408350218009107]])
def undistort(img_path,dst_path,balance=0.0, dim2=None, dim3=None):
	mapping = evaluatePixel_Front(np.array([1920,1080]))
	for fname in os.listdir(img_path):
		path = os.path.join(img_path, fname)
		img = cv2.imread(path)
		img_resized = cv2.resize(img, (1920, 1080))
		dim1 = img_resized.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
		assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
		if not dim2:
			dim2 = dim1
		if not dim3:
			dim3 = dim1
		scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
		scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
		new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
		map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
		undistorted_img = cv2.remap(img_resized, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
		
		
		barrel_img = barrelBackProjection(img_resized, 1920, 1080, mapping)
		cv2.imwrite(os.path.join(dst_path, "undistorted_" + fname), undistorted_img)

		
		# undistored_barr = cv2.undistort(barrel_img,K,D)
		cv2.imwrite(os.path.join(dst_path, "barrel_" + fname), barrel_img)
		print("finish1")

# image_folder = "Image"
# dst_folder = "Undistort"
# undistort(image_folder,dst_folder,balance=1.0)

