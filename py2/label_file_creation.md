#### Label file creation

You could follow these steps to create labels for your custom dataset:

1. Get the 3D bounding box surrounding the 3D object model. We use the already provided 3D object model for the LINEMOD dataset to get the 3D bounding box. If you would like to create a 3D model for a custom object, you can refer to the Section 3.5 of the following paper and the references therein: http://cmp.felk.cvut.cz/~hodanto2/data/hodan2017tless.pdf

2. Define the 8 corners of the 3D bounding box and the centroid of the 3D object model as the virtual keypoints of the object. 8 corners correspond to the [[min_x, min_y, min_z], [min_x, min_y, max_z], [min_x, max_y, min_z], [min_x, max_y, max_z], [max_x, min_y, min_z], [max_x, min_y, max_z], [max_x, max_y, min_z], [max_x, max_y, max_z]] positions of the 3D object model, and the centroid corresponds to the [0, 0, 0] position.

3. Project the 3D keypoints to 2D. You can use the [compute_projection](https://github.com/Microsoft/singleshotpose/blob/master/utils.py#L39:L44) function that we provide to project the 3D points in 2D. You would need to know the intrinsic calibration matrix of the camera and the ground-truth rotation and translation to project the 3D points in 2D. Typically, obtaining ground-truth Rt transformation matrices requires a manual and intrusive annotation effort. For an example of how to acquire ground-truth data for 6D pose estimation, please refer to the Section 3.1 of the [paper](http://cmp.felk.cvut.cz/~hodanto2/data/hodan2017tless.pdf) describing the T-LESS dataset.

4. Compute the width and height of a 2D rectangle tightly fitted to a masked region around the object. If you have the 2D bounding box information (e.g. width and height) for the custom object that you have, you can use those values in your label file. In practice, however, we fit a tight bounding box to the 8 corners of the projected 3D bounding box and use the width and height of that bounding box to represent these values.

5. Create an array consisting of the class, 2D keypoint location and the range information and write it into a text file. The label file is organized in the following order. 1st number: class label, 2nd number: x0 (x-coordinate of the centroid), 3rd number: y0 (y-coordinate of the centroid), 4th number: x1 (x-coordinate of the first corner), 5th number: y1 (y-coordinate of the first corner), ..., 18th number: x8 (x-coordinate of the eighth corner), 19th number: y8 (y-coordinate of the eighth corner), 20th number: x range, 21st number: y range.
