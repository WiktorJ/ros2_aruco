"""
This node locates Aruco or Charuco boards in images and publishes their poses.

Subscriptions:
   /camera/image_raw (sensor_msgs.msg.Image)
   /camera/camera_info (sensor_msgs.msg.CameraInfo)

Published Topics:
    /aruco_board_pose (geometry_msgs.msg.PoseStamped)
       Pose of the detected board.
    /aruco_board_image (sensor_msgs.msg.Image)
       Image with detected board and pose. Published when publish_board_image is true.

Parameters:
    board_type - type of board to detect, 'aruco' or 'charuco' (default 'aruco')
    aruco_dictionary_id - dictionary that was used to generate markers
                          (default DICT_5X5_250)
    image_topic - image topic to subscribe to (default /camera/image_raw)
    camera_info_topic - camera info topic to subscribe to
                         (default /camera/camera_info)
    camera_frame - camera optical frame to use (default from camera_info)
    publish_board_image - whether to publish the image with detected markers (default false)

    -- Aruco Board Parameters --
    markers_x - number of markers in X direction (default 5)
    markers_y - number of markers in Y direction (default 7)
    marker_length - marker side length in meters (default 0.04)
    marker_separation - separation between markers in meters (default 0.01)

    -- Charuco Board Parameters --
    squares_x - number of squares in X direction (default 5)
    squares_y - number of squares in Y direction (default 7)
    square_length - charuco square side length in meters (default 0.04)
    marker_length - marker side length in meters for charuco board.

"""

import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import numpy as np
import cv2
import tf_transformations
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import ParameterDescriptor, ParameterType


class ArucoBoardNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("aruco_board_node")

        # Declare and read parameters
        self.declare_parameter(
            name="board_type",
            value="aruco",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Type of board to detect, 'aruco' or 'charuco'.",
            ),
        )
        self.declare_parameter(
            name="aruco_dictionary_id",
            value="DICT_5X5_250",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Dictionary that was used to generate markers.",
            ),
        )
        self.declare_parameter(
            name="image_topic",
            value="/camera/image_raw",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Image topic to subscribe to.",
            ),
        )
        self.declare_parameter(
            name="camera_info_topic",
            value="/camera/camera_info",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera info topic to subscribe to.",
            ),
        )
        self.declare_parameter(
            name="camera_frame",
            value="",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera optical frame to use.",
            ),
        )
        self.declare_parameter(
            name="publish_board_image",
            value=False,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="Publish the board image with markers and pose detected.",
            ),
        )

        # Aruco board parameters
        self.declare_parameter(
            name="markers_x",
            value=5,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="Number of markers in X direction for Aruco board.",
            ),
        )
        self.declare_parameter(
            name="markers_y",
            value=7,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="Number of markers in Y direction for Aruco board.",
            ),
        )
        self.declare_parameter(
            name="marker_length",
            value=0.04,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Marker side length in meters.",
            ),
        )
        self.declare_parameter(
            name="marker_separation",
            value=0.01,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Separation between markers in meters.",
            ),
        )

        # Charuco board parameters
        self.declare_parameter(
            name="squares_x",
            value=5,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="Number of squares in X direction for Charuco board.",
            ),
        )
        self.declare_parameter(
            name="squares_y",
            value=7,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="Number of squares in Y direction for Charuco board.",
            ),
        )
        self.declare_parameter(
            name="square_length",
            value=0.04,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Charuco square side length in meters.",
            ),
        )

        # Get parameters
        self.board_type = (
            self.get_parameter("board_type").get_parameter_value().string_value
        )
        dictionary_id_name = (
            self.get_parameter("aruco_dictionary_id")
            .get_parameter_value()
            .string_value
        )
        image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        self.camera_frame = (
            self.get_parameter("camera_frame").get_parameter_value().string_value
        )
        self.publish_board_image = (
            self.get_parameter("publish_board_image").get_parameter_value().bool_value
        )

        self.get_logger().info(f"Board type: {self.board_type}")
        self.get_logger().info(f"Marker type: {dictionary_id_name}")
        self.get_logger().info(f"Image topic: {image_topic}")
        self.get_logger().info(f"Image info topic: {info_topic}")
        self.get_logger().info(f"Publish board image: {self.publish_board_image}")

        # Make sure we have a valid dictionary id:
        try:
            dictionary_id = cv2.aruco.__getattribute__(dictionary_id_name)
            if type(dictionary_id) != type(cv2.aruco.DICT_5X5_100):
                raise AttributeError
        except AttributeError:
            self.get_logger().error(f"bad aruco_dictionary_id: {dictionary_id_name}")
            options = "\n".join([s for s in dir(cv2.aruco) if s.startswith("DICT")])
            self.get_logger().error(f"valid options: {options}")
            raise AttributeError

        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters()

        self.aruco_board = None
        self.charuco_board = None
        self.aruco_detector = None
        self.charuco_detector = None

        # Create board
        if self.board_type == "aruco":
            markers_x = (
                self.get_parameter("markers_x").get_parameter_value().integer_value
            )
            markers_y = (
                self.get_parameter("markers_y").get_parameter_value().integer_value
            )
            marker_length = (
                self.get_parameter("marker_length").get_parameter_value().double_value
            )
            marker_separation = (
                self.get_parameter("marker_separation")
                .get_parameter_value()
                .double_value
            )
            self.get_logger().info(
                f"Aruco board: {markers_x}x{markers_y} markers, "
                f"length={marker_length}, separation={marker_separation}"
            )
            self.aruco_board = cv2.aruco.GridBoard(
                size=(markers_x, markers_y),
                markerLength=marker_length,
                markerSeparation=marker_separation,
                dictionary=self.aruco_dictionary,
            )
            self.aruco_detector = cv2.aruco.ArucoDetector(
                dictionary=self.aruco_dictionary, detectorParams=self.aruco_parameters
            )
        elif self.board_type == "charuco":
            squares_x = (
                self.get_parameter("squares_x").get_parameter_value().integer_value
            )
            squares_y = (
                self.get_parameter("squares_y").get_parameter_value().integer_value
            )
            square_length = (
                self.get_parameter("square_length").get_parameter_value().double_value
            )
            marker_length = (
                self.get_parameter("marker_length").get_parameter_value().double_value
            )
            self.get_logger().info(
                f"Charuco board: {squares_x}x{squares_y} squares, "
                f"square_length={square_length}, marker_length={marker_length}"
            )
            self.charuco_board = cv2.aruco.CharucoBoard(
                size=(squares_x, squares_y),
                squareLength=square_length,
                markerLength=marker_length,
                dictionary=self.aruco_dictionary,
            )
            self.charuco_detector = cv2.aruco.CharucoDetector(
                board=self.charuco_board, detectorParams=self.aruco_parameters
            )
        else:
            self.get_logger().error(f"Unknown board type: {self.board_type}")
            raise ValueError("board_type must be 'aruco' or 'charuco'")

        # Set up subscriptions
        self.info_sub = self.create_subscription(
            CameraInfo, info_topic, self.info_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, image_topic, self.image_callback, qos_profile_sensor_data
        )

        # Set up publisher
        self.pose_pub = self.create_publisher(PoseStamped, "aruco_board_pose", 10)
        if self.publish_board_image:
            self.board_image_pub = self.create_publisher(Image, "aruco_board_image", 10)

        # Set up fields for camera parameters
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

        self.bridge = CvBridge()

    def info_callback(self, info_msg):
        self.info_msg = info_msg
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))
        self.distortion = np.array(self.info_msg.d)
        # Assume that camera parameters will remain the same...
        self.destroy_subscription(self.info_sub)

    def image_callback(self, img_msg):
        if self.info_msg is None:
            self.get_logger().warn("No camera info has been received!")
            return

        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")

        output_image = None
        if self.publish_board_image:
            output_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

        rvec = None
        tvec = None
        success = False

        if self.board_type == "aruco":
            corners, marker_ids, rejected = self.aruco_detector.detectMarkers(cv_image)

            if self.publish_board_image:
                cv2.aruco.drawDetectedMarkers(output_image, corners, marker_ids)

            if marker_ids is not None:
                obj_points, img_points = self.aruco_board.matchImagePoints(
                    corners, marker_ids
                )
                if (
                    obj_points is not None
                    and img_points is not None
                    and len(obj_points) >= 4
                ):
                    success, rvec, tvec = cv2.solvePnP(
                        objectPoints=obj_points,
                        imagePoints=img_points,
                        cameraMatrix=self.intrinsic_mat,
                        distCoeffs=self.distortion,
                    )

        elif self.board_type == "charuco":
            (
                charuco_corners,
                charuco_ids,
                marker_corners,
                marker_ids,
            ) = self.charuco_detector.detectBoard(cv_image)

            if self.publish_board_image:
                if marker_ids is not None and len(marker_ids) > 0:
                    cv2.aruco.drawDetectedMarkers(
                        output_image, marker_corners, marker_ids
                    )
                if charuco_ids is not None and len(charuco_ids) > 0:
                    cv2.aruco.drawDetectedCornersCharuco(
                        output_image, charuco_corners, charuco_ids
                    )

            if charuco_ids is not None and len(charuco_ids) >= 4:
                obj_points = self.charuco_board.getChessboardCorners()[
                    charuco_ids.flatten()
                ]
                success, rvec, tvec = cv2.solvePnP(
                    objectPoints=obj_points,
                    imagePoints=charuco_corners,
                    cameraMatrix=self.intrinsic_mat,
                    distCoeffs=self.distortion,
                )

        if success:
            pose_stamped = PoseStamped()
            if self.camera_frame == "":
                pose_stamped.header.frame_id = self.info_msg.header.frame_id
            else:
                pose_stamped.header.frame_id = self.camera_frame
            pose_stamped.header.stamp = img_msg.header.stamp

            pose_stamped.pose.position.x = tvec[0][0]
            pose_stamped.pose.position.y = tvec[1][0]
            pose_stamped.pose.position.z = tvec[2][0]

            rot_matrix = np.eye(4)
            rot_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvec))[0]
            quat = tf_transformations.quaternion_from_matrix(rot_matrix)

            pose_stamped.pose.orientation.x = quat[0]
            pose_stamped.pose.orientation.y = quat[1]
            pose_stamped.pose.orientation.z = quat[2]
            pose_stamped.pose.orientation.w = quat[3]

            self.pose_pub.publish(pose_stamped)

            if self.publish_board_image:
                axis_length = 0.0
                if self.board_type == "aruco":
                    marker_length = (
                        self.get_parameter("marker_length")
                        .get_parameter_value()
                        .double_value
                    )
                    axis_length = marker_length * 2.0
                elif self.board_type == "charuco":
                    square_length = (
                        self.get_parameter("square_length")
                        .get_parameter_value()
                        .double_value
                    )
                    axis_length = square_length * 2.0
                cv2.drawFrameAxes(
                    output_image,
                    self.intrinsic_mat,
                    self.distortion,
                    rvec,
                    tvec,
                    axis_length,
                )

        if self.publish_board_image:
            img_msg_out = self.bridge.cv2_to_imgmsg(output_image, encoding="bgr8")
            img_msg_out.header = img_msg.header
            self.board_image_pub.publish(img_msg_out)


def main():
    rclpy.init()
    node = ArucoBoardNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
