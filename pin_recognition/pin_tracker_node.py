#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from linear_motor_msgs.srv import Mode


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def rect_upright_tilt_deg(rect) -> float:
    """
    minAreaRect の angle から「垂直からの傾き」を概算（0=垂直, 90=水平）。
    """
    (_, _), (w, h), angle = rect
    if w < h:
        w, h = h, w
        angle = angle + 90.0

    a = abs(angle)
    while a > 90.0:
        a -= 180.0
        a = abs(a)
    return abs(90.0 - a)


def tilt_deg_pca(xs: np.ndarray, ys: np.ndarray) -> float:
    """
    PCA主軸の向きから「垂直から何度傾いているか」を返す（0=垂直, 90=水平）。
    fitLineより安定しやすい。
    """
    pts = np.column_stack([xs, ys]).astype(np.float32)
    if pts.shape[0] < 2:
        return 90.0
    _, eigvec = cv2.PCACompute(pts, mean=None)
    vx, vy = float(eigvec[0, 0]), float(eigvec[0, 1])
    ang = abs(np.degrees(np.arctan2(vy, vx)))  # 0..180
    return abs(90.0 - ang)


def horiz_dev_deg_from_tilt(tilt_from_vertical: float) -> float:
    """
    tilt(0=垂直,90=水平) から「水平(90)からのズレ角」を返す。
    0deg=完全水平、90deg=完全垂直
    """
    return abs(90.0 - float(tilt_from_vertical))


class RedLineFollowerNode(Node):
    """
    赤線（テープ/線）を主ターゲットとして追跡する（横線版）。
    - 赤マスクから連結成分を取り、縦横比と傾きで「横の赤線」だけ採用
    - 採用成分の重心cxで旋回制御
    """

    def __init__(self):
        super().__init__("red_line_follower_node")

        # -------------------------
        # Params
        # -------------------------
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("show_windows", True)
        self.declare_parameter("publish_annotated", True)

        # debug
        self.declare_parameter("publish_red_mask", True)

        # red hsv
        self.declare_parameter("red_s_min", 120)
        self.declare_parameter("red_v_min", 90)

        # red morph
        self.declare_parameter("red_close_iters", 2)
        self.declare_parameter("red_open_iters", 0)

        # detection gate
        self.declare_parameter("min_red_pixels", 50)          # ノイズ除外
        self.declare_parameter("min_points", 60)

        # 線の細長さ（横でも縦でも「細長さ」なので共通）
        self.declare_parameter("aspect_min_far", 2.0)         # 遠いとき細長い
        self.declare_parameter("aspect_min_near", 1.2)        # 近いとき崩れるので緩め

        # ★横線ゲート：水平からの許容角（0=水平）
        self.declare_parameter("tilt_max_far", 15.0)          # 遠いとき厳しめ
        self.declare_parameter("tilt_max_near", 25.0)         # 近いとき緩め

        # near判定
        self.declare_parameter("near_major_ratio", 0.45)      # major >= 0.45*H なら近い扱い
        self.declare_parameter("near_area_ratio", 0.02)       # area >= 0.02*HW なら近い扱い

        # ★横長bbox条件（誤検出低減）
        self.declare_parameter("use_bbox_aspect_gate", True)
        self.declare_parameter("bbox_aspect_w_min", 1.2)      # w/h >= 1.2 を要求（横長）

        # control
        self.declare_parameter("angular_gain", 1.5)
        self.declare_parameter("max_angular_speed", 0.4)
        self.declare_parameter("linear_speed", 1.0)
        self.declare_parameter("search_yaw_rate", 0.3)

        # -------------------------
        # Read params
        # -------------------------
        self.image_topic = self.get_parameter("image_topic").value
        self.show_windows = bool(self.get_parameter("show_windows").value)
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)

        self.publish_red_mask = bool(self.get_parameter("publish_red_mask").value)

        self.red_s_min = int(self.get_parameter("red_s_min").value)
        self.red_v_min = int(self.get_parameter("red_v_min").value)
        self.red_close_iters = int(self.get_parameter("red_close_iters").value)
        self.red_open_iters = int(self.get_parameter("red_open_iters").value)

        self.min_red_pixels = int(self.get_parameter("min_red_pixels").value)
        self.min_points = int(self.get_parameter("min_points").value)

        self.aspect_min_far = float(self.get_parameter("aspect_min_far").value)
        self.aspect_min_near = float(self.get_parameter("aspect_min_near").value)
        self.tilt_max_far = float(self.get_parameter("tilt_max_far").value)
        self.tilt_max_near = float(self.get_parameter("tilt_max_near").value)

        self.near_major_ratio = float(self.get_parameter("near_major_ratio").value)
        self.near_area_ratio = float(self.get_parameter("near_area_ratio").value)

        self.use_bbox_aspect_gate = bool(self.get_parameter("use_bbox_aspect_gate").value)
        self.bbox_aspect_w_min = float(self.get_parameter("bbox_aspect_w_min").value)

        self.angular_gain = float(self.get_parameter("angular_gain").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.search_yaw_rate = float(self.get_parameter("search_yaw_rate").value)

        # -------------------------
        # ROS I/O
        # -------------------------
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, qos_profile_sensor_data
        )

        self.srv = self.create_service(Mode, "tracker_mode", self.srv_callback)
        self.mode = "OFF"

        self.bgr = None
        self.header = None

        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.annotated_pub = self.create_publisher(Image, "/red_line_follower/annotated", 1)
        self.detected_pub = self.create_publisher(Bool, "/red_line_follower/detected", 1)
        self.red_pub = self.create_publisher(Image, "/red_line_follower/red_mask", 1)

        self.get_logger().info(
            "red_line_follower_node(HORIZONTAL) started. "
            f"image_topic={self.image_topic} red_s_min={self.red_s_min} red_v_min={self.red_v_min} "
            f"min_red_pixels={self.min_red_pixels}"
        )

    # -------------------------
    # Red mask (HSV, wrap)
    # -------------------------
    def make_red_mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        smin = int(self.red_s_min)
        vmin = int(self.red_v_min)

        lower1 = np.array([0, smin, vmin], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, smin, vmin], dtype=np.uint8)
        upper2 = np.array([179, 255, 255], dtype=np.uint8)

        m1 = cv2.inRange(hsv, lower1, upper1)
        m2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(m1, m2)

        k3 = np.ones((3, 3), np.uint8)
        if self.red_open_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=self.red_open_iters)
        if self.red_close_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=self.red_close_iters)

        return mask

    # -------------------------
    # Detect horizontal red component
    # -------------------------
    def detect_horizontal_red(self, red_mask: np.ndarray, H: int, W: int):
        """
        赤マスクの連結成分から「横っぽい赤線」を1つ選ぶ。
        戻り値: (detected, cx, cy, bbox, dbg)
        """
        n, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
        if n <= 1:
            return False, None, None, None, "no_cc"

        best = None
        best_score = -1.0

        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_red_pixels:
                continue

            comp = (labels == i)
            ys, xs = np.where(comp)
            if ys.size < self.min_points:
                continue

            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            bbox = (x, y, w, h)

            # 横長bbox条件（任意だが強い）
            if self.use_bbox_aspect_gate:
                bbox_aspect_w = w / max(1, h)  # 横長ほど大きい
                if bbox_aspect_w < self.bbox_aspect_w_min:
                    continue

            # minAreaRectで細長さ & 傾き
            pts = np.column_stack([xs, ys]).astype(np.float32)
            rect = cv2.minAreaRect(pts)
            (_, _), (rw, rh), _ = rect
            major = float(max(rw, rh))
            minor = float(max(1.0, min(rw, rh)))
            aspect = major / minor

            # 0=垂直, 90=水平
            tilt_r_v = rect_upright_tilt_deg(rect)
            tilt_p_v = tilt_deg_pca(xs, ys)

            # 0=水平（水平からのズレ）
            dev_r_h = horiz_dev_deg_from_tilt(tilt_r_v)
            dev_p_h = horiz_dev_deg_from_tilt(tilt_p_v)

            # near判定
            is_near = (major >= self.near_major_ratio * H) or (area >= self.near_area_ratio * (H * W))
            aspect_min = self.aspect_min_near if is_near else self.aspect_min_far
            dev_max = self.tilt_max_near if is_near else self.tilt_max_far

            # 細長さ + 水平度でフィルタ
            if aspect < aspect_min:
                continue
            if dev_r_h > dev_max:
                continue
            if dev_p_h > dev_max:
                continue

            # スコア：大きい&細長いを優先
            score = area * aspect
            if score > best_score:
                best_score = score
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))
                best = (cx, cy, bbox, area, aspect, dev_r_h, dev_p_h, is_near)

        if best is None:
            return False, None, None, None, "no_horizontal_red"

        cx, cy, bbox, area, aspect, dev_r_h, dev_p_h, is_near = best
        dbg = f"ok area={area} asp={aspect:.2f} devH_R={dev_r_h:.1f} devH_P={dev_p_h:.1f} near={int(is_near)}"
        return True, cx, cy, bbox, dbg

    # -------------------------
    # ROS callback
    # -------------------------
    def srv_callback(self, request, response):
        if request.mode == "tracker_START":
            self.mode = "ON"
            self.get_logger().info("tracker_mode: ON")
        elif request.mode == "tracker_STOP":
            self.mode = "OFF"
            self.get_logger().info("tracker_mode: OFF")
        return response

    def image_callback(self, msg: Image):
        try:
            self.bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.header = msg.header
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")

    def timer_callback(self):
        if self.bgr is None or self.mode == "OFF":
            return

        H, W = self.bgr.shape[:2]

        # 1) 赤マスク
        red_mask = self.make_red_mask(self.bgr)

        # 2) 横赤検出
        detected, cx, cy, bbox, dbg = self.detect_horizontal_red(red_mask, H, W)

        # publish detected flag
        self.detected_pub.publish(Bool(data=bool(detected)))

        # publish red mask (optional)
        if self.publish_red_mask and self.header is not None:
            try:
                rm = self.bridge.cv2_to_imgmsg(red_mask, encoding="mono8")
                rm.header = self.header
                self.red_pub.publish(rm)
            except Exception:
                pass

        # annotated
        annotated = self.bgr.copy()
        cv2.line(annotated, (W // 2, 0), (W // 2, H), (255, 0, 0), 1)

        if detected and cx is not None and cy is not None:
            cv2.circle(annotated, (cx, cy), 8, (0, 255, 0), -1)
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(
                annotated,
                f"TRACK RED(H) {dbg} cx={cx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
            )
        else:
            cv2.putText(
                annotated,
                f"SEARCH RED(H) {dbg}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 180, 255),
                2,
            )

        if self.publish_annotated and self.header is not None:
            try:
                out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                out.header = self.header
                self.annotated_pub.publish(out)
            except Exception:
                pass

        if self.show_windows:
            try:
                cv2.imshow("red_line_follower_annotated", annotated)
                cv2.imshow("red_mask", red_mask)
                cv2.waitKey(1)
            except Exception:
                pass

        # -------------------------
        # control
        # -------------------------
        twist = Twist()
        if not detected or cx is None:
            twist.linear.x = 0.0
            twist.angular.z = float(self.search_yaw_rate)
        else:
            err = (float(cx) - (W / 2.0)) / (W / 2.0)  # -1..1
            wz = -self.angular_gain * err
            twist.angular.z = float(clamp(wz, -self.max_angular_speed, self.max_angular_speed))
            twist.linear.x = float(self.linear_speed)

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = RedLineFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
