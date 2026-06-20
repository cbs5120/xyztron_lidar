#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#=============================================
# 본 프로그램은 자이트론에서 제작한 것입니다.
# 상업라이센스에 의해 제공되므로 무단배포 및 상업적 이용을 금합니다.
# 교육과 실습 용도로만 사용가능하며 외부유출은 금지됩니다.
#=============================================

import rclpy
import time
import cv2
import math
import numpy as np

from rclpy.node import Node
from xycar_msgs.msg import XycarMotor
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge


#=============================================
# ROS2 Node 클래스 정의
#=============================================
class TrackDriverNode(Node):

    #=============================================
    # 클래스 생성 초기화 함수
    #=============================================
    def __init__(self):

        super().__init__('driver')
        self.get_logger().info('----- Xycar self-driving node started -----')

        # 센서 데이터 저장 변수
        self.image = None
        self.lidar_ranges = None
        self.angle_min = None
        self.angle_increment = None

        # 모터 메시지
        self.motor_msg = XycarMotor()
        self.bridge = CvBridge()

        # 조향 안정화 변수
        self.prev_angle = 0.0

        # 로그 출력 횟수 조절용
        self.log_count = 0

        #=============================================
        # 기본 주행 파라미터
        #=============================================

        # 속도가 빠르면 꼬깔 앞에서 조향이 늦어져 충돌하므로 낮춤
        self.base_speed = 8.0

        # 더 크게 회전하도록 최대 조향각 증가
        self.max_angle = 80.0

        # 조향 방향이 반대로 움직이면 -1.0으로 바꾸세요.
        self.steer_sign = 1.0

        # 라바콘 사이 예상 반폭
        # 차가 왼쪽 꼬깔에 너무 붙으면 값을 키우고,
        # 오른쪽 꼬깔에 너무 붙으면 값을 줄이세요.
        self.lane_half_width = 0.75

        #=============================================
        # 가까운 꼬깔 회피 파라미터
        #=============================================

        # 꼬깔을 더 일찍 감지해서 회피하도록 거리 증가
        self.cone_stop_distance = 0.65

        # 정면 장애물 감지 거리 증가
        self.front_stop_distance = 0.35

        # 후진 설정
        self.reverse_speed = -10.0
        self.reverse_duration = 0.35

        # 회피 전진 설정
        self.escape_speed = 6.0
        self.escape_angle = 55.0
        self.escape_duration = 0.75

        # 회피 직후 바로 다시 회피하지 않도록 대기 시간
        self.recovery_cooldown = 0.60

        # 회피 상태 변수
        self.recovery_state = "NORMAL"   # NORMAL, BACKUP, ESCAPE
        self.recovery_until = 0.0
        self.cooldown_until = 0.0

        # 가까운 꼬깔 위치
        # 1  = 왼쪽 꼬깔이 가까움 → 오른쪽으로 회피
        # -1 = 오른쪽 꼬깔이 가까움 → 왼쪽으로 회피
        self.recovery_side = 1

        #=============================================
        # ROS2 Publisher & Subscriber 설정
        #=============================================
        self.motor_pub = self.create_publisher(
            XycarMotor,
            'xycar_motor',
            10
        )

        self.sub_front = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',
            self.cam_callback,
            qos_profile_sensor_data
        )

        self.sub_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info("Track Driver Node Initialized")

    #=============================================
    # 카메라 콜백 함수
    #=============================================
    def cam_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    #=============================================
    # 라이다 콜백 함수
    #=============================================
    def lidar_callback(self, msg):
        self.lidar_ranges = list(msg.ranges)
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment

    #=============================================
    # 모터제어 토픽 발행 함수
    #=============================================
    def drive(self, angle, speed):
        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = float(speed)
        self.motor_pub.publish(self.motor_msg)

    #=============================================
    # 값 제한 함수
    #=============================================
    def clamp(self, value, min_value, max_value):
        return max(min_value, min(max_value, value))

    #=============================================
    # 라이다 거리값 유효성 검사
    #=============================================
    def is_valid_range(self, r):
        if r is None:
            return False
        if math.isnan(r) or math.isinf(r):
            return False
        if r < 0.10:
            return False
        if r > 6.0:
            return False
        return True

    #=============================================
    # LaserScan 데이터를 차량 기준 x-y 좌표로 변환
    # x: 차량 전방
    # y: 차량 왼쪽
    #=============================================
    def get_lidar_points(self):

        points = []

        if self.lidar_ranges is None:
            return points

        if self.angle_min is None or self.angle_increment is None:
            return points

        for i, r in enumerate(self.lidar_ranges):

            if not self.is_valid_range(r):
                continue

            theta = self.angle_min + i * self.angle_increment

            x = r * math.cos(theta)
            y = r * math.sin(theta)

            points.append((x, y, r, math.degrees(theta)))

        return points

    #=============================================
    # 라이다 점 중에서 라바콘 후보 찾기
    #=============================================
    def find_cone_points(self):

        points = self.get_lidar_points()

        left_points = []
        right_points = []

        for x, y, r, deg in points:

            # 차량 앞쪽만 사용
            if x < 0.10 or x > 5.0:
                continue

            # 너무 바깥쪽은 나무, 벽, 배경일 가능성이 높음
            if abs(y) > 2.5:
                continue

            # 너무 중앙은 노이즈일 수 있어 제외
            if abs(y) < 0.03:
                continue

            if y > 0:
                left_points.append((x, y, r, deg))
            else:
                right_points.append((x, y, r, deg))

        return left_points, right_points

    #=============================================
    # 후보 점들 중 가까운 점들의 대표 좌표 계산
    #=============================================
    def representative_point(self, points):

        if len(points) == 0:
            return None

        # 앞쪽에 가까운 후보 우선
        points = sorted(points, key=lambda p: p[0])

        # 가까운 후보 여러 개 사용
        selected = points[:10]

        xs = [p[0] for p in selected]
        ys = [p[1] for p in selected]

        return float(np.median(xs)), float(np.median(ys))

    #=============================================
    # 정면 장애물 최소 거리 확인
    #=============================================
    def front_min_distance(self):

        points = self.get_lidar_points()

        front = []

        for x, y, r, deg in points:

            # 기존보다 넓게 봄
            # 정면 가까운 꼬깔을 더 빨리 감지
            if 0.0 < x < 0.80 and abs(y) < 0.30:
                front.append(r)

        if len(front) == 0:
            return 999.0

        return min(front)

    #=============================================
    # 차량이 꼬깔과 너무 가까운지 확인
    # 반환값:
    #   near_dist : 가장 가까운 꼬깔 거리
    #   near_side : 1이면 왼쪽, -1이면 오른쪽, 0이면 모름
    #=============================================
    def near_cone_info(self):

        points = self.get_lidar_points()

        near_dist = 999.0
        near_side = 0

        for x, y, r, deg in points:

            # 기존보다 앞쪽과 좌우 범위를 넓게 봄
            # 꼬깔에 닿기 전에 미리 회피
            if 0.0 < x < 1.40 and 0.08 < abs(y) < 1.30:

                if r < near_dist:
                    near_dist = r

                    if y > 0:
                        near_side = 1
                    else:
                        near_side = -1

        return near_dist, near_side

    #=============================================
    # 회피 방향 추정
    # 가까운 꼬깔 위치가 불명확할 때 사용
    #=============================================
    def estimate_recovery_side(self):

        # 현재 왼쪽으로 꺾고 있었다면 왼쪽이 위험할 가능성이 큼
        if self.prev_angle > 2.0:
            return 1

        # 현재 오른쪽으로 꺾고 있었다면 오른쪽이 위험할 가능성이 큼
        if self.prev_angle < -2.0:
            return -1

        # 모르면 기본적으로 왼쪽 꼬깔을 피한다고 가정
        return 1

    #=============================================
    # 회피 시작
    #=============================================
    def start_recovery(self, side, reason):

        if side == 0:
            side = self.estimate_recovery_side()

        self.recovery_side = side
        self.recovery_state = "BACKUP"
        self.recovery_until = time.time() + self.reverse_duration

        side_text = "LEFT" if side == 1 else "RIGHT"

        self.get_logger().warn(
            f"Start recovery. reason={reason}, near_side={side_text}"
        )

    #=============================================
    # 회피 상태 처리
    # BACKUP: 잠깐 후진
    # ESCAPE: 가까운 꼬깔 반대 방향으로 빠져나가기
    #=============================================
    def recovery_drive_control(self):

        now = time.time()

        # 후진 단계
        if self.recovery_state == "BACKUP":

            if now < self.recovery_until:

                if self.log_count % 8 == 0:
                    self.get_logger().warn("Recovery BACKUP")

                # 후진은 직선으로 짧게 수행
                return 0.0, self.reverse_speed

            # 후진이 끝나면 회피 전진 단계로 전환
            self.recovery_state = "ESCAPE"
            self.recovery_until = now + self.escape_duration

        # 회피 전진 단계
        if self.recovery_state == "ESCAPE":

            if now < self.recovery_until:

                # recovery_side == 1  : 왼쪽 꼬깔이 가까움 → 오른쪽으로 회피
                # recovery_side == -1 : 오른쪽 꼬깔이 가까움 → 왼쪽으로 회피
                escape_angle = -self.recovery_side * self.escape_angle
                escape_angle = escape_angle * self.steer_sign

                if self.log_count % 8 == 0:
                    self.get_logger().warn(
                        f"Recovery ESCAPE angle={escape_angle:.1f}"
                    )

                return escape_angle, self.escape_speed

            # 회피 완료
            self.recovery_state = "NORMAL"
            self.cooldown_until = now + self.recovery_cooldown

            self.get_logger().info("Recovery finished. Return to normal driving.")

        return None

    #=============================================
    # 라이다 기반 라바콘 사이 주행 제어
    #=============================================
    def cone_drive_control(self):

        self.log_count += 1
        now = time.time()

        #=============================================
        # 회피 동작 중이면 회피 명령 우선 수행
        #=============================================
        recovery_cmd = self.recovery_drive_control()

        if recovery_cmd is not None:
            return recovery_cmd

        #=============================================
        # 가까운 꼬깔 또는 정면 장애물 감지
        #=============================================
        near_dist, near_side = self.near_cone_info()
        front_dist = self.front_min_distance()

        # 회피 직후에는 바로 다시 회피하지 않도록 잠깐 무시
        if now >= self.cooldown_until:

            if near_dist < self.cone_stop_distance:
                self.start_recovery(
                    near_side,
                    f"cone too close: {near_dist:.2f}m"
                )
                return 0.0, self.reverse_speed

            if front_dist < self.front_stop_distance:
                self.start_recovery(
                    near_side,
                    f"front too close: {front_dist:.2f}m"
                )
                return 0.0, self.reverse_speed

        #=============================================
        # 일반 라바콘 주행
        #=============================================
        left_points, right_points = self.find_cone_points()

        left_rep = self.representative_point(left_points)
        right_rep = self.representative_point(right_points)

        left_count = len(left_points)
        right_count = len(right_points)

        target_x = 1.5
        target_y = 0.0
        speed = self.base_speed
        mode = "NO_CONE"

        #=============================================
        # 양쪽 라바콘이 모두 보이는 경우
        #=============================================
        if left_rep is not None and right_rep is not None:

            left_x, left_y = left_rep
            right_x, right_y = right_rep

            target_x = (left_x + right_x) / 2.0
            target_y = (left_y + right_y) / 2.0

            mode = "BOTH"

        #=============================================
        # 왼쪽 라바콘만 보이는 경우
        #=============================================
        elif left_rep is not None:

            left_x, left_y = left_rep

            target_x = left_x
            target_y = left_y - self.lane_half_width

            mode = "LEFT_ONLY"

        #=============================================
        # 오른쪽 라바콘만 보이는 경우
        #=============================================
        elif right_rep is not None:

            right_x, right_y = right_rep

            target_x = right_x
            target_y = right_y + self.lane_half_width

            mode = "RIGHT_ONLY"

        #=============================================
        # 아무 라바콘도 못 찾은 경우
        #=============================================
        else:
            target_x = 1.5
            target_y = 0.0
            speed = 4.0
            mode = "NO_CONE"

        #=============================================
        # 목표점 방향으로 조향각 계산
        #=============================================
        raw_angle = math.degrees(math.atan2(target_y, max(target_x, 0.1)))

        # 조향 게인 증가
        # 기존 2.5보다 강하게 꺾도록 5.0 사용
        raw_angle = raw_angle * 5.0 * self.steer_sign

        # 조향 변화 완화
        # alpha를 낮춰 조향 반응을 빠르게 함
        alpha = 0.20
        angle = alpha * self.prev_angle + (1.0 - alpha) * raw_angle

        angle = self.clamp(angle, -self.max_angle, self.max_angle)
        self.prev_angle = angle

        #=============================================
        # 조향이 크면 감속
        #=============================================
        if abs(angle) > 45:
            speed = 4.0
        elif abs(angle) > 25:
            speed = 6.0
        else:
            speed = self.base_speed

        # 라바콘을 못 찾으면 천천히 직진
        if mode == "NO_CONE":
            speed = 4.0

        #=============================================
        # 로그는 10번에 1번만 출력
        #=============================================
        if self.log_count % 10 == 0:
            self.get_logger().info(
                f"mode={mode}, "
                f"L={left_count}, R={right_count}, "
                f"target_x={target_x:.2f}, target_y={target_y:.2f}, "
                f"angle={angle:.1f}, speed={speed:.1f}, "
                f"front={front_dist:.2f}, near={near_dist:.2f}"
            )

        return angle, speed

    #=============================================
    # 메인 루프
    #=============================================
    def main_loop(self):

        self.get_logger().info("======================================")
        self.get_logger().info("  L I D A R   C O N E   D R I V E     ")
        self.get_logger().info("======================================")

        # 라이다 데이터가 들어올 때까지 대기
        while rclpy.ok() and self.lidar_ranges is None:
            self.get_logger().info("Waiting for lidar data...")
            rclpy.spin_once(self, timeout_sec=0.1)
            self.drive(angle=0, speed=0)

        # 빠른 라이다 판단 루프
        while rclpy.ok():

            # 콜백을 자주 처리
            rclpy.spin_once(self, timeout_sec=0.001)

            # 라이다 기반 조향 계산
            angle, speed = self.cone_drive_control()

            # 모터 명령 발행
            self.drive(angle=angle, speed=speed)

            # 0.02초마다 판단 = 약 50Hz
            time.sleep(0.02)


#=============================================
# 메인 함수
#=============================================
def main(args=None):

    rclpy.init(args=args)
    node = TrackDriverNode()

    try:
        node.main_loop()

    except KeyboardInterrupt:
        pass

    finally:
        node.drive(angle=0, speed=0)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
