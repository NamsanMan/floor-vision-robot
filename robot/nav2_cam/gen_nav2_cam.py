#!/usr/bin/env python3
"""Nav2 카메라 연결용 사본 만들기 (로봇 컨테이너에서 실행, 2026-10-29). 공장 파일은 읽기만 하고 고치지 않는다.
만드는 것 (~/shared/nav2_cam/):
  nav2_params_cam.yaml          공장 nav2_params.yaml + 지역 코스트맵 camera_layer (기억 점은 표시만, 지우기 끝점은 지우기만)
                                + velocity_smoother 최고 속도 TEST_MAX_VEL (2026-10-30 주행 시험용)
  nav2_controller_teb_cam.yaml  공장 nav2_controller_teb.yaml + TEB max_vel_x·y = TEST_MAX_VEL (2026-10-30 주행 시험용)
  nav_cam.launch.py             공장 navigation.launch.py 사본: 아래 두 사본과 위 설정을 쓰게
  bringup_cam.launch.py         공장 bringup.launch.py 사본: navigation_base 대신 아래 사본을 포함
  navigation_base_cam.launch.py 공장 사본 + velocity_smoother 출력을 cmd_vel_smoothed로 + collision_monitor(컴포저블, lifecycle 포함)
collision_monitor_cam.yaml 은 같이 복사해 둔 파일을 그대로 쓴다.
실행: ros2 launch /home/ubuntu/shared/nav2_cam/nav_cam.launch.py map:=room_v2   (zsh 환경변수 필요 → nav_cam.sh 사용)"""
import os
import re

SRC = "/home/ubuntu/ros2_ws/src"
OUT = "/home/ubuntu/shared/nav2_cam"
TEST_MAX_VEL = 0.25          # 2026-10-30 주행 시험용 최고 속도(m/s). 공장값: TEB 0.4, velocity_smoother 0.35
os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)


def rd(p):
    return open(p, encoding="utf-8").read()


def wr(name, s):
    open(os.path.join(OUT, name), "w", encoding="utf-8").write(s)
    print("wrote", name)


def rep(s, a, b, n=1):
    assert s.count(a) == n, f"expected {n}x, found {s.count(a)}: {a[:70]!r}"
    return s.replace(a, b)


def rep_re(s, pat, b, n=1):
    out, k = re.subn(pat, b, s, flags=re.M)
    assert k == n, f"expected {n}x, found {k}: {pat!r}"
    return out


# 1) 설정: 지역 코스트맵 구간에만 camera_layer 추가
p = rd(f"{SRC}/navigation/config/nav2_params.yaml")
i0, i1 = p.index("\nlocal_costmap:"), p.index("\nglobal_costmap:")
loc = p[i0:i1]
loc = rep(loc, 'plugins: ["voxel_layer", "inflation_layer"]', 'plugins: ["voxel_layer", "camera_layer", "inflation_layer"]')
CAM = """      camera_layer:            # 2026-10-29 바닥 분할 노드의 장애물 점. LiDAR 층과 따로 둠(LiDAR 빔이 낮은 장애물 위를 지나며 지우지 않게)
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        footprint_clearing_enabled: False      # 사각지대로 들어간 장애물을 로봇 외곽이 지우지 않게
        combination_method: 1
        observation_sources: floor_memory floor_clearing
        floor_memory:                          # 표시만: 기억(odom) = 지금 보이는 장애물 + 사각지대·화면 밖에 남긴 점
          topic: /floor_obstacles/memory
          data_type: "PointCloud2"
          sensor_frame: ascamera_color_0       # 2026-10-31: 없으면 거리 제한(obstacle_max_range)을 점구름 좌표계(odom)의 원점에서 잼 →
                                               # 로봇이 odom 원점에서 멀어지면(손으로 되돌려도 odom은 그대로) 장애물이 코스트맵에서 빠져
                                               # TEB는 밀어붙이고 Collision Monitor는 막는 교착(주행 시험 8·11회차)
          marking: True
          clearing: False
          min_obstacle_height: -0.05
          max_obstacle_height: 0.30
          obstacle_max_range: 2.0
          obstacle_min_range: 0.0
          expected_update_rate: 1.0
          inf_is_valid: False
        floor_clearing:                        # 지우기만: 카메라에서 열마다 끝점까지 광선. 가장 가까운 보이는 바닥(카메라에서 약 0.23m)보다 가까운 사각지대는 안 지움
          topic: /floor_obstacles/clearing
          data_type: "PointCloud2"
          sensor_frame: ascamera_color_0
          marking: False
          clearing: True
          min_obstacle_height: -0.05
          max_obstacle_height: 0.30
          raytrace_min_range: 0.23
          raytrace_max_range: 1.7
          expected_update_rate: 1.0
          inf_is_valid: False
"""
loc = rep(loc, "      inflation_layer:\n", CAM + "      inflation_layer:\n")
# 2026-10-29 조용한 실패 대책(CPU 여유): 아무도 안 쓰는 3D 복셀 지도 발행 끄기
loc = rep(loc, "        publish_voxel_map: True", "        publish_voxel_map: False   # 2026-10-29: CPU 여유 (원래 True)")
p = p[:i0] + loc + p[i1:]
# 1-1) 전역 코스트맵에도 camera_layer (2026-10-31): 지역에만 넣었더니 전역 경로 계획기가 낮은 장애물을 몰라 1초마다 장애물 위로
#      곧장 가는 경로를 다시 보냈고, TEB는 장애물 앞에서 막혀 '진행 실패' → 복구 동작만 되풀이(주행 시험 8회차).
#      전역 쪽은 expected_update_rate 0(갱신 확인 안 함): 카메라 데이터가 잠깐 늦어도 경로 계획이 멈추지 않게
g0 = p.index("\nglobal_costmap:")
g1 = p.index("\n", p.index("global_costmap_client:", g0) - 3)                # global_costmap 구간 끝(global_costmap_client 직전)
glo = p[g0:g1]
glo = rep(glo, 'plugins: ["static_layer", "obstacle_layer", "inflation_layer"]',
          'plugins: ["static_layer", "obstacle_layer", "camera_layer", "inflation_layer"]')
CAM_G = CAM.replace("expected_update_rate: 1.0", "expected_update_rate: 0.0").replace(
    "# 2026-10-29 바닥 분할 노드의 장애물 점.", "# 2026-10-31 전역에도: 계획기가 낮은 장애물을 돌아가는 경로를 만들게.")
glo = rep(glo, "      inflation_layer:\n", CAM_G + "      inflation_layer:\n")
p = p[:g0] + glo + p[g1:]
# 2026-10-30 주행 시험용 속도 제한 (velocity_smoother, 주석 처리된 옛 줄 '#max_velocity'는 건드리지 않음)
v = TEST_MAX_VEL
p = rep_re(p, r"^([ \t]+)max_velocity:[ \t]*\[0\.35, *0\.35, *1\.0\]", rf"\1max_velocity: [{v}, {v}, 1.0]   # 2026-10-30 주행 시험 (원래 [0.35, 0.35, 1.0])")
p = rep_re(p, r"^([ \t]+)min_velocity:[ \t]*\[-0\.35, *-0\.35, *-1\.0\]", rf"\1min_velocity: [-{v}, -{v}, -1.0]   # 2026-10-30 주행 시험 (원래 -0.35)")
wr("nav2_params_cam.yaml", p)

# 1-2) TEB 설정 사본: 최고 속도만 TEST_MAX_VEL로 (TEB가 계획하는 속도와 velocity_smoother 상한을 맞춤)
teb = rd(f"{SRC}/navigation/config/nav2_controller_teb.yaml")
teb = rep_re(teb, r"^([ \t]+)max_vel_x:[ \t]*0\.4[ \t]*$", rf"\1max_vel_x: {v}   # 2026-10-30 주행 시험 (원래 0.4)")
teb = rep_re(teb, r"^([ \t]+)max_vel_y:[ \t]*0\.4[ \t]*$", rf"\1max_vel_y: {v}   # 2026-10-30 주행 시험 (원래 0.4)")
wr("nav2_controller_teb_cam.yaml", teb)

# 2) navigation_base: velocity_smoother 출력 → cmd_vel_smoothed, collision_monitor 추가
nb = rd(f"{SRC}/navigation/launch/include/navigation_base.launch.py")
nb = rep(nb, "('cmd_vel_smoothed', 'cmd_vel')", "('cmd_vel_smoothed', 'cmd_vel_smoothed')")
nb = rep(nb, "os.path.join(navigation_package_path, 'config/nav2_controller_teb.yaml')",
         "'/home/ubuntu/shared/nav2_cam/nav2_controller_teb_cam.yaml'")
nb = rep(nb, "'velocity_smoother']", "'velocity_smoother',\n                       'collision_monitor']")
CM = """            ComposableNode(                    # 2026-10-29: velocity_smoother(cmd_vel_smoothed) -> collision_monitor -> cmd_vel
                package='nav2_collision_monitor',
                plugin='nav2_collision_monitor::CollisionMonitor',
                name='collision_monitor',
                parameters=['/home/ubuntu/shared/nav2_cam/collision_monitor_cam.yaml'],
                remappings=remappings),
            ComposableNode(
                package='nav2_lifecycle_manager',"""
nb = rep(nb, "            ComposableNode(\n                package='nav2_lifecycle_manager',", CM)
wr("navigation_base_cam.launch.py", nb)

# 3) bringup: navigation_base 사본 포함
bu = rd(f"{SRC}/navigation/launch/include/bringup.launch.py")
bu = rep(bu, "os.path.join(navigation_package_path, 'launch/include/navigation_base.launch.py')",
         "'/home/ubuntu/shared/nav2_cam/navigation_base_cam.launch.py'")
wr("bringup_cam.launch.py", bu)

# 4) 최상위: bringup 사본 + 설정 사본
nv = rd(f"{SRC}/navigation/launch/navigation.launch.py")
nv = rep(nv, "os.path.join(navigation_package_path, 'launch/include/bringup.launch.py')", "'/home/ubuntu/shared/nav2_cam/bringup_cam.launch.py'")
nv = rep(nv, "'params_file': os.path.join(navigation_package_path, 'config', 'nav2_params.yaml')",
         "'params_file': '/home/ubuntu/shared/nav2_cam/nav2_params_cam.yaml'")
wr("nav_cam.launch.py", nv)
print("done")
