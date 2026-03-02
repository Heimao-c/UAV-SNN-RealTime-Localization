import time
import threading
import cv2
import os
import random
import sys
from djitellopy import Tello

class SinglePointCollector:
    def __init__(self):
        # --- 基础状态 ---
        self.tello = Tello()
        self.running = False      
        self.emergency = False    
        self.frame_lock = threading.Lock()
        self.current_frame = None 
        
        # --- 采集参数 ---
        self.photos_per_point = 20 # 每个点拍20张
        self.jitter_dist = 20     # 扰动距离 20cm
        self.jitter_deg = 10      # 扰动角度 10度
        
        # --- 存储路径 ---
        self.root_dir = r"C:\Users\16678\Pictures\new_criterion"
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        # ==========================================
        # 📍 路径定义 (使用叠加逻辑，避免代码冗余)
        # ==========================================
        
        # 1. 定义基础片段 (用来拼接长路径)
        # 阶段 1: 起点 (点位1, 略)
        
        # 阶段 2: 第一条长廊 (向右 7.5m)
        path_stage_2_full = [
            ("right", 200), ("sleep", 2), 
            ("right", 200), ("sleep", 2), 
            ("right", 200), ("sleep", 2), 
            ("right", 150)
        ]
        
        # 阶段 3: 前进 3m
        path_stage_3_add = [("forward", 300)]
        
        # 阶段 4: 第一次转弯 (顺90)
        path_stage_4_add = [("cw", 90)]
        
        # 基础路径 A (到达点位 7 的路径: 右7.5 -> 前3 -> 转90)
        base_path_to_7 = path_stage_2_full + path_stage_3_add + path_stage_4_add

        # -------------------------------------------------
        # 构造 flight_paths 字典
        # -------------------------------------------------
        self.flight_paths = {
            # --- 阶段 2: 向右飞行 (每2m悬停2s) ---
            2: [("right", 200)],
            3: [("right", 200), ("sleep", 2), ("right", 200)],
            4: [("right", 200), ("sleep", 2), ("right", 200), ("sleep", 2), ("right", 200)],
            5: path_stage_2_full, # 7.5m
            
            # --- 阶段 3: 向前飞行 3m ---
            6: path_stage_2_full + [("forward", 300)],
            
            # --- 阶段 4: 旋转 90度 ---
            7: base_path_to_7,
            
            # --- 阶段 5: 旋转后继续向右 (修改：总长改为 8m) ---
            # 8: 右 2m
            8: base_path_to_7 + [("right", 200)],
            
            # 9: 右 4m
            9: base_path_to_7 + [
                ("right", 200), ("sleep", 2), ("right", 200)
            ],
            
            # 10: 右 6m (原有逻辑) -> 修改：此处仅作为中间点，但因为你要采集数据，
            # 如果原计划Point 10就是6m处，保留它。
            # 如果你要把终点改为8m，那Point 10通常指代 6m 处。
            # 下面是累积到 6m
            10: base_path_to_7 + [
                ("right", 200), ("sleep", 2), ("right", 200), ("sleep", 2), ("right", 200)
            ],
            
            # [新增] 11: 右 8m (阶段5终点)
            # 路径: 基础A -> 右2 -> 停 -> 右2 -> 停 -> 右2 -> 停 -> 右2
            11: base_path_to_7 + [
                ("right", 200), ("sleep", 2), ("right", 200), ("sleep", 2), 
                ("right", 200), ("sleep", 2), ("right", 200)
            ],
        }

        # --- 定义到达 点位11 (右8m处) 后的基础路径，方便后面拼接 ---
        base_path_to_11 = self.flight_paths[11]

        # --- 阶段 6: 再次顺转 90度 ---
        self.flight_paths[12] = base_path_to_11 + [("cw", 90)]
        
        # 定义到达 点位12 (转弯后) 的基础路径
        base_path_to_12 = self.flight_paths[12]

        # --- 阶段 7: 向右飞 8m (分为 2, 4, 6, 8 四个采集点) ---
        # 13: 右 2m
        self.flight_paths[13] = base_path_to_12 + [("right", 200)]
        
        # 14: 右 4m
        self.flight_paths[14] = base_path_to_12 + [("right", 200), ("sleep", 2), ("right", 200)]
        
        # 15: 右 6m
        self.flight_paths[15] = base_path_to_12 + [
            ("right", 200), ("sleep", 2), ("right", 200), ("sleep", 2), ("right", 200)
        ]
        
        # 16: 右 8m (阶段7终点)
        self.flight_paths[16] = base_path_to_12 + [
            ("right", 200), ("sleep", 2), ("right", 200), ("sleep", 2), 
            ("right", 200), ("sleep", 2), ("right", 200)
        ]
        
        # 定义到达 点位16 的基础路径
        base_path_to_16 = self.flight_paths[16]

        # --- 阶段 8: 连续顺转 (90 + 90) ---
        # 17: 顺转 90
        self.flight_paths[17] = base_path_to_16 + [("cw", 90)]
        
        # 18: 再顺转 90 (此时相当于掉头或进入下一条路)
        self.flight_paths[18] = base_path_to_16 + [("cw", 90), ("sleep", 1), ("cw", 90)]
        
        # 定义到达 点位18 的基础路径
        base_path_to_18 = self.flight_paths[18]

        # --- 阶段 9: 向右飞 6m (分为 2, 4, 6 三个采集点) ---
        # 19: 右 2m
        self.flight_paths[19] = base_path_to_18 + [("right", 200)]
        
        # 20: 右 4m
        self.flight_paths[20] = base_path_to_18 + [("right", 200), ("sleep", 2), ("right", 200)]
        
        # 21: 右 6m
        self.flight_paths[21] = base_path_to_18 + [
            ("right", 200), ("sleep", 2), ("right", 200), ("sleep", 2), ("right", 200)
        ]

    def initialize(self):
        print("="*60)
        print("🎯 单点独立采集系统 (Extended Version)")
        print(f"📂 存储根目录: {self.root_dir}")
        print("="*60)
        try:
            self.tello.connect()
            self.tello.streamon()
            self.frame_read = self.tello.get_frame_read()
            time.sleep(1)
            
            bat = self.tello.get_battery()
            print(f"🔋 电量: {bat}%")
            if bat < 20:
                print("⚠️ 电量过低，请更换电池！")
            return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False

    def video_worker(self):
        """后台监控与急停 + 电池显示"""
        print("📹 视频监控开启 (按 'l' 紧急降落)")
        while self.running:
            frame = self.frame_read.frame
            if frame is not None:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # --- 新增功能：电池电量显示 ---
                try:
                    bat = self.tello.get_battery()
                    # 电量颜色: >20% 绿色, <=20% 红色
                    bat_color = (0, 255, 0) if bat > 20 else (0, 0, 255)
                    cv2.putText(bgr, f"Battery: {bat}%", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, bat_color, 2)
                except:
                    pass
                # ---------------------------

                with self.frame_lock:
                    self.current_frame = bgr.copy()
                
                cv2.imshow("Collector View", bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('l'):
                    print("\n🚨 紧急降落触发！")
                    self.emergency = True
                    self.running = False
                    self.tello.land()
                    break
            else:
                time.sleep(0.01)
        cv2.destroyAllWindows()

    def smart_move(self, action, val):
        """执行飞行动作"""
        if self.emergency: return
        print(f"   ✈️ 执行动作: {action} {val}")
        try:
            if action == "right": self.tello.move_right(val)
            elif action == "left": self.tello.move_left(val)
            elif action == "forward": self.tello.move_forward(val)
            elif action == "back": self.tello.move_back(val)
            elif action == "up": self.tello.move_up(val)
            elif action == "down": self.tello.move_down(val)
            elif action == "cw": self.tello.rotate_clockwise(val)
            elif action == "ccw": self.tello.rotate_counter_clockwise(val)
            elif action == "sleep": 
                print(f"      ...悬停等待 {val}s")
                time.sleep(val) 
            
            # 动作后稍微稳定一下，防止惯性累积
            time.sleep(0.5) 
        except Exception as e:
            print(f"   ⚠️ 动作执行警告: {e}")

    def collect_data(self, point_id):
        """在当前位置执行20次扰动采集"""
        folder = os.path.join(self.root_dir, f"point_{point_id:03d}")
        if not os.path.exists(folder): os.makedirs(folder)
        
        print(f"\n📸 [Point {point_id:03d}] 到达目标，开始采集 20 张图片...")
        
        for i in range(1, self.photos_per_point + 1):
            if self.emergency: break
            
            # 0:原地, 1-8:各方向扰动
            action_type = random.randint(0, 8) if i > 2 else 0
            reset_cmd = None
            
            try:
                # 1. 扰动
                if action_type == 1: # 左
                    self.tello.move_left(self.jitter_dist); reset_cmd = lambda: self.tello.move_right(self.jitter_dist)
                elif action_type == 2: # 右
                    self.tello.move_right(self.jitter_dist); reset_cmd = lambda: self.tello.move_left(self.jitter_dist)
                elif action_type == 3: # 前
                    self.tello.move_forward(self.jitter_dist); reset_cmd = lambda: self.tello.move_back(self.jitter_dist)
                elif action_type == 4: # 后
                    self.tello.move_back(self.jitter_dist); reset_cmd = lambda: self.tello.move_forward(self.jitter_dist)
                elif action_type == 5: # 上
                    self.tello.move_up(self.jitter_dist); reset_cmd = lambda: self.tello.move_down(self.jitter_dist)
                elif action_type == 6: # 下
                    self.tello.move_down(self.jitter_dist); reset_cmd = lambda: self.tello.move_up(self.jitter_dist)
                elif action_type == 7: # 顺转
                    self.tello.rotate_clockwise(self.jitter_deg); reset_cmd = lambda: self.tello.rotate_counter_clockwise(self.jitter_deg)
                elif action_type == 8: # 逆转
                    self.tello.rotate_counter_clockwise(self.jitter_deg); reset_cmd = lambda: self.tello.rotate_clockwise(self.jitter_deg)
                
                time.sleep(0.5) # 稳定

                # 2. 拍照
                with self.frame_lock:
                    if self.current_frame is not None:
                        fpath = os.path.join(folder, f"{i:02d}.jpg")
                        cv2.imwrite(fpath, self.current_frame)
                        print(f"   ✅ 拍照 {i}/{self.photos_per_point}")

                # 3. 复位
                if reset_cmd: 
                    reset_cmd()
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"   ⚠️ 采集出错: {e}")

    def run(self):
        # 1. 选择点位
        print("\n📍 可选采集点位 (已更新):")
        print("   [2-5]  第一阶段：右移 (2m - 7.5m)")
        print("   [6]    长廊终点：前移3m")
        print("   [7]    转角：顺转90°")
        print("   [8-11] 第二阶段：右移 (2m, 4m, 6m, 8m)")
        print("   [12]   转角：顺转90°")
        print("   [13-16]第三阶段：右移 (2m, 4m, 6m, 8m)")
        print("   [17]   转角：顺转90°")
        print("   [18]   转角：再次顺转90°")
        print("   [19-21]第四阶段：右移 (2m, 4m, 6m)")
        
        try:
            choice = int(input("\n⌨️ 请输入要采集的点位编号 (2-21): "))
            if choice not in self.flight_paths:
                print("❌ 无效的编号")
                return
        except ValueError:
            print("❌ 输入错误")
            return

        # 2. 启动系统
        if not self.initialize(): return
        
        self.running = True
        t = threading.Thread(target=self.video_worker, daemon=True)
        t.start()
        time.sleep(1)

        try:
            # 3. 标准起飞程序
            print("\n🛫 [阶段1] 起飞并直达 1.6m...")
            self.tello.takeoff()
            self.tello.move_up(80) # 80+80=1.6m
            time.sleep(1)

            # 4. 执行导航飞行
            path = self.flight_paths[choice]
            print(f"\n🚀 [阶段2] 正在前往点位 {choice:03d}...")
            # 简化打印，只打印动作名
            # print(f"   路径规划: {path}") 
            
            for action, val in path:
                if self.emergency: break
                self.smart_move(action, val)

            # 5. 到达后采集
            if not self.emergency:
                print(f"\n📍 已到达点位 {choice:03d}，开始稳定悬停...")
                time.sleep(2) # 采集前多稳一会
                self.collect_data(choice)

        except Exception as e:
            print(f"⚠️ 任务异常: {e}")

        # 6. 任务结束
        print("\n🏁 任务结束，正在降落...")
        if not self.emergency:
            self.tello.land()
        
        self.running = False
        self.tello.end()

if __name__ == "__main__":
    bot = SinglePointCollector()
    bot.run()