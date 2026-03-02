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
        self.jitter_dist = 20     # 扰动距离 20cm (保证位移明显)
        self.jitter_deg = 10      # 扰动角度 10度
        
        # --- 存储路径 ---
        self.root_dir = r"C:\Users\16678\Documents\junior_first\location_recognition\tello\flight_results"
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        # ==========================================
        # 📍 路径定义字典 (核心)
        # 格式: 点位ID: [动作列表]
        # 动作: ("动作名", 值)
        # 注意: Tello单次移动最大500cm，长距离需拆分
        # ==========================================
        self.flight_paths = {
            # 点位 1: 起点 (已采集，略)
            
            # --- 阶段 2: 向右飞行 (每2m悬停2s) ---
            
            # 2: 向右 2m (直接飞，无需中间悬停)
            2: [("right", 200)],
            
            # 3: 向右 4m -> 拆分为: 2m -> 停2s -> 2m
            3: [("right", 200), ("sleep", 2), ("right", 200)],
            
            # 4: 向右 6m -> 拆分为: 2m -> 停2s -> 2m -> 停2s -> 2m
            4: [("right", 200), ("sleep", 2), ("right", 200), ("sleep", 2), ("right", 200)],
            
            # 5: 向右 7.5m -> 拆分为: 2m -> 停 -> 2m -> 停 -> 2m -> 停 -> 1.5m
            5: [("right", 200), ("sleep", 2), ("right", 200), ("sleep", 2), 
                ("right", 200), ("sleep", 2), ("right", 150)],
            
            # --- 阶段 3: 向前飞行 3m (基于点位5的路径叠加) ---
            # 累计路径: 右7.5m (分段) -> 前3m
            6: [("right", 200), ("sleep", 2), ("right", 200), ("sleep", 2), 
                ("right", 200), ("sleep", 2), ("right", 150), 
                ("forward", 300)],
            
            # --- 阶段 4: 旋转 90度 ---
            # 累计路径: ... -> 顺转90
            7: [("right", 200), ("sleep", 2), ("right", 200), ("sleep", 2), 
                ("right", 200), ("sleep", 2), ("right", 150), 
                ("forward", 300), ("cw", 90)],
            
            # --- 阶段 5: 旋转后继续向右 (同样需要分段) ---
            
            # 8: ... -> 右 2m (直接飞)
            8: [("right", 200), ("sleep", 2), ("right", 200), ("sleep", 2), 
                ("right", 200), ("sleep", 2), ("right", 150), 
                ("forward", 300), ("cw", 90), 
                ("right", 200)],
            
            # 9: ... -> 右 4m (拆分: 2m -> 停 -> 2m)
            9: [("right", 200), ("sleep", 2), ("right", 200), ("sleep", 2), 
                ("right", 200), ("sleep", 2), ("right", 150), 
                ("forward", 300), ("cw", 90), 
                ("right", 200), ("sleep", 2), ("right", 200)],
            
            # 10: ... -> 右 6m (拆分: 2m -> 停 -> 2m -> 停 -> 2m)
            10: [("right", 200), ("sleep", 2), ("right", 200), ("sleep", 2), 
                 ("right", 200), ("sleep", 2), ("right", 150), 
                 ("forward", 300), ("cw", 90), 
                 ("right", 200), ("sleep", 2), ("right", 200), ("sleep", 2), ("right", 200)],
        }

    def initialize(self):
        print("="*60)
        print("🎯 单点独立采集系统 (Single Point Data Collector)")
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
        """后台监控与急停"""
        print("📹 视频监控开启 (按 'l' 紧急降落)")
        while self.running:
            frame = self.frame_read.frame
            if frame is not None:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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
            elif action == "sleep": time.sleep(val) # <--- 必须确保加上这个判断 执行悬停
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
        print("\n📍 可选采集点位:")
        print("   [2] 右移2m")
        print("   [3] 右移4m")
        print("   [4] 右移6m")
        print("   [5] 右移7.5m (转弯前)")
        print("   [6] 前移3m (长走廊终点)")
        print("   [7] 旋转90度 (悬停)")
        print("   [8] 旋转后-右移2m")
        print("   [9] 旋转后-右移4m")
        print("   [10] 旋转后-右移6m")
        
        try:
            choice = int(input("\n⌨️ 请输入要采集的点位编号 (2-10): "))
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
            print(f"   路径规划: {path}")
            
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