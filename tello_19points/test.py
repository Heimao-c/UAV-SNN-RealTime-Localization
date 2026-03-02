import time
import threading
import cv2
import os
import torch
import numpy as np
from datetime import datetime
from djitellopy import Tello

# 导入算法类
try:
    from deploy import FastDroneLocalizer
except ImportError:
    print("❌ 错误：找不到 deploy.py，请确保文件在同一目录下！")
    exit()

class SmartTelloMission:
    def __init__(self):
        # --- 基础状态 ---
        self.tello = Tello()
        self.running = False      # 控制视频线程的主开关
        self.emergency = False    # 紧急停止标志
        self.current_frame = None # 用于主线程获取最新画面
        self.frame_lock = threading.Lock()
        
        # --- 飞行参数 ---
        self.safe_step = 200      # 分段移动步长 (cm)，即每隔2米悬停一次
        self.stable_time = 2.0    # 悬停稳定时间 (也是拍照等待时间)
        
        # --- 存储路径 ---
        self.save_dir = r"C:\Users\16678\Pictures\flight_results"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # --- 算法模型相关 ---
        self.localizer = None
        self.device_name = "Unknown"
        self.model_path = "best_vpr_snn_model.pth"
        self.map_file = "offline_map_data.pt"

    def initialize(self):
        """初始化无人机和算法"""
        print("="*60)
        print("🚀 初始化智能飞行系统 (分段识别版)")
        print("="*60)

        # 1. 加载 AI 模型 (GPU 优先)
        print("🧠 [AI] 正在加载模型...")
        try:
            if torch.cuda.is_available():
                self.device_name = f"GPU ({torch.cuda.get_device_name(0)})"
                device = torch.device("cuda")
            else:
                self.device_name = "CPU"
                device = torch.device("cpu")
            
            self.localizer = FastDroneLocalizer(self.model_path, self.map_file, device=device)
            print(f"✅ [AI] 模型加载成功！运行设备: 【{self.device_name}】")
        except Exception as e:
            print(f"❌ [AI] 模型加载失败: {e}")
            return False

        # 2. 连接无人机
        print("\n🚁 [无人机] 正在连接...")
        try:
            self.tello.connect()
            self.tello.streamon()
            
            self.frame_read = self.tello.get_frame_read()
            time.sleep(1) # 等待视频流稳定
            
            bat = self.tello.get_battery()
            print(f"✅ [无人机] 连接成功 | 电量: {bat}%")
            if bat < 20:
                print("⚠️ 警告：电量过低！")
            return True
        except Exception as e:
            print(f"❌ [无人机] 初始化失败: {e}")
            return False

    def video_and_control_worker(self):
        """后台线程：显示画面 + 监听键盘 L 键"""
        print(f"📹 视频监控已启动 (请选中视频窗口，按 'l' 键紧急降落)")
        
        while self.running:
            rgb_frame = self.frame_read.frame
            if rgb_frame is not None:
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                
                with self.frame_lock:
                    self.current_frame = bgr_frame.copy()

                cv2.imshow("Tello Smart View (Press 'l' to Land)", bgr_frame)
                
                # 监听键盘
                key = cv2.waitKey(1) & 0xFF
                if key == ord('l'):
                    print("\n🚨🚨🚨 键盘触发紧急降落！🚨🚨🚨")
                    self.emergency = True
                    self.running = False 
                    try:
                        self.tello.land()
                    except:
                        self.tello.emergency()
                    break
            else:
                time.sleep(0.01)

        cv2.destroyAllWindows()

    def process_scene_recognition(self, step_tag):
        """拍照 -> 识别 -> 绘图 -> 保存"""
        if self.emergency: return

        print(f"   📸 [AI] {step_tag} 识别中...", end="")
        
        frame_to_process = None
        with self.frame_lock:
            if self.current_frame is not None:
                frame_to_process = self.current_frame.copy()
        
        if frame_to_process is None:
            print(" 失败 (无画面)")
            return

        try:
            # 算法推理
            t0 = time.time()
            place, dist = self.localizer.predict_frame(frame_to_process)
            t_cost = (time.time() - t0) * 1000
            
            print(f" 完成! {t_cost:.1f}ms | {place}")

            # 绘图
            cv2.rectangle(frame_to_process, (0, 0), (640, 110), (0, 0, 0), -1)
            color = (0, 255, 0) if dist < 0.8 else (0, 0, 255)
            
            cv2.putText(frame_to_process, f"LOC: {place}", (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame_to_process, f"DIST: {dist:.4f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            info_str = f"TIME: {t_cost:.0f}ms | DEV: {self.device_name}"
            cv2.putText(frame_to_process, info_str, (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 保存
            ts = datetime.now().strftime("%H%M%S")
            safe_tag = step_tag.replace(" ", "_")
            fname = os.path.join(self.save_dir, f"{ts}_{safe_tag}_{place}.jpg")
            cv2.imwrite(fname, frame_to_process)

        except Exception as e:
            print(f"   ❌ 识别出错: {e}")

    def safe_move(self, distance, move_func, action_name):
        """
        分段移动逻辑 (核心修改)
        - distance: 总距离
        - move_func: Tello移动函数
        - action_name: 动作名称
        
        逻辑：每移动一段 (self.safe_step = 200cm)，就悬停并拍照识别一次。
        """
        segments = distance // self.safe_step
        remainder = distance % self.safe_step
        
        # 1. 执行完整分段 (每2米)
        for i in range(segments):
            if self.emergency: return
            
            print(f"    -> {action_name} 分段 {i+1}/{segments} ({self.safe_step}cm)...")
            move_func(self.safe_step) # 移动
            
            # 移动后悬停
            print(f"       ⏳ 悬停稳定 {self.stable_time}s...")
            time.sleep(self.stable_time)
            
            # === 关键：分段位置拍照识别 ===
            self.process_scene_recognition(f"{action_name}_Seg{i+1}")
            
        # 2. 执行剩余距离
        if remainder > 0 and not self.emergency:
            print(f"    -> {action_name} 剩余段 ({remainder}cm)...")
            move_func(remainder)
            
            # 移动后悬停
            print(f"       ⏳ 悬停稳定 {self.stable_time}s...")
            time.sleep(self.stable_time)
            
            # === 关键：终点位置拍照识别 ===
            self.process_scene_recognition(f"{action_name}_End")

    def run_mission(self):
        """主飞行任务"""
        self.running = True
        t = threading.Thread(target=self.video_and_control_worker, daemon=True)
        t.start()
        time.sleep(2)

        # 定义后续移动计划 (去掉起飞和上升，因为我们会单独处理)
        # 格式: (名称, 动作类型, 距离/角度)
        move_plan = [
            ("右移770",    "right",   770), # 这里的每2m都会识别
            ("前移200",    "forward", 200),
            ("顺转90",     "cw",      90),
            ("右移800",    "right",   800),
            ("顺转90",     "cw",      90),
            ("右移600",    "right",   600),
            ("顺转90",     "cw",      90),
            ("顺转90",     "cw",      90),
            ("右移400",    "right",   400),
            ("降落",       "land",    0)
        ]
        
        print("\n🛫 === 开始自动巡检任务 ===")
        
        try:
            if not self.emergency:
                # === 1. 直接起飞至 1.8m (User Request) ===
                print("\n📍 [步骤] 起飞并直达 1.6m (Takeoff -> Up 80)")
                self.tello.takeoff()       # 默认约 80cm
                self.tello.move_up(80)    # 补 80cm -> 总共 1.6m
                
                # 到达高度后，稳定并识别一次
                print(f"   ⏳ 到达高度，稳定 {self.stable_time}s...")
                time.sleep(self.stable_time)
                self.process_scene_recognition("Takeoff_Height_1.6m")

            # === 2. 执行移动计划 ===
            for name, action, val in move_plan:
                if self.emergency: break
                
                print(f"\n📍 [步骤] {name}")
                
                if action == "cw": 
                    # 旋转动作通常不需要分段，转完拍一张即可
                    self.tello.rotate_clockwise(val)
                    time.sleep(1.0) # 旋转后稍作等待
                    self.process_scene_recognition(f"{name}_Rotated")
                    
                elif action == "right": 
                    # 使用支持分段识别的移动函数
                    self.safe_move(val, self.tello.move_right, name)
                    
                elif action == "forward": 
                    self.safe_move(val, self.tello.move_forward, name)
                    
                elif action == "back": 
                    self.safe_move(val, self.tello.move_back, name)
                    
                elif action == "land":
                    print("⬇️ 正在降落...")
                    self.tello.land()
                    
        except Exception as e:
            print(f"⚠️ 任务异常: {e}")
            if not self.emergency:
                self.tello.land()

        print("\n✅ 任务结束")
        self.running = False
        time.sleep(1)
        self.tello.end()

if __name__ == "__main__":
    mission = SmartTelloMission()
    if mission.initialize():
        try:
            mission.run_mission()
        except KeyboardInterrupt:
            print("\n🛑 用户强制中断")
            mission.running = False
            mission.tello.land()