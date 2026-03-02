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
        # 对应 Collection 中的 ("sleep", 2) + smart_move 的 0.5s 缓冲
        # Collection 逻辑: 动作 -> sleep(0.5) -> sleep(2) -> sleep(0.5) = 总共约 3.0s
        self.stable_time = 2.0    
        self.buffer_time = 0.5    # 动作后的机械缓冲时间
        
        # --- 存储路径 ---
        self.save_dir = r"C:\Users\16678\Documents\junior_first\location_recognition\tello_10points_3\flight_results_10points"
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
        print("🚀 初始化智能飞行系统 (高稳定性复刻版)")
        print("="*60)

        # 1. 加载 AI 模型
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
            time.sleep(1) 
            
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
        """拍照 -> 识别 -> 保存"""
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

            # 保存
            ts = datetime.now().strftime("%H%M%S")
            safe_tag = step_tag.replace(" ", "_")
            fname = os.path.join(self.save_dir, f"{ts}_{safe_tag}_{place}.jpg")
            cv2.imwrite(fname, frame_to_process)

        except Exception as e:
            print(f"   ❌ 识别出错: {e}")

    def execute_step(self, action_type, val, step_name):
        """
        执行单个步骤: 严格复刻 Collection.py 的时序逻辑
        逻辑: 动作 -> 缓冲0.5s -> 悬停2s -> 缓冲0.5s -> 识别
        """
        if self.emergency: return

        print(f"\n📍 [步骤] {step_name}: {action_type} {val}")
        
        try:
            # 1. 执行飞行动作
            if action_type == "right":
                self.tello.move_right(val)
            elif action_type == "forward":
                self.tello.move_forward(val)
            elif action_type == "cw":
                self.tello.rotate_clockwise(val)
            
            # --- 关键修改：复刻 Collection.py 的 smart_move 缓冲 ---
            # Collection 在每次 move 后都会 sleep(0.5)
            print(f"       ⏳ 机械缓冲 {self.buffer_time}s...", end="")
            time.sleep(self.buffer_time) 
            print("OK")
            
            # 2. 模拟 Collection 中的 ("sleep", 2)
            # Collection 遇到 sleep 指令会: sleep(val) 然后再 sleep(0.5)
            print(f"       ⚓ 稳定悬停 {self.stable_time}s (+缓冲)...")
            time.sleep(self.stable_time)
            time.sleep(self.buffer_time) # 第二次缓冲，确保完全静止

            # 3. 视觉识别
            self.process_scene_recognition(step_name)

        except Exception as e:
            print(f"   ⚠️ 动作执行异常: {e}")

    def run_mission(self):
        """主飞行任务"""
        self.running = True
        t = threading.Thread(target=self.video_and_control_worker, daemon=True)
        t.start()
        time.sleep(2)

        # =========================================================
        # 📍 路径规划 (与 Collection.py 路径完全一致)
        # 
        # Collection Point 5 路径: R200, S2, R200, S2, R200, S2, R150
        # Collection Point 6 路径: ... R150, Forward 300
        # =========================================================
        move_plan = [
            # --- 阶段 2: 向右飞行 ---
            ("Pt2_Right_200", "right", 200),
            ("Pt3_Right_200", "right", 200),
            ("Pt4_Right_200", "right", 200),
            ("Pt5_Right_150", "right", 150), # 注意这里是150，与Collection一致

            # --- 阶段 3: 向前飞行 ---
            ("Pt6_Forward_300", "forward", 300),

            # --- 阶段 4: 旋转 ---
            ("Pt7_CW_90", "cw", 90),

            # --- 阶段 5: 旋转后右移 ---
            ("Pt8_Right_200", "right", 200),
            ("Pt9_Right_200", "right", 200),
            ("Pt10_Right_200", "right", 200),
        ]
        
        print("\n🛫 === 开始验证飞行 (高稳复刻版) ===")
        
        try:
            if not self.emergency:
                # === 1. 起飞 ===
                print("\n📍 [起飞] Takeoff -> Up 80")
                self.tello.takeoff()       
                self.tello.move_up(80)    
                
                # 初始悬停 (复刻 sleep 2 + buffer)
                time.sleep(self.stable_time + self.buffer_time)
                self.process_scene_recognition("Pt1_Start")

            # === 2. 执行计划 ===
            for name, action, val in move_plan:
                if self.emergency: break
                self.execute_step(action, val, name)

            if not self.emergency:
                print("\n⬇️ 任务完成，降落...")
                self.tello.land()
                    
        except Exception as e:
            print(f"⚠️ 任务异常: {e}")
            if not self.emergency:
                self.tello.land()

        print("\n✅ 系统关闭")
        self.running = False
        time.sleep(1)
        self.tello.end()

if __name__ == "__main__":
    mission = SmartTelloMission()
    if mission.initialize():
        try:
            mission.run_mission()
        except KeyboardInterrupt:
            mission.running = False
            mission.tello.land()