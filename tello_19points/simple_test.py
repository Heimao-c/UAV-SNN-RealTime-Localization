import time
import cv2
import os
import torch
from djitellopy import Tello

# 关键：从师兄的 deploy.py 中导入场景识别类
# 这样既不用复制一大堆代码，也能直接复用他的算法

from deploy import FastDroneLocalizer


def main():
    # ================= 配置区域 =================
    MODEL_PATH = "best_vpr_snn_model.pth" 
    MAP_FILE = "offline_map_data.pt"
    SAVE_DIR = "test_results_photos"
    INTERVAL = 5.0
    # ===========================================

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print("="*60)
    print("          SNN 场景识别算法可行性测试 (静态)")
    print("="*60)

    # 1. 初始化算法模型
    print("⏳ [1/3] 正在加载 PyTorch 模型 (使用 GPU/CPU)...")
    try:
        # 这里的 device 逻辑会沿用 deploy.py 里的写法
        # 如果你已经在 deploy.py 里改成了 cpu，这里也会是 cpu
        localizer = FastDroneLocalizer(MODEL_PATH, MAP_FILE)
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("提示：请检查 MODEL_PATH 和 MAP_FILE 路径是否正确。")
        return

    # 2. 连接无人机
    print("\n⏳ [2/3] 连接 Tello 无人机...")
    tello = Tello()
    try:
        tello.connect()
        tello.streamon()
        bat = tello.get_battery()
        print(f"✅ Tello 连接成功 | 电量: {bat}%")
        if bat < 15:
            print("⚠️ 警告：电量过低！")
    except Exception as e:
        print(f"❌ Tello 连接失败 (检查Wi-Fi或防火墙): {e}")
        return

    # 3. 开启循环
    print("\n⏳ [3/3] 启动视频流处理...")
    frame_reader = tello.get_frame_read()
    time.sleep(2) # 等待视频流稳定

    print("\n🎥 系统就绪！")
    print(f"   - 策略: 每 {INTERVAL} 秒识别一次")
    print("   - 操作: 按 'q' 键退出程序")
    print("-" * 60)

    last_process_time = time.time() - INTERVAL # 让第一次循环立即触发
    
    while True:
        # 获取画面 (Tello 发送的是 RGB 格式)
        frame_rgb = frame_reader.frame
        if frame_rgb is None:
            time.sleep(0.01)
            continue

        # === 关键修正：颜色空间转换 ===
        # OpenCV 显示和 deploy.py 里的预测函数都期望输入是 BGR 格式
        # 所以必须先把 RGB 转为 BGR，否则人脸会变蓝，且识别准确率下降
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 实时显示画面
        cv2.imshow("PyTorch Algorithm Test", frame_bgr)

        # 检查是否到达时间间隔
        current_time = time.time()
        if current_time - last_process_time > INTERVAL:
            print("\n📸 正在捕获并识别...", end="", flush=True)
            
            try:
                # --- 调用算法 ---
                start_t = time.time()
                place, dist = localizer.predict_frame(frame_bgr) # 传入BGR图像
                cost_time = (time.time() - start_t) * 1000
                
                print(f" 完成! ({cost_time:.1f}ms)")
                
                # --- 处理结果 ---
                # dist 越小越相似。通常 <0.8 认为是确信的
                confidence = "High" if dist < 0.8 else "Low"
                result_str = f"Loc: {place} | Dist: {dist:.3f}"
                
                print(f"   👉 识别结果: {place}")
                print(f"   👉 特征距离: {dist:.4f} ({confidence})")

                # --- 绘制并保存图片 ---
                save_img = frame_bgr.copy()
                
                # 画一个黑色背景条，让文字更清晰
                cv2.rectangle(save_img, (0, 0), (640, 40), (0, 0, 0), -1)
                
                # 写入文字 (绿色代表置信度高，红色代表低)
                color = (0, 255, 0) if dist < 0.8 else (0, 0, 255)
                cv2.putText(save_img, result_str, (10, 28), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # 保存文件
                timestamp = time.strftime("%H%M%S")
                filename = f"{SAVE_DIR}/rec_{timestamp}_{place}.jpg"
                cv2.imwrite(filename, save_img)
                print(f"   💾 已保存图片: {filename}")
                
            except Exception as e:
                print(f"\n❌ 识别过程出错: {e}")
            
            last_process_time = current_time

        # 键盘退出控制
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 用户请求退出...")
            break

    # 资源清理
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()
    print("✅ 测试结束，资源已释放")

if __name__ == "__main__":
    main()