import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

from djitellopy import Tello


class SeqToANNContainer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)
            
    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)
    
@torch.jit.script
def heaviside(x: torch.Tensor):
    
    return (x>=0).float()

@torch.jit.script
def grad_cal(grad_output: torch.Tensor, x: torch.Tensor, gamma: torch.Tensor):
    return grad_output * ((gamma - x.abs()).clamp(min=0)) / (gamma * gamma)

class SG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma):
        out = heaviside(input)
        ctx.save_for_backward(input)
        ctx.gamma = gamma
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_cal(grad_output, ctx.saved_tensors[0], ctx.gamma), None


class LIF(nn.Module):
    def __init__(self, T=5, v_th=0.5, gamma=0.5, target_rate=0.20, rate_margin=0.03):
        super(LIF, self).__init__()
        self.T = T
        self.heaviside = SG.apply
        self.register_buffer('v_th', torch.tensor(v_th))
        self.register_buffer('gamma', torch.tensor(gamma))
        self.register_buffer('w', torch.tensor(0.5))

        self.target_rate = torch.tensor(0.2)
        self.lower = target_rate - rate_margin 
        self.upper = target_rate + rate_margin
        self.alpha = nn.Parameter(torch.as_tensor(0.0))
        self.beta = nn.Parameter(torch.as_tensor(0.0))

        self.enable_monitor = False 
        self.monitor_data = {}

    def forward(self, x_exc, x_inh):
        mem_v = []
        mem = torch.zeros_like(x_exc[0])
        spike = torch.zeros_like(mem)
        ema_firing_rate = torch.full_like(mem, 0.17)
        inh_weight = torch.zeros_like(mem)
        beta = self.beta.sigmoid()
        alpha = 4 * self.alpha.sigmoid()
        temp_mem_t = [] 
        temp_ema_flat = [] 
        for t in range(self.T):
            e_input  = x_exc[t] / (1.0 + alpha * x_inh[t])
            mem = self.w * mem + e_input - beta * x_inh[t] * (1 - inh_weight)
            
            if self.enable_monitor:
                temp_mem_t.append(mem.detach().cpu().flatten().float())
            
            spike = self.heaviside(mem - self.v_th, self.gamma)
            
            ema_firing_rate = (0.9 * ema_firing_rate + 0.1 * spike.detach()).mean(dim=(0,2,3), keepdim=True)
            
            if self.enable_monitor:
                
                temp_ema_flat.append(ema_firing_rate.detach().cpu().flatten().float())

            if t < (self.T -1):
                inh_weight = 4.0 * (torch.sigmoid(self.lower - ema_firing_rate) - torch.sigmoid(ema_firing_rate - self.upper))
            
            mem_v.append(spike)
            
            mem = mem - self.v_th * spike
            
        if self.enable_monitor:
            
            self.monitor_data['mem_t'] = torch.stack(temp_mem_t).numpy()
            
            self.monitor_data['ema_flat'] = torch.cat(temp_ema_flat).numpy()
            
        return torch.stack(mem_v, dim=0)
    
class Layer(nn.Module):
    def __init__(self, T, in_plane, out_plane, kernel_size, stride, padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, 2 * out_plane, kernel_size, stride, padding),
            nn.BatchNorm2d(2 * out_plane),
            nn.ReLU())
        self.act = LIF(T)
    def forward(self, x):
        x_exc, x_inh = self.fwd(x).chunk(2, dim=2)
        x = self.act(x_exc, x_inh)
        return x




class VGGSNN(nn.Module):
    def __init__(self, T=5):
        super(VGGSNN, self).__init__()
        
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.T = T
        self.features = nn.Sequential(
            Layer(T, 3, 32, 3, 1, 1),
            Layer(T, 32, 64, 3, 1, 1),

            pool,
            Layer(T, 64, 128, 3, 1, 1),
            Layer(T, 128, 128, 3, 1, 1),

            pool,
            Layer(T, 128, 256, 3, 1, 1),
            Layer(T, 256, 256, 3, 1, 1),

            pool,
            Layer(T, 256, 512, 3, 1, 1),
            Layer(T, 512, 512, 3, 1, 1),

            pool,
        )
        self.p = nn.Parameter(torch.ones(1) * 3)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1) 
        out_spikes = self.features(x)
        firing_rate = out_spikes.mean(dim=0)
        gem_out = f.avg_pool2d(firing_rate.clamp(min=1e-6).pow(self.p), 
                               (firing_rate.size(-2), firing_rate.size(-1)))
        gem_out = gem_out.pow(1.0 / self.p)
        descriptor = gem_out.view(gem_out.size(0), -1)
        descriptor = f.normalize(descriptor, p=2, dim=1)
        return descriptor
   

class FastDroneLocalizer:
    def __init__(self, model_path, map_file_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = 64
        self.T = 4
        
        # 1. 加载 SNN 模型 (用于处理无人机画面)
        print(f"[System] Loading SNN model...")
        self.model = VGGSNN(T=self.T).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # 2. 直接加载预处理好的地图文件
        print(f"[System] Loading Offline Map from {map_file_path}...")
        if not os.path.exists(map_file_path):
            raise FileNotFoundError("找不到地图文件，请先运行 build_offline_map.py")
            
        map_data = torch.load(map_file_path, map_location=self.device)
        
        self.gallery_feats = map_data["features"].to(self.device)
        self.gallery_labels_idx = map_data["labels_idx"].to(self.device)
        self.class_names = map_data["class_names"]
        
        print(f"[System] Map Loaded! Contains {len(self.gallery_feats)} points.")

        # 预处理 (仅用于无人机画面)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4695, 0.4685, 0.4592],
                     std=[0.1899, 0.1964, 0.2099])
        ])

    def predict_frame(self, frame_bgr):
        # 处理单帧
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            query_feat = self.model(img_tensor)
        
        # 匹配
        dists = torch.cdist(query_feat, self.gallery_feats, p=2)
        min_val, min_idx = torch.min(dists, dim=1)
        
        idx = min_idx.item()
        
        # 从保存的信息中恢复地点名称
        pred_label_idx = self.gallery_labels_idx[idx].item()
        pred_place = self.class_names[pred_label_idx]
        
        return pred_place, min_val.item()

def main():
    # --- 配置 ---
    MODEL_PATH = "./checkpoint/DLIF/best_vpr_snn_model.pth"
    MAP_FILE = "offline_map_data.pt"  # 刚才生成的文件
    
    # 初始化
    localizer = FastDroneLocalizer(MODEL_PATH, MAP_FILE)
    
    # 连接 Tello
    print("Connecting to Tello...")
    me = Tello()
    try:
        me.connect()
        print(f"Battery: {me.get_battery()}%")
    except Exception as e:
        print("Connection failed. Check Wi-Fi.")
        return

    me.streamon()
    frame_read = me.get_frame_read()
    
    print("System Ready. Press 'q' to quit.")
    
    while True:
        frame = frame_read.frame
        if frame is None: continue
        
        display_frame = cv2.resize(frame, (640, 480))
        
        # 识别
        place, dist = localizer.predict_frame(display_frame)
        
        # 显示
        if dist < 0.8:
            cv2.putText(display_frame, f"LOC: {place}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, f"Uncertain ({place})", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        cv2.imshow("Fast Tello VPR", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    me.streamoff()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()