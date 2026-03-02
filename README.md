# UAV-SNN-RealTime-Localization | 无人机 SNN 实时位置识别系统

[English Version](#english-version) | [中文版](#中文版)

---

## English Version

A real-time position recognition and localization system for Tello UAVs based on **Spiking Neural Networks (SNN)**. This project implements a complete "Collection-Training-Deployment" engineering loop, leveraging GPU acceleration to achieve high-accuracy scene identification during flight.

### 🌟 Key Features
* **SNN Edge Intelligence**: Utilizes the event-driven and low-power characteristics of Spiking Neural Networks to address the high energy consumption of traditional vision systems.
* **GPU-Accelerated Inference**: Built a high-speed communication link between the Tello drone and a laptop GPU, enabling real-time recognition with an accuracy of over **90%**.
* **Robust Data Pipeline**: Implemented a standardized collection process with manual disturbances (20 images per scene) to enhance algorithm reliability against flight turbulence.
* **Cross-Platform Potential**: The current GPU-based deployment serves as a verification step for future migration to neuromorphic chips and Jetson platforms.

### 👥 Contributors & Credits
* **Yixiang Xu**: Responsible for **standardized dataset collection** (training & test sets) and **model deployment** on the GPU platform to achieve real-time flight recognition.
* **Shaowei Gan**: Responsible for core **model training** and architecture optimization.

### 📊 Experimental Results
* **10-Point Localization**: The core experimental setup achieved stable and high-precision scene recognition.
* **19-Point Full-Process**: A comprehensive exploration of complex scenarios (provided as a reference for large-scale dataset performance).

### 📂 Repository Structure
* `/dataset`: Standardized data collection samples.
* `/10_point_experiment`: Source code and logs for the core 10-point recognition experiment.
* `/19_point_reference`: Experimental data for the 19-point full-process (Reference only).
* `/reports`: Final presentation (PPT) and technical reports.
* `/references`: Key academic papers and documentation.

---

## 中文版

本项目是一个基于**脉冲神经网络 (SNN)** 的 Tello 无人机实时位置识别与定位系统。通过构建完整的“采集-训练-部署”工程闭环，利用笔记本 GPU 算力实现了高准确率的实时场景识别。

### 🚀 项目亮点
* **SNN 算法落地**：利用 SNN 事件驱动、低功耗特性，探索其在无人机视觉定位中的应用，解决传统处理器功耗巨大的痛点。
* **GPU 实时推理**：构建了无人机飞行拍摄与笔记本 GPU 计算的联动环境，调用 GPU 提供高算力支持，识别准确率达 **90% 以上**。
* **标准化数据采集**：通过代码构建标准化流程，每个场景采集 20 张包含人工扰动的数据，增强了算法在真实飞行（气流干扰、机身晃动）中的鲁棒性。
* **系统化验证**：搭建了从数据采集到实时部署的完整链路，为后续将算法迁移至 Jetson 与事件相机系统提供了实验依据。

### 👥 贡献与分工
* **Yixiang Xu**：负责**标准化数据集采集**（包含训练集与测试集）及**模型部署**，实现了 Tello 无人机具备飞行实时场景识别的能力。
* **Shaowei Gan**：负责核心**模型训练**与架构优化。

### 📈 实验结果
* **10 点识别实验**：核心验证实验，在实时飞行环境下实现了稳定定位。
* **19 点全流程实验**：针对更复杂场景的全流程探索。由于时间与数据集规模限制，目前作为实验参考。

### 📂 仓库结构
* `/dataset`: 标准化采集的数据集样本。
* `/10_point_experiment`: 10 点识别核心实验代码与日志。
* `/19_point_reference`: 19 点全流程实验参考数据。
* `/reports`: 期末答辩 PPT 及结课报告。
* `/references`: 相关参考文献。

---

## 🛠️ 环境要求 (Requirements)
* Python 3.10
* NVIDIA GPU + CUDA 驱动
* Tello SDK
* PyTorch / SpikingJelly (或相关 SNN 框架)

## 📝 免责声明 (Disclaimer)
本项目仅供学术交流与个人能力展示使用。
