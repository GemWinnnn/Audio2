# Heart Sound Classification Experiments  

## 📌 Overview  
This project explores different preprocessing, feature extraction, augmentation, and model architectures for **heart sound classification**.  
The focus was on:  
- Balancing datasets  
- Testing denoising approaches  
- Combining CNNs with LSTMs for better feature learning  

---

## 🔍 Observations  

### 1. Denoising  
- Used [mHealthBuet/Heart-Sound-Denoising](https://github.com/mHealthBuet/Heart-Sound-Denoising) on Google Colab.  
- Results: not significantly better for accuracy, but improved **listening quality**.  
- Conclusion: performance depended more on **feature extraction** than denoising.  

---

### 2. Feature Extraction  
- Input shape: **563 timesteps × 181 features**  
- Extracted features:  
  - MFCC  
  - ZCR  
  - Chroma  
  - Dimension  
  - RMS  
  - MelSpec (dB)  
- Adding these features gave better accuracy on Bi-LSTM compared to MFCC-only baselines.  

---

### 3. Augmentation & Segmentation  
- Original dataset: **PhysioNet Challenge 2016**  
- Segmentation:  
  - Hand-annotated + **Springer algorithm** (S1, S2, systole, diastole)  
- My approach:  
  - Applied Springer segmentation to my dataset (not clinically validated).  
  - Balanced classes by augmenting all to **250 samples each**.  

---

### 4. Model Architectures Tested  
(Ref: [Different CNN–LSTM Combinations](https://medium.com/@mijanr/different-ways-to-combine-cnn-and-lstm-networks-for-time-series-classification-tasks-b03fc37e91b6))  

- ✅ **1D CNN → LSTM** (best performance: CNN = local patterns, LSTM = temporal patterns)  
- 1D CNN  
- Bi-LSTM  
- Bi-LSTM → 1D CNN  

---

### 5. Training Strategies  
- Tried:  
  - Class rebalancing  
  - Focal Loss  
  - Label smoothing  
- Best result:  
  - **Focal Loss** → helped abnormal classes, especially **extra systoles**  
- Challenge:  
  - Validation accuracy skewed due to **small test sets** (e.g., only 14 samples for extra systole).  

---

### 6. Additional Dataset  
- Added dataset from Hugging Face:  
  👉 [DynamicSuperb/HeartAnomalyDetection_HeartbeatSounds](https://huggingface.co/datasets/DynamicSuperb/HeartAnomalyDetection_HeartbeatSounds)  
- Total: **45 extra samples**  

---

### 7. Two-Stage Model Setup  
To handle **normal class imbalance**, I split classification into two stages:  

**Stage 1**: Normal vs Abnormal  
- Model: **1D CNN**  
- Results: see `/Stage1` folder  

**Stage 2**: Abnormal Subclasses  
- Model: **1D CNN + Bi-LSTM**  
- Results: see `/Stage2` folder  

---

## 🚀 Next Steps  
- Further evaluate **Springer segmentation accuracy**  
- Test on more **external datasets**  
- Explore **attention-based models** for abnormal subclass classification  
