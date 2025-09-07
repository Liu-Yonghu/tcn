# Thesis Project

This repository contains code for **Neural Architecture Search (NAS)** and **Temporal Convolutional Networks (TCN)** experiments.  
It is built on **TensorFlow/Keras** and includes modules for **hyperparameter tuning, model training, and experiment logging**.

---

## Installation

### 1. Install dependencies
```
    pip install -r requirements.txt
```
### 2. Core requirements
- Python >= 3.10  
- TensorFlow 2.14.0 (TensorFlow 2.18.0)
- Keras 2.14.0 (keras 3.0)
- Keras Tuner
---
## Basic Info

### 1. Data clean
All data cleaning progress are included in this module, `clean_data/` folder would contain the clear data.
```
clean_data.py:
        - Raw data read and feature extract 
        - Interpolation by temporal and samples
        - Global Normalization (minMax, z-Score)
        - Fix the ultrasound trajectory
        - Downsample mean labels (x,y) align with their features (N,64)
        - Save the features and labels together
drawing.py:
        - Trajectory plot
        - Heatmap plot
        - Dataset analyze
        - Compare different exp* data
```

### 2. Experiment data
Place your dataset (CSV format) into the `exp_data/` folder, e.g.:
```
exp_data/std_TOFEXP1.csv
```

### 3. Run normal training 
Please comment out the `run_search` and `collect_tuner_results` functions. You can set more parameter through `ArgumentParser`
```
python main.py --data_path ./exp_data/std_TOFEXP1.csv --model simple --epochs 200
```
### 4. Run simple hyperparameter search
Please comment out the `optimizer` functions.
```
python main.py 
```

### 5. Run complete hyperparameter search
`NAS_2.py` corresponding with keras 2.0+, and `NAS_3.py` corresponding with keras 3.0+
```
python NAS_2.py or python NAS_3.py 
```

### 6. Check results
- simple hyperparameter search
  - Best results saved under:  
    ```
    results/tcn_search/<experiment_name>/<trial_number>/
    ```
- complete hyperparameter search
  - Best results saved under:  
    ```
    autokeras_res/<experiment_name>/<trial_number>/<data_split>/
    ```
---
#### Key arguments
- `--training_part` (float): Training split ratio (default: 0.7)  
- `--validing_part` (float): Validation split ratio (default: 0.15)  
- `--time_windows` (int): Sliding window size 5s (default: 20) 
- `--model` ("simple"/"complete"): Choose model variant  
- `--epochs` (int): Number of training epochs  
- `--batch_size` (int): Batch size  

## Experiments

- **Cross-validation:** 6-fold (70% train, 15% validation, 15% ignored).  
- **Search space:**  
  - Hidden layers ∈ {2, 3, 4}  
  - Filters ∈ {8, 16, 32}  
  - Kernel size ∈ {2, 3, 4, 5}  
  - Dense units ∈ {8, 16, 32}

## Functions description
```
main.py:
        - main entry of project
NAS_*.py:
        - Net architecture automation 
tcn_model.py:
        - TCN model simple (only dilation block)
        - TCN model complex (dialtion block + normalization layers + residual block)
utils.py:
        - how to create and split dataset
        - how to compute and save test
        - how to evaluate SPARC
        - other model infos
```

---
## Author
- **Liu Yonghu**  
  Thesis Project, 2025  
  Contact: S313442@studenti.polito.it  

---