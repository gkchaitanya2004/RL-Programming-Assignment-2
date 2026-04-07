## 1 Team Members

| Name     | Roll No |
| -------- | ------- |
| Nikhil Enugu  | DA25M010    |
| Krishna Chaitanya | DA25M011     |
| G.C.V Sairam    | DA25M012    |


## 2 Setup

First install required libraries to run the code
```bash
pip install -r requirements.txt
```

## 3 Run

First run **main.py** file using

```
python3 main.py
```
and save the results in appropriate **naming conventions** as mentioned in the main.py file.

### 3.1 For Plots
For **plots.py** we need `.npy` files of various `truncation_length`, `rho`,`batch_size` and `target_update_frequency`

#### Experiment - 1
Here we play with different **truncation_lengths** so generate 3 files for `truncation_lengths = [200,1000,2000]` by updating the truncation_lengths in **Hyperparameters** in **main.py**

#### Experiment - 2
Here we play with different **rho** values so generate 4 files for `rho = [1,2,4,8]` by updating the rho in **Hyperparameters** in **main.py**.

#### Experiment - 3
Here we play with different **batch_size** and **rho** values so generate 16 files for `batch_size = [16,24, 64, 128]` for `rho = [1,4]`
by updating their respective values in **Hyperparameters** in **main.py**

#### Experiment - 4
Here we play with different **target_update_frequency** and **rho** values so generate 16 files for `target_update_frequency = [25, 40 ,100, 200]` for `rho = [1,4]`
by updating their respective values in **Hyperparameters** in **main.py**



Then run **plots.py**

```bash
python3 plots.py
```

Finally Run the bonus part **bonus.py** to compare this implementation with using **Prioritized Experience Replay Buffer**



```bash
python3 bonus.py
```

