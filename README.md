# Rolling-Window Alpha Parameter Fitting

This repository contains the code used for fitting three alpha parameters on rolling windows of stock price data. The script reads price CSV files, extracts the first 240 valid price observations, divides them into four fixed 60-observation blocks, normalizes each block separately into a cumulative return curve, and estimates the best-fitting parameters for each rolling window using Particle Swarm Optimization (PSO).

The purpose of this repository is to provide a clear and reproducible implementation of the fitting pipeline used in the paper. The code is focused on parameter estimation and result export; plotting and downstream analysis can be performed separately from the generated CSV files.

## Repository Structure

```text
├── main.py
└── out_core/                  # generated after running the script
    ├── all_fit_windows.csv    # generated after running the script
    └── skipped_tickers.csv    # generated only if some tickers are skipped
```

`main.py` is the main executable script. The `out_core/` directory is the default output folder created by the script.

## Main Workflow

The script performs the following steps:

1. Read one CSV file per ticker.
2. Extract the first 240 valid price observations.
3. Split these 240 observations into four fixed 60-observation blocks:
   - `BLOCK1_001_060`
   - `BLOCK2_061_120`
   - `BLOCK3_121_180`
   - `BLOCK4_181_240`
4. Normalize each 60-observation block separately into a cumulative return curve, with the first value of each block reset to 1.
5. Apply a rolling window within each block-normalized cumulative return curve.
6. Fit three parameters, `alpha1`, `alpha2`, and `alpha3`, for each rolling window.
7. Save all fitted parameters and objective values to a CSV file.

By default, each block contains 60 observations and each rolling window has length 10. Therefore, each block contains 51 rolling windows, and each ticker produces 204 fitted windows.

The rolling-window length is configurable, but each block contains only 60 observations. A window size greater than 60 produces no fitting windows.

The normalization is performed once at the block level. Rolling windows are then extracted from the already normalized block-level cumulative return curve. The rolling windows are not normalized again.

## Input Data Format

Each ticker should be stored as a separate CSV file:

```text
<TICKER>.csv
```

For example:

```text
WFF.csv
AAPL.csv
MSFT.csv
```

The script supports either a regular single-level CSV column structure or a two-level column structure such as those produced by some financial data download tools. The price column should be one of the following:

```text
Adj Close
Close
Price
```

All prices must be positive because the script computes log returns. Missing and non-finite price values are removed before processing. Each ticker must have at least 240 valid price observations. The script uses the original CSV row order and does not sort observations by date.

## Installation

Create a Python environment and install the required packages:

```bash
pip install numpy pandas torch pyswarms
```

For GPU execution, install a CUDA-enabled version of PyTorch that matches your local CUDA environment.

## Selecting Tickers

The default ticker list is defined near the top of the script:

```python
DEFAULT_TICKERS = ["WFF"]
```

To run the code on a different ticker, edit this list directly in the script.

For example, to run the code on Apple, change it to:

```python
DEFAULT_TICKERS = ["AAPL"]
```

To run the code on multiple tickers, include all target ticker symbols in the same list:

```python
DEFAULT_TICKERS = ["WFF", "AAPL", "MSFT"]
```

The script processes the tickers listed in `DEFAULT_TICKERS` when no ticker arguments are provided.

Run the script using:

```bash
python "main.py"
```

Alternatively, ticker symbols can be specified directly through the command line:

```bash
python "main.py" --tickers AAPL MSFT
```

Ticker symbols can also be read from a text file:

```bash
python "main.py" --ticker-file tickers.txt
```

The ticker file can contain symbols separated by commas or newlines. All ticker symbols are converted to uppercase, and duplicates are removed.

When `--ticker-file` is used without overriding `--tickers`, the tickers in `DEFAULT_TICKERS` are also included.

## Common Command-Line Arguments

| Argument | Default | Description |
|---|---:|---|
| `--csv-dir` | `.` | Directory containing input price CSV files. |
| `--out-dir` | `out_core` | Directory where output files are saved. |
| `--output-name` | `all_fit_windows.csv` | Name of the main output CSV file. |
| `--tickers` | `WFF` | Ticker symbols to process. |
| `--ticker-file` | `None` | Optional text file containing ticker symbols. |
| `--device` | `auto` | Computing device: `auto`, `cpu`, or `cuda`. |
| `--dtype` | `float64` | Floating-point precision: `float32` or `float64`. |
| `--window-size` | `10` | Number of observations in each rolling window. |
| `--particles` | `96` | Number of particles used by PSO. |
| `--iters` | `120` | Number of PSO iterations. |
| `--bounds` | `1000.0` | Symmetric search bounds for each alpha parameter. |
| `--smooth-window` | `3` | Moving-average window length, which must be a positive odd integer. |
| `--seed` | `12345` | Base random seed for PSO initialization. |

Note: Although both float32 and float64 are accepted as command-line options, the current implementation should use float64. The invalid-cost value is set to 1e300, which exceeds the representable range of float32 and causes an overflow error when constructing the cost tensor.

## Default Fitting Configuration

The main fitting configuration is defined in the `FitConfig` dataclass inside `main.py`.

| Parameter | Default | Description |
|---|---:|---|
| `window_size` | `10` | Number of observations in each rolling window. |
| `n_particles` | `96` | Number of particles used by PSO. |
| `n_iters` | `120` | Number of PSO iterations. |
| `alpha_bounds` | `alpha1, alpha2: [0, 1000]; alpha3: [-1000, 1000]` | Search range for each alpha parameter.|
| `smooth_window` | `3` | Moving-average window used to smooth fitted and observed curves in the objective function. |
| `trend_scale_floor` | `5e-3` | Minimum scale used to avoid unstable loss values when the target curve has very small variation. |
| `raw_weight` | `0.45` | Weight of the raw pointwise fitting error. |
| `trend_corr_weight` | `0.10` | Weight of the smoothed trend correlation loss. |
| `trend_diff_weight` | `0.25` | Weight of the adjacent-difference shape loss. |
| `trend_end_weight` | `0.10` | Weight of the endpoint fitting loss. |
| `trend_dir_weight` | `0.10` | Weight of the overall direction mismatch penalty. |
| `base_seed` | `12345` | Base random seed used for PSO initialization. |
| `invalid_cost` | `1e300` | Penalty assigned to invalid parameter candidates. |
| `eps_value` | `1e-12` | Numerical tolerance used in validity checks and correlation calculations. |
| `exp_limit` | `700.0` | Maximum allowed absolute exponential argument for valid candidates. |
| `pso_options` | `c1=1.49, c2=1.49, w=0.72` | PSO cognitive coefficient, social coefficient, and inertia weight. |

The objective-function weights sum to 1.00:



```text
0.45 + 0.10 + 0.25 + 0.10 + 0.10 = 1.00
```

## Mathematical Procedure

### 1. Price Preprocessing and Block-Level Normalization

For each ticker, the script reads the price sequence:

$$ P_1, P_2, P_3, \ldots, P_T $$

The code removes missing and non-finite values and requires all remaining prices to be positive.

The first 240 valid observations are then divided into four fixed 60-observation blocks:

$$
B_1 = \left(P_1, P_2, \ldots, P_{60}\right)
$$

$$
B_2 = \left(P_{61}, P_{62}, \ldots, P_{120}\right)
$$

$$
B_3 = \left(P_{121}, P_{122}, \ldots, P_{180}\right)
$$

$$
B_4 = \left(P_{181}, P_{182}, \ldots, P_{240}\right)
$$

Each block is normalized separately. This means that the first observation of every 60-observation block is reset to 1.

For a given 60-observation block $B_b$, let the raw prices inside that block be:

$$ P_{b,1}, P_{b,2}, P_{b,3}, \ldots, P_{b,60} $$

The log return inside the block is computed as:

$$ r_{b,t} = \log\left(\frac{P_{b,t}}{P_{b,t-1}}\right), \qquad t = 2, 3, \ldots, 60 $$

The block-normalized cumulative return curve is then defined as:

$$ C_{b,1} = 1 $$

and

$$ C_{b,t} = \exp\left(\sum_{k=2}^{t} r_{b,k}\right), \qquad t = 2, 3, \ldots, 60 $$

Equivalently, because the cumulative sum of log returns equals the log price ratio within the same block:

$$ C_{b,t} = \frac{P_{b,t}}{P_{b,1}}, \qquad t = 1, 2, \ldots, 60 $$

Therefore, each 60-observation block has its own normalized cumulative return curve:

$$ C_b = \left(C_{b,1}, C_{b,2}, \ldots, C_{b,60}\right) $$

with:

$$ C_{b,1} = 1 $$

This block-level normalization is applied once per 60-observation block. The rolling windows are then taken from the already normalized block-level curve. The first value of each rolling window is not separately reset to 1 unless that window starts at the first observation of the block.

### 2. Block Construction and Rolling Windows

After block-level normalization, the four target curves are:

$$
C_1, C_2, C_3, C_4
$$

Each normalized block contains 60 observations. The default rolling-window length is:

$$
W = 10
$$

Therefore, the number of rolling windows in each block is:

$$
60 - W + 1 = 51
$$

For each ticker, the total number of fitted windows is:

$$
4 \times 51 = 204
$$

### 3. Rolling-Window Target Curve

For each rolling window, the observed target curve is:

$$
y = \left(y_0, y_1, \ldots, y_{W-1}\right)
$$

where \(W = 10\) by default.

The corresponding time index is:

$$
t = 0, 1, 2, \ldots, W-1
$$

The first two observed values are used to define the initial value and an approximation of the initial slope. The initial value is set to $p_0 = y_0$, while the initial slope is approximated by the forward difference $p_1 = y_1 - y_0$.

### 4. Fitted Curve Definition

For each rolling window, the script fits three parameters:

$$
\alpha_1, \alpha_2, \alpha_3
$$

Given a candidate parameter vector:

$$
\theta = \left(\alpha_1, \alpha_2, \alpha_3\right)
$$

the model first computes the discriminant:

$$
D = \alpha_2^2 - 4\alpha_1\alpha_3
$$

The candidate must satisfy the following conditions:

$$
|\alpha_1| > \epsilon
$$

$$
|\alpha_3| > \epsilon
$$

$$
D > 0
$$

$$
\sqrt{D} > \epsilon
$$

where $\epsilon = 10^{-12}$. The implementation also requires finite intermediate values and exponential arguments with absolute values below 700.

The two characteristic roots are then computed as:

$$
\lambda_{+} =
\frac{-\alpha_2 + \sqrt{D}}{2\alpha_3}
$$

and

$$
\lambda_{-} =
\frac{-\alpha_2 - \sqrt{D}}{2\alpha_3}
$$

The fitted curve has the form:

$$
\hat{y}(t) = -\frac{1}{\alpha_1}
+
c_{-}e^{\lambda_{-}t}
+
c_{+}e^{\lambda_{+}t}
$$

The constants $c_{+}$ and $c_{-}$ are determined from the initial value and the initial slope. The initial conditions are:

$$
p_0 = y_0
$$

and

$$
p_1 = y_1 - y_0
$$

The fitted curve is constructed to satisfy:

$$
\hat{y}(0) = p_0
$$

and

$$
\hat{y}'(0) = p_1
$$

Let:

$$
A = p_0 + \frac{1}{\alpha_1}
$$

and

$$
R = \sqrt{D}
$$

The two coefficients are:

$$
c_{-} =
-\frac{2\alpha_3 p_1 + A(\alpha_2 - R)}{2R}
$$

and

$$
c_{+} =
\frac{2\alpha_3 p_1 + A(\alpha_2 + R)}{2R}
$$

The coefficient formulas require a nonzero value of $R$. The implementation enforces $R > \epsilon$ to avoid division by zero and numerically unstable parameter combinations.

Candidates that fail the validity checks are assigned an invalid cost of $10^{300}$. The PSO algorithm minimizes the resulting objective values, but the returned best parameter vector is not independently revalidated after optimization.

### 5. Smoothing

The objective function uses both raw and smoothed curves. The smoothed curve is computed using a moving average with an odd window length. The default smoothing window is 3.

For a curve \(x_t\), the smoothed version is denoted by:

$$
\tilde{x}_t
$$

The implementation uses replicated padding at the boundaries before applying the moving average.

The observed curve and fitted curve are smoothed as:

$$
\tilde{y}_t
$$

and

$$
\tilde{\hat{y}}_t
$$

### 6. Scale Normalization

To avoid unstable loss values when the observed curve has very small variation, the script defines a scale term:

$$
s = \max\left(\max_t \tilde{y}_t - \min_t \tilde{y}_t,\ s_{\min}\right)
$$

where the default minimum scale is:

$$
s_{\min} = 5 \times 10^{-3}
$$

All error terms based on value differences are normalized by this scale.

## Objective Function

The objective function is designed to evaluate both pointwise fitting accuracy and trend consistency. For each candidate fitted curve \(\hat{y}\), the total objective value is:

$$
L = 0.45L_{\mathrm{raw}}
+
0.10L_{\mathrm{corr}}
+
0.25L_{\mathrm{diff}}
+
0.10L_{\mathrm{end}}
+
0.10L_{\mathrm{dir}}
$$

The PSO algorithm searches for the parameter vector:

$$
\theta^* = \arg\min_{\theta}
L(\theta)
$$

where:

$$
\theta = \left(\alpha_1, \alpha_2, \alpha_3\right)
$$

### 1. Raw Pointwise Error

The raw pointwise loss measures the direct difference between the fitted curve and the observed curve:

$$
L_{\mathrm{raw}} = \frac{1}{W}
\sum_{t=0}^{W-1}
\left(
\frac{\hat{y}_t - y_t}{s}
\right)^2
$$

This term measures direct fitting accuracy at each time point.

### 2. Smoothed Correlation Loss

The smoothed correlation loss is included to measure the similarity in the overall shape of the fitted curve and the observed curve after short-term fluctuations have been reduced by smoothing. While the raw pointwise loss evaluates the absolute distance between the fitted and observed values at each time point, the correlation term evaluates whether the two curves move in a similar pattern over the rolling window.

Let the observed curve be denoted by:

$$ y = \left(y_0, y_1, \ldots, y_{W-1}\right) $$

and let the fitted curve be denoted by:

$$ \hat{y} = \left(\hat{y}_0, \hat{y}_1, \ldots, \hat{y}_{W-1}\right) $$

Before computing the correlation, both curves are smoothed using the moving-average procedure described above. The smoothed observed curve is denoted by:

$$ \tilde{y} = \left(\tilde{y}_0, \tilde{y}_1, \ldots, \tilde{y}_{W-1}\right) $$

and the smoothed fitted curve is denoted by:

$$ \tilde{\hat{y}} = \left(\tilde{\hat{y}}_0, \tilde{\hat{y}}_1, \ldots, \tilde{\hat{y}}_{W-1}\right) $$

The mean values of the two smoothed curves are:

$$ \bar{\tilde{y}} = \frac{1}{W}\sum_{t=0}^{W-1}\tilde{y}_t $$

and

$$ \bar{\tilde{\hat{y}}} = \frac{1}{W}\sum_{t=0}^{W-1}\tilde{\hat{y}}_t $$

The centered smoothed curves are then:

$$ \tilde{y}_t - \bar{\tilde{y}} $$

and

$$ \tilde{\hat{y}}_t - \bar{\tilde{\hat{y}}} $$

The Pearson correlation coefficient between the two smoothed curves is computed as:

$$ \rho =
\frac{
\sum_{t=0}^{W-1}
\left(\tilde{\hat{y}}_t - \bar{\tilde{\hat{y}}}\right)
\left(\tilde{y}_t - \bar{\tilde{y}}\right)
}{
\sqrt{
\left[
\sum_{t=0}^{W-1}
\left(\tilde{\hat{y}}_t - \bar{\tilde{\hat{y}}}\right)^2
\right]
\left[
\sum_{t=0}^{W-1}
\left(\tilde{y}_t - \bar{\tilde{y}}\right)^2
\right]
+
\epsilon
}
+
\epsilon
} $$

Here the small constant is set to:

$$ \epsilon = 10^{-12} $$

The role of $\epsilon$ is numerical stabilization. If either the smoothed observed curve or the smoothed fitted curve has almost no variation, the denominator of the Pearson correlation can become zero or extremely small. Adding $\epsilon$ prevents division by zero and reduces the risk of producing unstable numerical values.

After the correlation is computed, it is clipped to the interval:

$$ -1 \leq \rho \leq 1 $$

This follows the natural range of the Pearson correlation coefficient and prevents small numerical rounding errors from producing values slightly outside the valid interval.

The correlation loss is then defined as:

$$ L_{\mathrm{corr}} = 1 - \rho $$

This transformation is used because the optimization procedure minimizes the objective function. A high positive correlation indicates that the smoothed fitted curve and the smoothed observed curve move in a similar direction and have a similar shape. In that case, $\rho$ is close to 1 and the loss becomes close to 0:

$$ \rho \approx 1 \quad \Rightarrow \quad L_{\mathrm{corr}} \approx 0 $$

If the two smoothed curves are weakly related, then $\rho$ is close to 0 and the loss is close to 1:

$$ \rho \approx 0 \quad \Rightarrow \quad L_{\mathrm{corr}} \approx 1 $$

If the two smoothed curves move in opposite directions, then $\rho$ becomes negative and the loss becomes larger than 1:

$$ \rho < 0 \quad \Rightarrow \quad L_{\mathrm{corr}} > 1 $$

Therefore, this term rewards fitted curves that preserve the smoothed trend of the observed curve and penalizes fitted curves that move in an inconsistent or opposite direction.

The main purpose of this component is to supplement the raw pointwise error. A fitted curve may have a relatively small pointwise error but still fail to reproduce the overall trajectory of the observed data. By including the smoothed correlation loss, the objective function encourages not only numerical closeness but also shape consistency. This is especially useful for short rolling windows, where local noise may affect individual observations but the general trend remains important for evaluating the quality of the fitted curve.


### 3. Adjacent-Difference Shape Loss

The adjacent-difference shape loss is used to compare the local changes of the smoothed fitted curve and the smoothed observed curve. While the smoothed correlation loss measures whether the two curves have a similar overall shape, the adjacent-difference loss focuses more directly on whether the fitted curve reproduces the step-by-step changes of the observed trajectory.

After smoothing, the local change of the observed curve between two consecutive time points is defined as:

$$ d_t = \tilde{y}_{t+1} - \tilde{y}_t, \qquad t = 0, 1, \ldots, W-2 $$

Similarly, the local change of the fitted curve is defined as:

$$ \hat{d}_t = \tilde{\hat{y}}_{t+1} - \tilde{\hat{y}}_t, \qquad t = 0, 1, \ldots, W-2 $$

These adjacent differences describe the local slope or short-term movement of each curve. A positive value indicates an upward movement between two consecutive points, while a negative value indicates a downward movement.

The adjacent-difference shape loss is then computed as:

$$ L_{\mathrm{diff}} =
\frac{1}{W-1}
\sum_{t=0}^{W-2}
\left(
\frac{\hat{d}_t - d_t}{s}
\right)^2
$$

where $s$ is the scale-normalization term defined above. Dividing by $s$ makes the loss comparable across rolling windows with different levels of variation. Without this normalization, windows with larger price movements would naturally produce larger difference errors, even if the relative fitting quality were similar.

This term penalizes discrepancies between the local movements of the fitted and observed curves. If the observed curve increases from one time point to the next, the fitted curve is encouraged to show a similar increase. If the observed curve decreases, the fitted curve is encouraged to reproduce a similar decrease. Therefore, this loss helps the model capture local shape information rather than only matching the overall level of the curve.

The adjacent-difference loss complements both the raw pointwise loss and the smoothed correlation loss. The raw pointwise loss measures direct value differences, and the smoothed correlation loss measures global shape similarity. In contrast, the adjacent-difference loss emphasizes local directional and slope consistency. This is useful because two curves may have similar overall correlation but still differ in their short-term movements. By including this term, the objective function encourages the fitted curve to follow the observed trajectory more faithfully at the local level.

### 4. Endpoint Loss

The endpoint loss measures whether the fitted curve ends close to the final observed value in the rolling window. For a rolling window of length $W$, the final observed value is $y_{W-1}$, and the final fitted value is $\hat{y}_{W-1}$.

The endpoint loss is defined as:

$$ L_{\mathrm{end}} = \left(\frac{\hat{y}_{W-1} - y_{W-1}}{s}\right)^2 $$

where $s$ is the scale-normalization term defined above. Dividing by $s$ makes the endpoint error comparable across windows with different levels of price variation.

This term is included because the final value is directly related to the net movement of the rolling window. In addition to matching the overall curve shape, the fitting procedure should also reflect whether the trajectory ultimately ends higher or lower. If the fitted curve follows the middle part of the observed curve but ends at a noticeably different level, it may give an inaccurate representation of the final upward or downward movement.

By penalizing endpoint deviations, this term encourages the fitted curve to preserve the terminal level of the observed trajectory and improves consistency with the final direction of movement within the window.

### 5. Direction Mismatch Penalty

The direction mismatch penalty evaluates whether the fitted curve and the observed curve have the same overall direction of movement within the rolling window. This term focuses on the net change from the beginning to the end of the window, rather than on pointwise fitting accuracy.

The observed net movement is defined as:

$$ D_y = y_{W-1} - y_0 $$

The fitted net movement is defined as:

$$ D_{\hat{y}} = \hat{y}_{W-1} - \hat{y}_0 $$

Here, $D_y$ indicates whether the observed trajectory ends higher or lower than its initial value, while $D_{\hat{y}}$ gives the corresponding direction for the fitted trajectory.

To avoid treating very small numerical fluctuations as meaningful upward or downward movements, the script defines a small threshold:

$$ \tau = 0.25s_{\min} $$

Using the default value $s_{\min} = 5 \times 10^{-3}$, this becomes:

$$ \tau = 1.25 \times 10^{-3} $$

The direction mismatch penalty is then defined as:

$$ L_{\mathrm{dir}} =
\begin{cases}
1, & \text{if } D_y > \tau \text{ and } D_{\hat{y}} < -\tau \\
1, & \text{if } D_y < -\tau \text{ and } D_{\hat{y}} > \tau \\
0, & \text{otherwise}
\end{cases}
$$

This means that a penalty is applied only when the observed curve shows a clear upward movement but the fitted curve shows a clear downward movement, or when the observed curve shows a clear downward movement but the fitted curve shows a clear upward movement. If either movement is smaller than the threshold, it is not treated as a strong directional signal.

This term is included because the final upward or downward direction of the rolling window is an important part of the fitting target. A fitted curve may have a reasonable pointwise error, but if it predicts the opposite overall direction, the fitted trajectory can be misleading. By adding this binary penalty, the objective function discourages solutions that move against the observed net direction.

## PSO Optimization

For each rolling window, the script uses Particle Swarm Optimization (PSO) to search for the parameter vector that minimizes the total objective function. Each particle represents one candidate set of the three fitted parameters:

$$ \theta = \left(\alpha_1, \alpha_2, \alpha_3\right) $$

The optimization problem for each rolling window is:

$$ \theta^* = \arg\min_{\theta} L(\theta) $$

where $L(\theta)$ is the total objective function defined above.

By default, each alpha parameter is searched within the symmetric range:

$$ \alpha_i \in [-1000, 1000], \qquad i = 1,2,3 $$

The PSO procedure uses 96 particles and 120 iterations by default. Each particle has a position and a velocity in the three-dimensional parameter space. The position of particle $i$ at iteration $k$ is denoted by:

$$ x_i^{(k)} = \left(\alpha_{1,i}^{(k)}, \alpha_{2,i}^{(k)}, \alpha_{3,i}^{(k)}\right) $$

and the corresponding velocity is denoted by:

$$ v_i^{(k)} = \left(v_{1,i}^{(k)}, v_{2,i}^{(k)}, v_{3,i}^{(k)}\right) $$

During the search, each particle keeps track of its own best previous position, denoted by $p_i^{(k)}$. The swarm also keeps track of the best position found by all particles, denoted by $g^{(k)}$. At each iteration, the velocity of each particle is updated using three components: the previous velocity, the attraction toward its own best position, and the attraction toward the global best position:

$$ v_i^{(k+1)} = wv_i^{(k)} + c_1r_1^{(k)}\left(p_i^{(k)} - x_i^{(k)}\right) + c_2r_2^{(k)}\left(g^{(k)} - x_i^{(k)}\right) $$

The particle position is then updated as:

$$ x_i^{(k+1)} = x_i^{(k)} + v_i^{(k+1)} $$

The default coefficients are:

$$ w = 0.72, \qquad c_1 = 1.49, \qquad c_2 = 1.49 $$

Here, $w$ is the inertia weight, which controls how much of the previous velocity is retained. The coefficient $c_1$ controls the particle's tendency to return to its own best previous solution, while $c_2$ controls its tendency to move toward the best solution found by the whole swarm. The random factors $r_1^{(k)}$ and $r_2^{(k)}$ introduce stochastic exploration during the search.

For each candidate parameter vector, the script generates the fitted curve:

$$
\hat{y}(t) = -\frac{1}{\alpha_1} + c_-e^{\lambda_-t} + c_+e^{\lambda_+t}
$$

where $\lambda_+$, $\lambda_-$, $c_+$, and $c_-$ are computed from the candidate alpha parameters and the initial conditions $p_0 = y_0$ and $p_1 = y_1 - y_0$.

The fitted curve is then evaluated using the total objective function:

$$ L = 0.45L_{\mathrm{raw}} + 0.10L_{\mathrm{corr}} + 0.25L_{\mathrm{diff}} + 0.10L_{\mathrm{end}} + 0.10L_{\mathrm{dir}} $$

During optimization, particles that produce lower objective values update their personal-best positions. If a particle produces the lowest objective value among all particles, the global-best position is updated. Through repeated movement and evaluation, the swarm searches for the parameter vector that best fits the observed rolling-window trajectory.

At the end of the optimization, the best parameter vector is returned as:

$$ \theta^{\star} = \left(\alpha_{1}^{\star}, \alpha_{2}^{\star}, \alpha_{3}^{\star}\right) $$

The corresponding best objective value found by PSO is:

$$ L\left(\theta^{\star}\right) $$

These values are saved in the output CSV file as `alpha1`, `alpha2`, `alpha3`, and `objective_error`.

## Output Files

The main output file is:

```text
out_core/all_fit_windows.csv
```

It contains one row per ticker, block, and rolling window.

The output columns are:

| Column | Description |
|---|---|
| `ticker` | Stock ticker symbol. |
| `block` | Block label. |
| `start_idx_1based_in_block` | Rolling-window start index inside the block, using one-based indexing. |
| `alpha1` | Fitted value of the first alpha parameter. |
| `alpha2` | Fitted value of the second alpha parameter. |
| `alpha3` | Fitted value of the third alpha parameter. |
| `objective_error` | Final objective value returned by PSO. |

If any ticker cannot be processed, the script writes a separate report:

```text
out_core/skipped_tickers.csv
```

This file contains the ticker symbol and the reason why it was skipped.

The skipped-ticker report is written only when at least one ticker fails during the current run. The script does not automatically remove an existing skipped_tickers.csv file from previous runs.

## Reproducibility

The script uses a base random seed to make the PSO fitting procedure more reproducible. The default base seed is:

$$ \mathrm{base\ seed} = 12345 $$

For each rolling window, the actual seed is generated by adding the zero-based window start index within the block to the configured base seed:

$$
\mathrm{seed}_{j} = \mathrm{base\_seed} + j
$$

where $j$ is the zero-based start index of the rolling window within its block. With the default base seed of 12345, this becomes $\mathrm{seed}_{j} = 12345 + j$. Therefore, different start positions within the same block use different random seeds. Windows with the same start index in different blocks use the same seed offset.

This makes repeated runs easier to reproduce because the same ticker, block, and rolling-window start index will use the same random seed when the script is run again with the same settings. The seed depends only on the rolling-window start index within each block, so different blocks and different tickers may use the same seed.

However, exact reproducibility may still depend on the software environment, hardware, numerical precision, and computing device. In particular, CPU and GPU runs may produce small numerical differences because floating-point operations are not always executed in exactly the same order. For the most consistent results, use the same input data, package versions, device setting, dtype setting, and random seed.
