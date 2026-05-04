import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


DEFAULT_TICKERS = ["WFF"]
DEFAULT_BLOCKS = [
    ("BLOCK1_001_060", 0, 60),
    ("BLOCK2_061_120", 60, 120),
    ("BLOCK3_121_180", 120, 180),
    ("BLOCK4_181_240", 180, 240),
]


@dataclass(frozen=True)
class FitConfig:
    csv_dir: Path = Path(".")
    out_dir: Path = Path("out_core")
    output_name: str = "all_fit_windows.csv"
    tickers: Tuple[str, ...] = tuple(DEFAULT_TICKERS)
    blocks: Tuple[Tuple[str, int, int], ...] = tuple(DEFAULT_BLOCKS)
    window_size: int = 10
    dtype: torch.dtype = torch.float64
    invalid_cost: float = 1e300
    eps_value: float = 1e-12
    exp_limit: float = 700.0
    n_particles: int = 96
    n_iters: int = 120
    pso_options: Dict[str, float] = field(
        default_factory=lambda: {"c1": 1.49, "c2": 1.49, "w": 0.72}
    )
    alpha_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (-1000.0, 1000.0),
        (-1000.0, 1000.0),
        (-1000.0, 1000.0),
    )
    smooth_window: int = 3
    trend_scale_floor: float = 5e-3
    raw_weight: float = 0.45
    trend_corr_weight: float = 0.10
    trend_diff_weight: float = 0.25
    trend_end_weight: float = 0.10
    trend_dir_weight: float = 0.10
    base_seed: int = 12345

    @property
    def output_csv(self) -> Path:
        return self.out_dir / self.output_name

    @property
    def skipped_csv(self) -> Path:
        return self.out_dir / "skipped_tickers.csv"


def parse_tickers(values: Sequence[str], ticker_file: Optional[Path] = None) -> Tuple[str, ...]:
    tickers: List[str] = []

    for value in values:
        for item in value.replace("\n", ",").split(","):
            item = item.strip().upper()
            if item:
                tickers.append(item)

    if ticker_file is not None:
        text = ticker_file.read_text(encoding="utf-8")
        for item in text.replace("\n", ",").split(","):
            item = item.strip().upper()
            if item:
                tickers.append(item)

    unique_tickers = tuple(dict.fromkeys(tickers))
    if not unique_tickers:
        raise ValueError("No tickers were provided.")
    return unique_tickers


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def get_bounds_arrays(config: FitConfig) -> Tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray(config.alpha_bounds, dtype=np.float64)
    return bounds[:, 0].copy(), bounds[:, 1].copy()


def find_price_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        for name in ("Adj Close", "Close", "Price"):
            if (name, ticker) in df.columns:
                return df[(name, ticker)]

        valid_names = {"adj close", "close", "price"}
        candidates = [col for col in df.columns if str(col[0]).strip().lower() in valid_names]
        if len(candidates) == 1:
            return df[candidates[0]]
        raise ValueError(f"[{ticker}] ambiguous or missing price column")

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    for name in ("Adj Close", "Close", "Price"):
        if name in df.columns:
            return df[name]
    raise ValueError(f"[{ticker}] no Adj Close, Close, or Price column found")


def read_price_dataframe(csv_path: Path, ticker: str) -> pd.DataFrame:
    load_errors: List[str] = []

    for header in (0, [0, 1]):
        try:
            df = pd.read_csv(csv_path, header=header)
            find_price_series(df, ticker)
            return df
        except Exception as exc:  # noqa: BLE001
            load_errors.append(str(exc))

    joined_errors = " | ".join(load_errors)
    raise ValueError(f"[{ticker}] failed to read price column from {csv_path}: {joined_errors}")


def load_price_array(ticker: str, config: FitConfig) -> np.ndarray:
    csv_path = config.csv_dir / f"{ticker}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"[{ticker}] CSV not found: {csv_path}")

    df = read_price_dataframe(csv_path, ticker)
    prices = pd.to_numeric(find_price_series(df, ticker), errors="coerce")
    prices = prices.dropna().to_numpy(dtype=np.float64)
    prices = prices[np.isfinite(prices)]

    required_points = max(end for _, _, end in config.blocks)
    if prices.size < required_points:
        raise ValueError(
            f"[{ticker}] insufficient valid price points: "
            f"need {required_points}, got {prices.size}"
        )
    if np.any(prices <= 0.0):
        raise ValueError(f"[{ticker}] price must be positive to compute log returns")
    return prices


def price_to_normalized_cumulative_return(price_arr: np.ndarray) -> np.ndarray:
    price_arr = np.asarray(price_arr, dtype=np.float64)
    if price_arr.size < 2:
        raise ValueError("Need at least two price points.")
    if not np.all(np.isfinite(price_arr)):
        raise ValueError("Price array contains non-finite values.")
    if np.any(price_arr <= 0.0):
        raise ValueError("Price values must be positive to compute log returns.")

    log_returns = np.log(price_arr[1:] / price_arr[:-1])
    cumulative_return = np.empty(price_arr.size, dtype=np.float64)
    cumulative_return[0] = 1.0
    cumulative_return[1:] = np.exp(np.cumsum(log_returns))

    if not np.all(np.isfinite(cumulative_return)):
        raise ValueError("Normalized cumulative return contains non-finite values.")
    return cumulative_return


def smooth_batch(x: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return x
    if window % 2 == 0:
        raise ValueError("The smoothing window must be odd.")

    pad = window // 2
    x3 = x.unsqueeze(1)
    xpad = F.pad(x3, (pad, pad), mode="replicate")
    kernel = torch.ones((1, 1, window), device=x.device, dtype=x.dtype) / float(window)
    return F.conv1d(xpad, kernel).squeeze(1)


def smooth_vector(x: torch.Tensor, window: int) -> torch.Tensor:
    return smooth_batch(x.reshape(1, -1), window=window).reshape(-1)


def target_scale(values: torch.Tensor, config: FitConfig) -> torch.Tensor:
    scale = torch.max(values) - torch.min(values)
    floor = torch.tensor(config.trend_scale_floor, device=values.device, dtype=values.dtype)
    return torch.clamp(scale, min=floor)


def batch_corrcoef(x: torch.Tensor, y: torch.Tensor, config: FitConfig) -> torch.Tensor:
    eps = torch.tensor(config.eps_value, device=x.device, dtype=x.dtype)
    xm = x - torch.mean(x, dim=1, keepdim=True)
    ym = y - torch.mean(y)
    numerator = torch.sum(xm * ym[None, :], dim=1)
    denominator = torch.sqrt(torch.sum(xm * xm, dim=1) * torch.sum(ym * ym) + eps)
    return torch.clamp(numerator / (denominator + eps), min=-1.0, max=1.0)


@torch.no_grad()
def c_alpha123_batch(
    alpha1: torch.Tensor,
    alpha2: torch.Tensor,
    alpha3: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
    t: torch.Tensor,
    config: FitConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = alpha1.device
    dtype = alpha1.dtype

    alpha1 = alpha1.reshape(-1).to(device=device, dtype=dtype)
    alpha2 = alpha2.reshape(-1).to(device=device, dtype=dtype)
    alpha3 = alpha3.reshape(-1).to(device=device, dtype=dtype)
    t = torch.as_tensor(t, device=device, dtype=dtype)
    p0 = torch.as_tensor(p0, device=device, dtype=dtype).reshape(1)
    p1 = torch.as_tensor(p1, device=device, dtype=dtype).reshape(1)
    eps = torch.tensor(config.eps_value, device=device, dtype=dtype)

    valid = torch.isfinite(alpha1) & torch.isfinite(alpha2) & torch.isfinite(alpha3)
    valid = valid & (torch.abs(alpha1) > eps) & (torch.abs(alpha3) > eps)

    alpha1_safe = torch.where(valid, alpha1, torch.ones_like(alpha1))
    alpha2_safe = torch.where(valid, alpha2, torch.zeros_like(alpha2))
    alpha3_safe = torch.where(valid, alpha3, torch.ones_like(alpha3))

    one_over_alpha1 = 1.0 / alpha1_safe
    radical_inside = (alpha2_safe * alpha2_safe - 4.0 * alpha1_safe) / (
        alpha3_safe * alpha3_safe
    )

    valid = valid & torch.isfinite(one_over_alpha1)
    valid = valid & torch.isfinite(radical_inside) & (radical_inside >= 0.0)

    radical = torch.sqrt(torch.where(valid, radical_inside, torch.zeros_like(radical_inside)))
    minus_alpha2_over_alpha3 = -alpha2_safe / alpha3_safe
    lambda_plus = 0.5 * (minus_alpha2_over_alpha3 + radical)
    lambda_minus = 0.5 * (minus_alpha2_over_alpha3 - radical)
    valid = valid & torch.isfinite(lambda_plus) & torch.isfinite(lambda_minus)

    exp_plus_1 = torch.exp(
        torch.clamp(lambda_plus, min=-config.exp_limit, max=config.exp_limit)
    )
    exp_minus_1 = torch.exp(
        torch.clamp(lambda_minus, min=-config.exp_limit, max=config.exp_limit)
    )
    denom = exp_plus_1 - exp_minus_1
    valid = valid & torch.isfinite(denom) & (torch.abs(denom) > eps)

    denom_safe = torch.where(valid, denom, torch.ones_like(denom))
    p0_minus = p0 - one_over_alpha1
    p1_minus = p1 - one_over_alpha1

    c_plus = (p1_minus - p0_minus * exp_minus_1) / denom_safe
    c_minus = (p0_minus * exp_plus_1 - p1_minus) / denom_safe
    valid = valid & torch.isfinite(c_plus) & torch.isfinite(c_minus)

    expo_plus_t = lambda_plus[:, None] * t[None, :]
    expo_minus_t = lambda_minus[:, None] * t[None, :]
    valid = valid & (torch.abs(expo_plus_t).amax(dim=1) < config.exp_limit)
    valid = valid & (torch.abs(expo_minus_t).amax(dim=1) < config.exp_limit)

    exp_plus_t = torch.exp(
        torch.clamp(expo_plus_t, min=-config.exp_limit, max=config.exp_limit)
    )
    exp_minus_t = torch.exp(
        torch.clamp(expo_minus_t, min=-config.exp_limit, max=config.exp_limit)
    )

    c_values = one_over_alpha1[:, None] + c_plus[:, None] * exp_plus_t + c_minus[:, None] * exp_minus_t
    valid = valid & torch.isfinite(c_values).all(dim=1)
    return c_values, valid


@torch.no_grad()
def objective_metric_batch(
    c_hat: torch.Tensor,
    c_target: torch.Tensor,
    config: FitConfig,
) -> torch.Tensor:
    c_target = c_target.reshape(-1)
    c_target_smooth = smooth_vector(c_target, window=config.smooth_window)
    c_hat_smooth = smooth_batch(c_hat, window=config.smooth_window)

    scale = target_scale(c_target_smooth, config)
    raw_loss = torch.mean(((c_hat - c_target[None, :]) / scale) ** 2, dim=1)

    corr_loss = 1.0 - batch_corrcoef(c_hat_smooth, c_target_smooth, config)

    target_diff = c_target_smooth[1:] - c_target_smooth[:-1]
    pred_diff = c_hat_smooth[:, 1:] - c_hat_smooth[:, :-1]
    diff_loss = torch.mean(((pred_diff - target_diff[None, :]) / scale) ** 2, dim=1)

    end_loss = ((c_hat[:, -1] - c_target[-1]) / scale) ** 2

    target_net = c_target[-1] - c_target[0]
    pred_net = c_hat[:, -1] - c_hat[:, 0]
    small = torch.tensor(
        0.25 * config.trend_scale_floor,
        device=c_hat.device,
        dtype=c_hat.dtype,
    )
    mismatch = ((target_net > small) & (pred_net < -small)) | (
        (target_net < -small) & (pred_net > small)
    )
    direction_loss = mismatch.to(dtype=c_hat.dtype)

    return (
        config.raw_weight * raw_loss
        + config.trend_corr_weight * corr_loss
        + config.trend_diff_weight * diff_loss
        + config.trend_end_weight * end_loss
        + config.trend_dir_weight * direction_loss
    )


def make_objective(
    c_target: torch.Tensor,
    t: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
    device: torch.device,
    config: FitConfig,
) -> Callable[[np.ndarray], np.ndarray]:
    c_target = torch.as_tensor(c_target, device=device, dtype=config.dtype)
    t = torch.as_tensor(t, device=device, dtype=config.dtype)
    p0 = torch.as_tensor(p0, device=device, dtype=config.dtype)
    p1 = torch.as_tensor(p1, device=device, dtype=config.dtype)

    def objective(x: np.ndarray) -> np.ndarray:
        x_t = torch.as_tensor(x, device=device, dtype=config.dtype)
        alpha1, alpha2, alpha3 = x_t.T
        c_hat, valid = c_alpha123_batch(alpha1, alpha2, alpha3, p0, p1, t, config)

        cost = torch.full(
            (x_t.shape[0],),
            config.invalid_cost,
            device=device,
            dtype=config.dtype,
        )
        if bool(valid.any().item()):
            cost[valid] = objective_metric_batch(c_hat[valid], c_target, config).to(
                dtype=config.dtype
            )
        return cost.detach().cpu().numpy()

    return objective


def run_pso(
    c_target: torch.Tensor,
    t: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
    device: torch.device,
    seed: int,
    config: FitConfig,
) -> Tuple[np.ndarray, float]:
    try:
        import pyswarms as ps
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: pyswarms. Install it with `pip install pyswarms`."
        ) from exc

    objective = make_objective(c_target, t, p0, p1, device, config)
    lower_bounds, upper_bounds = get_bounds_arrays(config)

    np.random.seed(int(seed))
    optimizer = ps.single.GlobalBestPSO(
        n_particles=int(config.n_particles),
        dimensions=3,
        options=config.pso_options,
        bounds=(lower_bounds, upper_bounds),
    )

    best_cost, best_position = optimizer.optimize(
        objective,
        iters=int(config.n_iters),
        verbose=False,
    )
    return np.asarray(best_position, dtype=np.float64).copy(), float(best_cost)


def prepare_window(
    block_values: np.ndarray,
    start_idx: int,
    device: torch.device,
    config: FitConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values = block_values[start_idx : start_idx + config.window_size]
    c_target = torch.tensor(values, device=device, dtype=config.dtype)
    t = torch.arange(config.window_size, device=device, dtype=config.dtype)
    return c_target, t, c_target[0], c_target[1]


def fit_ticker(ticker: str, device: torch.device, config: FitConfig) -> List[Dict[str, object]]:
    prices = load_price_array(ticker, config)
    rows: List[Dict[str, object]] = []

    for block_label, block_start, block_end in config.blocks:
        block_prices = prices[block_start:block_end]
        block_values = price_to_normalized_cumulative_return(block_prices)
        n_windows = len(block_values) - config.window_size + 1

        for start_idx in range(n_windows):
            c_target, t, p0, p1 = prepare_window(block_values, start_idx, device, config)
            best_position, best_cost = run_pso(
                c_target,
                t,
                p0,
                p1,
                device,
                seed=config.base_seed + start_idx,
                config=config,
            )

            rows.append(
                {
                    "ticker": ticker,
                    "block": block_label,
                    "start_idx_1based_in_block": start_idx + 1,
                    "alpha1": float(best_position[0]),
                    "alpha2": float(best_position[1]),
                    "alpha3": float(best_position[2]),
                    "objective_error": best_cost,
                }
            )

    return rows


def save_results(
    rows: Sequence[Dict[str, object]],
    skipped: Sequence[Tuple[str, str]],
    config: FitConfig,
) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.output_csv, index=False)

    if skipped:
        pd.DataFrame(skipped, columns=["ticker", "reason"]).to_csv(
            config.skipped_csv,
            index=False,
        )


def build_config(args: argparse.Namespace) -> FitConfig:
    if args.bounds <= 0:
        raise ValueError("--bounds must be positive.")
    if args.window_size < 2:
        raise ValueError("--window-size must be at least 2.")
    if args.smooth_window < 1 or args.smooth_window % 2 == 0:
        raise ValueError("--smooth-window must be a positive odd integer.")

    tickers = parse_tickers(args.tickers, args.ticker_file)
    dtype = dtype_from_name(args.dtype)
    bounds = (
        (-args.bounds, args.bounds),
        (-args.bounds, args.bounds),
        (-args.bounds, args.bounds),
    )

    return FitConfig(
        csv_dir=args.csv_dir,
        out_dir=args.out_dir,
        output_name=args.output_name,
        tickers=tickers,
        window_size=args.window_size,
        dtype=dtype,
        n_particles=args.particles,
        n_iters=args.iters,
        alpha_bounds=bounds,
        smooth_window=args.smooth_window,
        base_seed=args.seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit alpha parameters on rolling windows of price data."
    )
    parser.add_argument("--csv-dir", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("out_core"))
    parser.add_argument("--output-name", type=str, default="all_fit_windows.csv")
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--ticker-file", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--particles", type=int, default=96)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--bounds", type=float, default=1000.0)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12345)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = build_config(args)
    device = select_device(args.device)

    print(f"Device: {device}")
    print(f"Dtype: {str(config.dtype).replace('torch.', '')}")
    print(f"Tickers: {len(config.tickers)}")
    print(f"Output: {config.output_csv}")

    all_rows: List[Dict[str, object]] = []
    skipped: List[Tuple[str, str]] = []

    for ticker in config.tickers:
        try:
            ticker_rows = fit_ticker(ticker, device, config)
            all_rows.extend(ticker_rows)
            print(f"[DONE] {ticker}: {len(ticker_rows)} windows")
        except Exception as exc:  # noqa: BLE001
            skipped.append((ticker, str(exc)))
            print(f"[SKIP] {ticker}: {exc}")

    save_results(all_rows, skipped, config)
    print(f"Saved: {config.output_csv}")
    if skipped:
        print(f"Skipped ticker report: {config.skipped_csv}")
    print(f"Rows: {len(all_rows)} | Skipped tickers: {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
