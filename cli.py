"""Command-line interface.

Examples
--------
Train on Apple with defaults from config.yaml::

    python -m src.cli train --ticker AAPL --epochs 50

Forecast the next few days from a saved checkpoint::

    python -m src.cli predict --plot

"""

from __future__ import annotations

import argparse
import os

from .config import apply_overrides, load_config
from .train import resolve_device, train


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="config.yaml", help="path to YAML config")
    p.add_argument("--ticker", help="ticker symbol, e.g. AAPL")
    p.add_argument("--checkpoint", help="checkpoint path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stock-forecaster",
        description="Train an LSTM to forecast stock/option prices.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    pt = sub.add_parser("train", help="download data and train a model")
    _add_common(pt)
    pt.add_argument("--csv", help="load data from a local CSV instead of downloading")
    pt.add_argument(
        "--target_mode",
        choices=["logreturn", "price"],
        help="predict log returns (recommended) or raw price level",
    )
    pt.add_argument("--start", help="history start date YYYY-MM-DD")
    pt.add_argument("--end", help="history end date YYYY-MM-DD")
    pt.add_argument("--interval", help="bar interval, e.g. 1d, 1h")
    pt.add_argument("--lookback", type=int, help="days of history fed to the model")
    pt.add_argument("--horizon", type=int, help="days to forecast ahead")
    pt.add_argument("--epochs", type=int)
    pt.add_argument("--batch_size", type=int)
    pt.add_argument("--lr", type=float)
    pt.add_argument("--device", choices=["auto", "cpu", "cuda"])
    pt.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="continue training from the existing checkpoint instead of starting fresh",
    )

    # predict
    pp = sub.add_parser("predict", help="forecast future prices from a checkpoint")
    _add_common(pp)
    pp.add_argument("--plot", action="store_true", help="also save a forecast plot")
    pp.add_argument("--device", choices=["auto", "cpu", "cuda"])

    # backtest
    pb = sub.add_parser("backtest", help="measure held-out accuracy vs a naive baseline")
    _add_common(pb)
    pb.add_argument("--eval_split", type=float, help="fraction of recent data to test on")
    pb.add_argument("--device", choices=["auto", "cpu", "cuda"])

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    overrides = {
        k: v
        for k, v in vars(args).items()
        if k not in {"command", "config", "plot"} and v is not None
    }
    cfg = apply_overrides(cfg, overrides)

    if args.command == "train":
        train(cfg)
        return 0

    if args.command == "predict":
        from .predict import backtest_plot, forecast

        device = resolve_device(cfg.train.device)
        if not os.path.exists(cfg.paths.checkpoint):
            raise SystemExit(
                f"No checkpoint at '{cfg.paths.checkpoint}'. Run `train` first."
            )
        preds = forecast(cfg.paths.checkpoint, device)
        print("\nForecast:")
        print(preds.to_string(float_format=lambda v: f"{v:,.2f}"))
        if args.plot:
            os.makedirs(cfg.paths.output_dir, exist_ok=True)
            out = os.path.join(cfg.paths.output_dir, f"{cfg.data.ticker}_forecast.png")
            backtest_plot(cfg.paths.checkpoint, out, device)
            print(f"\nSaved plot -> {out}")
        return 0

    if args.command == "backtest":
        from .backtest import format_report, run_backtest

        device = resolve_device(cfg.train.device)
        if not os.path.exists(cfg.paths.checkpoint):
            raise SystemExit(
                f"No checkpoint at '{cfg.paths.checkpoint}'. Run `train` first."
            )
        metrics = run_backtest(cfg.paths.checkpoint, device, eval_split=args.eval_split)
        print("\n" + format_report(metrics))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
