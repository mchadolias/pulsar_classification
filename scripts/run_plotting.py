"""
Standalone runner for generating all model comparison plots
from an existing Poller JSON file.

Usage:
    python scripts/run_plotting.py --poller outputs/metrics/poller/poller.json
    python scripts/run_plotting.py --pdf
    python scripts/run_plotting.py --png
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is on Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.plotter import ModelComparisonPlotter
from src.utils import LoggerManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate model comparison plots from an existing Poller file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--poller",
        type=str,
        default="outputs/metrics/poller/poller.json",
        help="Path to Poller JSON file produced during training.",
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "pdf", "svg"],
        default=["png", "pdf"],
        help="Output formats to generate.",
    )

    parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Logging
    log_manager = LoggerManager(log_dir="logs", level=(0 if args.debug else 20))
    logger = log_manager.get_logger()

    logger.info("=== MODEL COMPARISON PLOT RUNNER ===")
    logger.info(f"Reading poller state from: {args.poller}")
    poller_path = Path(args.poller)

    if not poller_path.exists():
        raise FileNotFoundError(f"Poller file not found: {poller_path}")

    # Initialise plotter
    plotter = ModelComparisonPlotter.from_poller_file(str(poller_path))

    logger.info(f"Generating plots in formats: {args.formats}")
    try:
        plotter.plot_all(formats=tuple(args.formats))
        logger.info("Plot generation complete.")
    except Exception as e:
        logger.error(f"Plotting failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
