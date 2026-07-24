import argparse
import os
from src.train import train
from src.evaluate import evaluate
from src.plot_results import plot_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Precision Rocket Soft-Landing RL")
    parser.add_argument("--train", action="store_true", help="Run PPO training")
    parser.add_argument(
        "--evaluate", action="store_true", help="Run evaluation benchmark"
    )
    parser.add_argument("--plot", action="store_true", help="Generate training plots")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/ppo_model_final.pt",
        help="Path to model weights",
    )
    args = parser.parse_args()
    if args.evaluate:
        if not os.path.exists(args.model):
            alt_model = f"../{args.model}"
            if os.path.exists(alt_model):
                args.model = alt_model
    if args.train:
        print("Starting training...")
        train()
    if args.evaluate:
        print(f"Evaluating model: {args.model}")
        evaluate(args.model)
    if args.plot:
        print("Generating plots...")
        plot_metrics()
    if not (args.train or args.evaluate or args.plot):
        parser.print_help()


if __name__ == "__main__":
    main()
