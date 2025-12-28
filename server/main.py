"""Server main entry point."""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.federated_server import start_server


def main():
    """Main entry point for federated learning server."""
    parser = argparse.ArgumentParser(
        description="Federated Learning Server for Medical Diagnosis"
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=None,
        help="Number of training rounds",
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=None,
        help="Minimum clients required",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model architecture (MedicalCNN, MedicalResNet)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=14,
        help="Number of output classes",
    )
    parser.add_argument(
        "--no-dp",
        action="store_true",
        help="Disable differential privacy",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Target epsilon for DP",
    )
    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=None,
        help="Noise multiplier for DP",
    )
    
    args = parser.parse_args()
    
    # Build kwargs from args
    kwargs = {}
    if args.num_rounds:
        kwargs["num_rounds"] = args.num_rounds
    if args.min_clients:
        kwargs["min_clients"] = args.min_clients
    if args.model:
        kwargs["model_architecture"] = args.model
    if args.num_classes:
        kwargs["num_classes"] = args.num_classes
    if args.no_dp:
        kwargs["enable_dp"] = False
    if args.epsilon:
        kwargs["target_epsilon"] = args.epsilon
    if args.noise_multiplier:
        kwargs["noise_multiplier"] = args.noise_multiplier
    
    # Start the server
    results = start_server(**kwargs)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Final ε (epsilon): {results['final_epsilon']:.4f}")
    print(f"Final δ (delta): {results['final_delta']:.2e}")
    print(f"Model saved to: {results['model_path']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
