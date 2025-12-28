"""Example script for running a federated learning simulation.

This demonstrates how to set up and run the federated learning platform
locally for testing and development.
"""

import sys
import os
import argparse
import threading
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_server(num_rounds: int = 5, min_clients: int = 2):
    """Run the federated learning server."""
    from server.federated_server import FederatedServer
    
    server = FederatedServer(
        model_architecture="MedicalCNN",
        num_classes=3,  # Simple classification for demo
        num_rounds=num_rounds,
        min_clients=min_clients,
        enable_dp=True,
        target_epsilon=8.0,
    )
    
    return server.start()


def run_client(hospital_id: str, server_address: str = "localhost:8080"):
    """Run a hospital client."""
    from clients.hospital_client import start_client
    
    start_client(
        hospital_id=hospital_id,
        data_path="./data",  # Synthetic data will be generated
        server_address=server_address,
        num_classes=3,
        enable_dp=True,
    )


def main():
    """Main entry point for simulation."""
    parser = argparse.ArgumentParser(
        description="Federated Learning Platform Simulation"
    )
    parser.add_argument(
        "--mode",
        choices=["server", "client", "simulation"],
        default="simulation",
        help="Run mode: server, client, or full simulation",
    )
    parser.add_argument(
        "--hospital-id",
        type=str,
        default="hospital_1",
        help="Hospital ID (for client mode)",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default="localhost:8080",
        help="Server address",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=3,
        help="Number of training rounds",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=2,
        help="Number of clients (for simulation mode)",
    )
    
    args = parser.parse_args()
    
    if args.mode == "server":
        print("Starting Federated Learning Server...")
        results = run_server(
            num_rounds=args.num_rounds,
            min_clients=args.num_clients,
        )
        print(f"Training complete! Final epsilon: {results['final_epsilon']:.4f}")
        
    elif args.mode == "client":
        print(f"Starting Hospital Client: {args.hospital_id}")
        run_client(
            hospital_id=args.hospital_id,
            server_address=args.server_address,
        )
        
    elif args.mode == "simulation":
        print("=" * 60)
        print("Federated Learning Platform - Local Simulation")
        print("=" * 60)
        print(f"Rounds: {args.num_rounds}")
        print(f"Clients: {args.num_clients}")
        print("=" * 60)
        
        # Start server in background thread
        server_thread = threading.Thread(
            target=run_server,
            kwargs={
                "num_rounds": args.num_rounds,
                "min_clients": args.num_clients,
            },
            daemon=True,
        )
        server_thread.start()
        
        # Wait for server to start
        print("Waiting for server to start...")
        time.sleep(3)
        
        # Start clients
        client_threads = []
        for i in range(args.num_clients):
            hospital_id = f"hospital_{i+1}"
            print(f"Starting client: {hospital_id}")
            
            thread = threading.Thread(
                target=run_client,
                kwargs={
                    "hospital_id": hospital_id,
                    "server_address": args.server_address,
                },
                daemon=True,
            )
            thread.start()
            client_threads.append(thread)
            time.sleep(0.5)  # Stagger client starts
        
        # Wait for training to complete
        server_thread.join()
        
        print("\n" + "=" * 60)
        print("Simulation Complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()
