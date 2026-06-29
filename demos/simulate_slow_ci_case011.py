import time
import sys
from pathlib import Path
import json

def main():
    # We will use Case 011 from the balanced dataset: skops-dev/skops, a massive 61,000 line log!
    # This is perfect for a "slow running CI" that takes 15 minutes to fail.
    
    case_dir = Path("comparison_results/balanced_50_full_logs_20260516") / "011_failure_skops"
    
    # Automatically resolve paths from project root
    project_root = Path(__file__).resolve().parents[1]
    base_dir = project_root / "comparison_results" / "balanced_50_full_logs_20260516"
    if not base_dir.exists():
        print(f"Error: {base_dir} not found.")
        return
        
    target_dir = next((d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("011_failure_skops")), None)
    if not target_dir:
        print("Error: Case 011 skops not found.")
        return
        
    log_file = target_dir / "full_log.txt"
    if not log_file.exists():
        print(f"Error: {log_file} not found.")
        return
        
    print(f"Starting GitHub Actions Runner for skops-dev/skops...")
    print(f"Workflow: build-test.yml (Run #1688)")
    print("Setting up job...\n")
    time.sleep(2)
    
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                # Print the line exactly as it appears
                print(line, end="")
                sys.stdout.flush()
                
                # Add a realistic delay to make it take ~15 minutes (61,000 lines * 0.015s = 915 seconds)
                time.sleep(0.015)
                
    except KeyboardInterrupt:
        print("\n[Runner Interrupted]")

if __name__ == "__main__":
    main()
