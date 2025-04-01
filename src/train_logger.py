import json
import time
from pathlib import Path

class TrainingLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = time.strftime("%Y-%m-%d_%H%M")  # Format: YYYYMMDD_HHMMSS
        self.log_file = self.log_dir / f"training_log_{self.start_time}.txt"

        self.logs = {
            "parameters": {},
            "start_time": None,
            "end_time": None,
            "training_duration": None,
            "epoch_losses": [],
        }

    def log_parameters(self, **kwargs):
        """Log model hyperparameters and any relevant configuration details."""
        self.logs["parameters"].update(kwargs)

    def start_timer(self):
        """Record training start time."""
        self.logs["start_time"] = time.strftime("%Y-%m-%d %H:%M")
        self.start_time_unix = time.time()

    def end_timer(self):
        """Record training end time and compute total duration."""
        self.logs["end_time"] = time.strftime("%Y-%m-%d %H:%M")
        duration_seconds = round(time.time() - self.start_time_unix, 2)
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.logs["training_duration"] = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

    def log_epoch_loss(self, epoch, loss):
        """Log loss per epoch."""
        self.logs["epoch_losses"].append({"epoch": epoch + 1, "loss": loss})

    def save_log(self):
        """Save logs to a .txt file."""
        with open(self.log_file, "w") as f:
            json.dump(self.logs, f, indent=4)
