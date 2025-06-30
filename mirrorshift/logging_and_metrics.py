"""
Rich-based logging and metrics display for training visualization.
"""

from rich.table import Table
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich import box


class RichLogger:
    def __init__(self, total_steps: int, batch_size: int):
        self.console = Console()
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.progress_table = Table(title="[ TRAINING PROGRESS ]", title_style="bold white", box=box.SIMPLE)
        self.progress_table.add_column("METRIC", style="white", no_wrap=True)
        self.progress_table.add_column("VALUE", style="bright_white")
        self.validation_table = Table(title="[ VALIDATION METRICS ]", title_style="bold white", box=box.SIMPLE)
        self.validation_table.add_column("METRIC", style="white", no_wrap=True)
        self.validation_table.add_column("VALUE", style="bright_white")
        self.generation_table = Table(title="[ GENERATED TEXT SAMPLE ]", title_style="bold white", box=box.SIMPLE)
        self.generation_table.add_column("OUTPUT", style="dim white", no_wrap=False)
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main_content")
        )
        title_text = Text("MIRRORSHIFT", style="bold bright_white")
        self.layout["header"].update(
            Panel(Align.center(title_text), border_style="white")
        )
        self.layout["main_content"].split_row(
            Layout(name="left_panel", ratio=1),
            Layout(name="right_panel", ratio=1)
        )
        self.layout["main_content"]["left_panel"].split_column(
            Layout(Panel(self.progress_table, border_style="white"), name="progress"),
            Layout(Panel(self.validation_table, border_style="white"), name="validation")
        )
        self.layout["main_content"]["right_panel"].update(
            Panel(self.generation_table, border_style="white")
        )
        self.live = Live(self.layout, console=self.console, refresh_per_second=4, auto_refresh=True)
        self._init_progress_table()
        self._init_validation_table()
        self._init_generation_table()

    def _init_progress_table(self):
        self.progress_table.add_row("EPOCH", "0")
        self.progress_table.add_row("BATCH SIZE", str(self.batch_size))
        self.progress_table.add_row("STEP", "0")
        self.progress_table.add_row("TOTAL STEPS", str(self.total_steps))
        self.progress_table.add_row("PROGRESS", "0.0%")
        self.progress_table.add_row("LOSS", "INITIALIZING...")
        self.progress_table.add_row("PERPLEXITY", "INITIALIZING...")
        self.progress_table.add_row("LEARNING RATE", "INITIALIZING...")
        self.progress_table.add_row("SEC/STEP", "INITIALIZING...")
        self.progress_table.add_row("STEPS/SEC", "INITIALIZING...")

    def _init_validation_table(self):
        self.validation_table.add_row("LAST VAL STEP", "PENDING...")
        self.validation_table.add_row("VAL LOSS", "PENDING...")
        self.validation_table.add_row("VAL PERPLEXITY", "PENDING...")
        self.validation_table.add_row("BEST VAL LOSS", "PENDING...")

    def _init_generation_table(self):
        self.generation_table.add_row("AWAITING FIRST SAMPLE...")

    def start(self):
        self.live.start()

    def stop(self):
        self.live.stop()

    def update_progress(self, epoch: int, step: int, loss: float, perplexity: float, 
                       learning_rate: float, avg_step_time: float, steps_per_second: float):
        progress_percent = (step / self.total_steps) * 100
        new_progress_table = Table(title="[ TRAINING PROGRESS ]", title_style="bold white", box=box.SIMPLE)
        new_progress_table.add_column("METRIC", style="white", no_wrap=True)
        new_progress_table.add_column("VALUE", style="bright_white")
        new_progress_table.add_row("EPOCH", str(epoch))
        new_progress_table.add_row("BATCH SIZE", str(self.batch_size))
        new_progress_table.add_row("STEP", str(step))
        new_progress_table.add_row("TOTAL STEPS", str(self.total_steps))
        new_progress_table.add_row("PROGRESS", f"{progress_percent:.1f}%")
        new_progress_table.add_row("LOSS", f"{loss:.5f}")
        new_progress_table.add_row("PERPLEXITY", f"{perplexity:.5f}")
        new_progress_table.add_row("LEARNING RATE", f"{learning_rate:.2e}")
        new_progress_table.add_row("SEC/STEP", f"{avg_step_time:.3f}")
        new_progress_table.add_row("STEPS/SEC", f"{steps_per_second:.3f}")
        self.layout["main_content"]["left_panel"]["progress"].update(Panel(new_progress_table, border_style="white"))

    def update_validation(self, step: int, val_loss: float, val_perplexity: float, best_val_loss: float = None):
        new_validation_table = Table(title="[ VALIDATION METRICS ]", title_style="bold white", box=box.SIMPLE)
        new_validation_table.add_column("METRIC", style="white", no_wrap=True)
        new_validation_table.add_column("VALUE", style="bright_white")
        new_validation_table.add_row("LAST VAL STEP", str(step))
        new_validation_table.add_row("VAL LOSS", f"{val_loss:.5f}")
        new_validation_table.add_row("VAL PERPLEXITY", f"{val_perplexity:.5f}")
        if best_val_loss is not None:
            new_validation_table.add_row("BEST VAL LOSS", f"{best_val_loss:.5f}")
        else:
            new_validation_table.add_row("BEST VAL LOSS", f"{val_loss:.5f}")
        self.layout["main_content"]["left_panel"]["validation"].update(Panel(new_validation_table, border_style="white"))

    def update_generation(self, generated_text: str, step: int):
        new_generation_table = Table(title=f"[ GENERATED TEXT SAMPLE - STEP {step} ]", title_style="bold white", box=box.SIMPLE)
        new_generation_table.add_column("OUTPUT", style="dim white", no_wrap=False)
        new_generation_table.add_row(generated_text)
        self.layout["main_content"]["right_panel"].update(Panel(new_generation_table, border_style="white"))

    def print_epoch_start(self, epoch: int):
        self.stop()
        self.console.print(f"\n[bold white]>>> EPOCH {epoch} INITIATED <<<[/bold white]")
        self.start()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
