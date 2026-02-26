from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    competition_id: int = 43  # FIFA World Cup
    season_id: int = 106      # 2022
    pitch_length: float = 120.0
    pitch_width: float = 80.0
    grid_x: int = 16
    grid_y: int = 12
    goal_center_x: float = 120.0
    goal_center_y: float = 40.0
    goal_left_y: float = 36.0
    goal_right_y: float = 44.0
    xt_iterations: int = 50
    shot_lookahead_actions: int = 5
    test_size: float = 0.2
    random_state: int = 42

    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
