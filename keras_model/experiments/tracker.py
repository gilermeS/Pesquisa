"""Experiment tracking for reproducible research."""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib


class ExperimentTracker:
    """Lightweight experiment tracker for model training."""
    
    def __init__(
        self,
        experiment_dir: Optional[Path] = None,
        experiment_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if experiment_dir is None:
            experiment_dir = Path('experiments')
        
        self.experiment_dir = Path(experiment_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_id = hashlib.md5(
            f"{timestamp}_{time.time()}".encode()
        ).hexdigest()[:8]
        
        if experiment_name is None:
            experiment_name = f"exp_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_path = self.experiment_dir / f"{experiment_name}_{self.experiment_id}"
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        
        self.tags = tags or {}
        self.metadata = metadata or {}
        self.metadata['start_time'] = timestamp
        self.metadata['experiment_id'] = self.experiment_id
        
        self.metrics = {}  # Initialize metrics storage before _save_config() call
        self._save_config()
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        self.params = params
        self._save_config()
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value at a specific step."""
        if name not in self.metrics:
            self.metrics[name] = {'values': [], 'steps': []}
        
        self.metrics[name]['values'].append(float(value))
        self.metrics[name]['steps'].append(step if step is not None else len(self.metrics[name]['values']))
        
        self._save_metrics()
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_model(self, model_path: Path, metric_value: Optional[float] = None) -> None:
        """Log a model checkpoint."""
        import shutil
        
        model_dest = self.experiment_path / 'models'
        model_dest.mkdir(exist_ok=True)
        
        model_name = f"model_{len(list(model_dest.glob('model_*')))}"
        if metric_value is not None:
            model_name += f"_val{metric_value:.4f}"
        model_name += '.keras'
        
        shutil.copy2(model_path, model_dest / model_name)
    
    def get_best_epoch(self, metric: str = 'val_loss') -> int:
        """Get the epoch with best metric value."""
        if metric not in self.metrics:
            raise ValueError(f"Metric {metric} not found")
        
        values = self.metrics[metric]['values']
        return int(min(range(len(values)), key=lambda i: values[i])) + 1
    
    def _save_config(self) -> None:
        """Save configuration to JSON."""
        config = {
            'experiment_name': self.experiment_name,
            'experiment_id': self.experiment_id,
            'experiment_path': str(self.experiment_path),
            'tags': self.tags,
            'metadata': self.metadata,
        }
        
        if hasattr(self, 'params'):
            config['params'] = self.params
        
        with open(self.experiment_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def _save_metrics(self) -> None:
        """Save metrics to JSON."""
        with open(self.experiment_path / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
    
    def __repr__(self) -> str:
        return f"ExperimentTracker(name='{self.experiment_name}', id='{self.experiment_id}')"