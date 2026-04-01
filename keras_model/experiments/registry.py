"""Model registry for tracking trained models."""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import shutil


class ModelRegistry:
    """Registry for trained models with versioning and metadata."""
    
    def __init__(self, registry_dir: Optional[Path] = None):
        if registry_dir is None:
            registry_dir = Path('models')
        
        self.registry_dir = Path(registry_dir)
        self.registry_file = self.registry_dir / 'registry.json'
        
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {'models': []}
    
    def register_model(
        self,
        model_path: Path,
        model_name: str,
        architecture: str,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None,
        parent_model: Optional[str] = None,
    ) -> str:
        """Register a trained model."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_id = f"{model_name}_{timestamp}"
        
        model_dir = self.registry_dir / version_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = model_dir / 'model.keras'
        shutil.copy2(model_path, dest_path)
        
        metadata = {
            'version_id': version_id,
            'model_name': model_name,
            'architecture': architecture,
            'metrics': metrics,
            'config': config,
            'tags': tags or {},
            'parent_model': parent_model,
            'registered_at': timestamp,
            'model_path': str(dest_path),
        }
        
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.registry['models'].append(metadata)
        self._save_registry()
        
        return version_id
    
    def get_model(self, version_id: str) -> Path:
        """Get path to a registered model."""
        for model in self.registry['models']:
            if model['version_id'] == version_id:
                return Path(model['model_path'])
        
        raise KeyError(f"Model {version_id} not found in registry")
    
    def list_models(
        self,
        architecture: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        sort_by: Optional[str] = None,
        ascending: bool = True,
    ) -> List[Dict[str, Any]]:
        """List models with optional filtering."""
        models = self.registry['models']
        
        if architecture is not None:
            models = [m for m in models if m['architecture'] == architecture]
        
        if tags is not None:
            def matches_tags(m):
                for key, value in tags.items():
                    if m.get('tags', {}).get(key) != value:
                        return False
                return True
            models = [m for m in models if matches_tags(m)]
        
        if sort_by is not None:
            def get_metric(m):
                return m.get('metrics', {}).get(sort_by, float('inf'))
            models = sorted(models, key=get_metric, reverse=not ascending)
        
        return models
    
    def _save_registry(self) -> None:
        """Save registry to JSON."""
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        registry_copy = {
            'models': [
                {k: v for k, v in m.items() if k != 'model_path'}
                for m in self.registry['models']
        }
        
        with open(self.registry_file, 'w') as f:
            json.dump(registry_copy, f, indent=2)