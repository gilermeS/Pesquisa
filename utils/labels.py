"""Bilingual label system for visualizations."""
from typing import Dict, Optional


class BilingualLabel:
    """Manages bilingual labels for plot elements."""
    
    def __init__(self, pt: str, en: str):
        self.pt = pt
        self.en = en
    
    def get(self, language: str = 'both') -> str:
        """Get label in requested language format."""
        if language == 'pt':
            return self.pt
        elif language == 'en':
            return self.en
        elif language == 'both':
            return f"{self.pt}\n({self.en})"
        else:
            raise ValueError(f"Unknown language: {language}")


LABELS: Dict[str, BilingualLabel] = {
    'redshift': BilingualLabel('Deslocamento para o vermelho (z)', 'Redshift (z)'),
    'hubble': BilingualLabel('Parâmetro de Hubble H(z)', 'Hubble Parameter H(z)'),
    'hubble_unit': BilingualLabel('H(z) [km s⁻¹ Mpc⁻¹]', 'H(z) [km s⁻¹ Mpc⁻¹]'),
    'h0': BilingualLabel('Constante de Hubble H₀', 'Hubble Constant H₀'),
    'h0_unit': BilingualLabel('H₀ [km s⁻¹ Mpc⁻¹]', 'H₀ [km s⁻¹ Mpc⁻¹]'),
    'omega_m': BilingualLabel('Parâmetro de densidade de matéria Ωₘ', 'Matter density parameter Ωₘ'),
    'omega_de': BilingualLabel('Parâmetro de densidade de energia escura Ω₌', 'Dark energy density parameter Ω₌'),
    'w0': BilingualLabel('Parâmetro EoS da energia escura w₀', 'Dark energy EoS parameter w₀'),
    'wa': BilingualLabel('Parâmetro de evolução da EoS wₐ', 'EoS evolution parameter wₐ'),
    'loss': BilingualLabel('Perda', 'Loss'),
    'mse': BilingualLabel('Erro quadrático médio (MSE)', 'Mean Squared Error (MSE)'),
    'mae': BilingualLabel('Erro absoluto médio (MAE)', 'Mean Absolute Error (MAE)'),
    'rmse': BilingualLabel('Raiz do erro quadrático médio (RMSE)', 'Root Mean Squared Error (RMSE)'),
    'r2': BilingualLabel('Coeficiente de determinação R²', 'Coefficient of determination R²'),
    'val_loss': BilingualLabel('Perda de validação', 'Validation Loss'),
    'train_loss': BilingualLabel('Perda de treino', 'Training Loss'),
    'epoch': BilingualLabel('Época', 'Epoch'),
    'learning_rate': BilingualLabel('Taxa de aprendizado', 'Learning Rate'),
    'batch_size': BilingualLabel('Tamanho do batch', 'Batch Size'),
    'true_value': BilingualLabel('Valor verdadeiro', 'True Value'),
    'predicted_value': BilingualLabel('Valor previsto', 'Predicted Value'),
    'residual': BilingualLabel('Residual', 'Residual'),
    'residuals': BilingualLabel('Residuais', 'Residuals'),
    'uncertainty': BilingualLabel('Incerteza', 'Uncertainty'),
    'confidence_interval': BilingualLabel('Intervalo de confiança', 'Confidence Interval'),
    'feature_importance': BilingualLabel('Importância das features', 'Feature Importance'),
    'permutation_importance': BilingualLabel('Importância por permutação', 'Permutation Importance'),
    'timestep_importance': BilingualLabel('Importância por passo temporal', 'Timestep Importance'),
    'relative_importance': BilingualLabel('Importância relativa', 'Relative Importance'),
    'decrease_performance': BilingualLabel('Queda de desempenho', 'Performance Decrease'),
    'cnn': BilingualLabel('CNN', 'CNN'),
    'dense': BilingualLabel('Densa', 'Dense'),
    'rnn': BilingualLabel('RNN', 'RNN'),
    'rnn_bi': BilingualLabel('RNN Bidirecional', 'Bidirectional RNN'),
    'svm': BilingualLabel('SVM', 'SVM'),
    'lcdm': BilingualLabel('LCDM', 'LCDM'),
    'wcdm': BilingualLabel('wCDM', 'wCDM'),
    'wacdm': BilingualLabel('w(a)CDM', 'w(a)CDM'),
    'sample': BilingualLabel('Amostra', 'Sample'),
    'frequency': BilingualLabel('Frequência', 'Frequency'),
    'distribution': BilingualLabel('Distribuição', 'Distribution'),
    'comparison': BilingualLabel('Comparação', 'Comparison'),
    'theory': BilingualLabel('Teoria', 'Theory'),
    'observation': BilingualLabel('Observação', 'Observation'),
    'simulation': BilingualLabel('Simulação', 'Simulation'),
}


def get_label(key: str, language: str = 'both') -> str:
    """Get bilingual label for a key."""
    if key not in LABELS:
        raise KeyError(f"Unknown label key: {key}")
    return LABELS[key].get(language)


def format_axis(ax, x_key: Optional[str] = None, y_key: Optional[str] = None,
                title_key: Optional[str] = None, language: str = 'both') -> None:
    """Format axes with bilingual labels."""
    if x_key:
        ax.set_xlabel(get_label(x_key, language), fontsize=ax.xaxis.label.get_size())
    if y_key:
        ax.set_ylabel(get_label(y_key, language), fontsize=ax.yaxis.label.get_size())
    if title_key:
        ax.set_title(get_label(title_key, language), fontsize=ax.title.get_size())
