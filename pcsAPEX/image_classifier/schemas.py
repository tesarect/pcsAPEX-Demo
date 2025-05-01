"""
Pydantic schemas for data validation and settings management
"""
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

class TrainingConfig(BaseSettings):
    """Configuration for model training"""
    training_epochs: int = Field(10, description="Number of epochs to train the model")
    batch_size: int = Field(32, description="Batch size for training")
    use_gpu: Union[Literal["auto"], bool] = Field("auto", description="Use GPU for training if available (auto, true, false)")
    learning_rate: float = Field(0.001, description="Learning rate for optimizer")
    min_confidence_threshold: float = Field(70.0, description="Minimum confidence threshold (%) for auto-labeling")
    auto_suggest_labels: bool = Field(False, description="Automatically suggest labels for newly uploaded images")
    
    model_config = SettingsConfigDict(
        env_prefix="ML_",  # Environment variables will be prefixed with ML_
    )
    
    @classmethod
    def from_system_config(cls):
        """Load configuration from SystemConfig model"""
        from .models import SystemConfig
        
        # Get values from SystemConfig with appropriate defaults and types
        training_epochs = SystemConfig.get_value('training_epochs', '10', as_type=int)
        batch_size = SystemConfig.get_value('batch_size', '32', as_type=int)
        use_gpu = SystemConfig.get_value('use_gpu', 'auto')
        learning_rate = SystemConfig.get_value('learning_rate', '0.001', as_type=float)
        min_confidence = SystemConfig.get_value('min_confidence_threshold', '70', as_type=float)
        auto_suggest = SystemConfig.get_value('auto_suggest_labels', 'false', as_type=bool)
        
        # Convert use_gpu string to appropriate type
        if use_gpu == 'auto':
            use_gpu_value = "auto"
        else:
            use_gpu_value = use_gpu.lower() in ('true', 'yes', '1', 'on')
        
        # Create and return the config object
        return cls(
            training_epochs=training_epochs,
            batch_size=batch_size,
            use_gpu=use_gpu_value,
            learning_rate=learning_rate,
            min_confidence_threshold=min_confidence,
            auto_suggest_labels=auto_suggest
        )

class PredictionResult(BaseModel):
    """Schema for prediction results"""
    success: bool
    category: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    
    model_config = ConfigDict(
        extra="ignore"  # Ignore extra fields
    )

class TrainingResult(BaseModel):
    """Schema for training results"""
    success: bool
    accuracy: Optional[float] = None
    model_path: Optional[str] = None
    encoder_path: Optional[str] = None
    error: Optional[str] = None
    
    model_config = ConfigDict(
        extra="ignore"  # Ignore extra fields
    )