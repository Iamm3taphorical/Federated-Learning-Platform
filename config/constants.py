"""Constants for the federated learning platform."""

# Model architectures
class ModelArchitecture:
    MEDICAL_CNN = "MedicalCNN"
    RESNET18 = "ResNet18"
    RESNET50 = "ResNet50"
    VGG16 = "VGG16"
    EFFICIENTNET = "EfficientNet"
    TRANSFORMER_EHR = "TransformerEHR"


# Imaging modalities
class Modality:
    XRAY = "xray"
    MRI = "mri"
    CT = "ct"
    ULTRASOUND = "ultrasound"
    MAMMOGRAPHY = "mammography"
    EHR = "ehr"
    MULTIMODAL = "multimodal"


# Training round status
class RoundStatus:
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Privacy solution types
class PrivacySolution:
    DP_SGD = "dp_sgd"  # Differential Privacy with SGD
    LOCAL_DP = "local_dp"  # Local Differential Privacy
    GLOBAL_DP = "global_dp"  # Global/Central DP
    SECURE_AGG = "secure_aggregation"  # Secure Aggregation only


# Audit log actions
class AuditAction:
    # Hospital actions
    HOSPITAL_REGISTER = "hospital.register"
    HOSPITAL_UPDATE = "hospital.update"
    HOSPITAL_DEACTIVATE = "hospital.deactivate"
    
    # Model actions
    MODEL_CREATE = "model.create"
    MODEL_UPDATE = "model.update"
    
    # Training actions
    TRAINING_START = "training.start"
    TRAINING_COMPLETE = "training.complete"
    TRAINING_FAIL = "training.fail"
    ROUND_START = "round.start"
    ROUND_COMPLETE = "round.complete"
    
    # Client actions
    CLIENT_JOIN = "client.join"
    CLIENT_LEAVE = "client.leave"
    UPDATE_RECEIVE = "update.receive"
    
    # Auth actions
    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_FAIL = "auth.fail"
    TOKEN_REVOKE = "token.revoke"


# Default hyperparameters
DEFAULT_HYPERPARAMETERS = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "local_epochs": 1,
    "optimizer": "adam",
    "weight_decay": 1e-4,
}

# DP defaults
DEFAULT_DP_PARAMS = {
    "epsilon": 8.0,
    "delta": 1e-5,
    "max_grad_norm": 1.0,
    "noise_multiplier": 1.1,
}

# Medical imaging defaults
IMAGE_SIZE = {
    "xray": (224, 224),
    "mri": (256, 256),
    "ct": (256, 256),
    "mammography": (224, 224),
}

# Number of classes for common medical tasks
NUM_CLASSES = {
    "chest_xray": 14,  # ChestX-ray14 dataset
    "covid": 3,  # Normal, COVID, Pneumonia
    "diabetic_retinopathy": 5,
    "skin_lesion": 7,
}
