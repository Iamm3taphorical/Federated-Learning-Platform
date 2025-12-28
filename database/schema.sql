-- Federated Learning Platform - PostgreSQL Schema
-- This database stores ONLY system metadata and compliance tracking
-- Never stores raw medical/patient data

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Hospitals (Federated Clients)
CREATE TABLE hospitals (
    hospital_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    region TEXT,
    public_key TEXT NOT NULL,
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_hospitals_region ON hospitals(region);
CREATE INDEX idx_hospitals_active ON hospitals(is_active);

-- ML Models
CREATE TABLE models (
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    architecture TEXT NOT NULL,
    modality TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    hyperparameters JSONB DEFAULT '{}'
);

CREATE INDEX idx_models_modality ON models(modality);
CREATE INDEX idx_models_architecture ON models(architecture);

-- Training Rounds
CREATE TABLE training_rounds (
    round_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES models(model_id) ON DELETE CASCADE,
    round_number INTEGER NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT DEFAULT 'pending',
    config JSONB DEFAULT '{}'
);

CREATE INDEX idx_training_rounds_model ON training_rounds(model_id);
CREATE INDEX idx_training_rounds_status ON training_rounds(status);
CREATE INDEX idx_training_rounds_number ON training_rounds(model_id, round_number);

-- Round Participation (tracks which hospitals participated in each round)
CREATE TABLE round_participation (
    round_id UUID REFERENCES training_rounds(round_id) ON DELETE CASCADE,
    hospital_id UUID REFERENCES hospitals(hospital_id) ON DELETE CASCADE,
    update_received BOOLEAN DEFAULT FALSE,
    update_size_bytes BIGINT,
    training_time_seconds FLOAT,
    local_samples_used INTEGER,
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    PRIMARY KEY (round_id, hospital_id)
);

CREATE INDEX idx_round_participation_hospital ON round_participation(hospital_id);

-- Model Versions (stores evaluation metrics for each trained version)
CREATE TABLE model_versions (
    version_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES models(model_id) ON DELETE CASCADE,
    round_id UUID REFERENCES training_rounds(round_id) ON DELETE CASCADE,
    accuracy FLOAT,
    auc FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    fairness_score FLOAT,
    loss FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    weights_path TEXT,
    additional_metrics JSONB DEFAULT '{}'
);

CREATE INDEX idx_model_versions_model ON model_versions(model_id);
CREATE INDEX idx_model_versions_round ON model_versions(round_id);
CREATE INDEX idx_model_versions_accuracy ON model_versions(accuracy DESC);

-- Privacy Budget (tracks DP parameters per round)
CREATE TABLE privacy_budget (
    round_id UUID REFERENCES training_rounds(round_id) ON DELETE CASCADE,
    epsilon FLOAT NOT NULL,
    delta FLOAT NOT NULL,
    noise_multiplier FLOAT NOT NULL,
    max_grad_norm FLOAT NOT NULL DEFAULT 1.0,
    cumulative_epsilon FLOAT,
    cumulative_delta FLOAT,
    PRIMARY KEY (round_id)
);

-- Audit Logs (append-only for compliance)
CREATE TABLE audit_logs (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    resource_type TEXT,
    resource_id UUID,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address TEXT,
    details JSONB DEFAULT '{}',
    success BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_audit_logs_actor ON audit_logs(actor);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp DESC);
CREATE INDEX idx_audit_logs_resource ON audit_logs(resource_type, resource_id);

-- Authentication tokens (for OAuth)
CREATE TABLE auth_tokens (
    token_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hospital_id UUID REFERENCES hospitals(hospital_id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    revoked BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_auth_tokens_hospital ON auth_tokens(hospital_id);
CREATE INDEX idx_auth_tokens_expires ON auth_tokens(expires_at);

-- Create a function for audit log triggers
CREATE OR REPLACE FUNCTION audit_trigger_func()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_logs (actor, action, resource_type, resource_id, details)
        VALUES ('system', 'CREATE', TG_TABLE_NAME, NEW.hospital_id, row_to_json(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_logs (actor, action, resource_type, resource_id, details)
        VALUES ('system', 'UPDATE', TG_TABLE_NAME, NEW.hospital_id, 
                jsonb_build_object('old', row_to_json(OLD), 'new', row_to_json(NEW)));
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_logs (actor, action, resource_type, resource_id, details)
        VALUES ('system', 'DELETE', TG_TABLE_NAME, OLD.hospital_id, row_to_json(OLD));
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;
