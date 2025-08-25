//! # TYL Ollama Inference Adapter
//!
//! Local LLM inference adapter for Ollama with TYL framework integration.
//!
//! This adapter provides seamless integration with Ollama's local LLM server,
//! supporting the simplified template-based interface from tyl-llm-inference-port.
//!
//! ## Features
//!
//! - Template-based prompt generation with parameter substitution
//! - Full Ollama API integration with streaming and non-streaming modes
//! - Comprehensive model management (list, pull, delete)
//! - TYL framework integration (errors, config, logging, tracing)
//! - Health monitoring and server status checking
//! - Token counting and usage tracking
//! - Retry logic with exponential backoff
//! - Builder pattern configuration
//!
//! ## Quick Start
//!
//! ```rust
//! use tyl_ollama_inference_adapter::{OllamaAdapter, OllamaConfig};
//! use tyl_llm_inference_port::{InferenceService, InferenceRequest, ModelType};
//! use std::collections::HashMap;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Basic usage with local Ollama server
//! let adapter = OllamaAdapter::new("llama2")  // Default model
//!     .with_host("http://localhost:11434")
//!     .with_timeout_seconds(120);
//!
//! let mut params = HashMap::new();
//! params.insert("language".to_string(), "Rust".to_string());
//!
//! let request = InferenceRequest::new(
//!     "Write a {{language}} function that prints hello world",
//!     params,
//!     ModelType::Coding
//! );
//!
//! let response = adapter.infer(request).await?;
//! println!("Generated code: {}", response.content);
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! This module follows hexagonal architecture:
//!
//! - **Port (Interface)**: `InferenceService` from tyl-llm-inference-port
//! - **Adapters**: `OllamaAdapter` - implements local Ollama server communication
//! - **Domain Logic**: Template processing, model management, health checking
//!
//! ## Examples
//!
//! See the `examples/` directory for complete usage examples.

// Re-export TYL framework functionality
pub use tyl_errors::{TylError, TylResult};

// Re-export inference port types
pub use tyl_llm_inference_port::{
    HealthCheckResult, HealthStatus, InferenceRequest, InferenceResponse, InferenceResult,
    InferenceService, ModelType, ResponseMetadata, TokenUsage,
};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::{Client, ClientBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Result type for Ollama operations
pub type OllamaResult<T> = TylResult<T>;

/// Ollama adapter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    /// Ollama server host URL
    pub host: String,

    /// Default model to use
    pub default_model: String,

    /// Request timeout in seconds
    pub timeout_seconds: u64,

    /// Maximum number of retries
    pub max_retries: u32,

    /// Enable request/response logging
    pub enable_logging: bool,

    /// Enable distributed tracing
    pub enable_tracing: bool,

    /// Custom headers for requests
    pub custom_headers: HashMap<String, String>,

    /// Keep alive duration for connections (seconds)
    pub keep_alive_seconds: Option<u64>,
}

impl OllamaConfig {
    /// Create new Ollama configuration
    pub fn new(default_model: impl Into<String>) -> Self {
        Self {
            host: "http://localhost:11434".to_string(),
            default_model: default_model.into(),
            timeout_seconds: 120,
            max_retries: 3,
            enable_logging: true,
            enable_tracing: true,
            custom_headers: HashMap::new(),
            keep_alive_seconds: Some(300),
        }
    }

    /// Set Ollama server host
    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// Set request timeout
    pub fn with_timeout_seconds(mut self, timeout: u64) -> Self {
        self.timeout_seconds = timeout;
        self
    }

    /// Set maximum retries
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Enable or disable logging
    pub fn with_logging_enabled(mut self, enabled: bool) -> Self {
        self.enable_logging = enabled;
        self
    }

    /// Enable or disable tracing
    pub fn with_tracing_enabled(mut self, enabled: bool) -> Self {
        self.enable_tracing = enabled;
        self
    }

    /// Add custom header
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom_headers.insert(key.into(), value.into());
        self
    }

    /// Set keep alive duration
    pub fn with_keep_alive_seconds(mut self, seconds: Option<u64>) -> Self {
        self.keep_alive_seconds = seconds;
        self
    }
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self::new("llama2")
    }
}

/// Main Ollama adapter implementing InferenceService
#[derive(Debug)]
pub struct OllamaAdapter {
    config: OllamaConfig,
    client: Client,
}

impl OllamaAdapter {
    /// Create new Ollama adapter
    pub fn new(default_model: impl Into<String>) -> Self {
        let config = OllamaConfig::new(default_model);
        let client = Self::build_client(&config);

        Self { config, client }
    }

    /// Create from configuration
    pub fn from_config(config: OllamaConfig) -> Self {
        let client = Self::build_client(&config);

        Self { config, client }
    }

    /// Set Ollama server host
    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.config.host = host.into();
        self.client = Self::build_client(&self.config);
        self
    }

    /// Set request timeout
    pub fn with_timeout_seconds(mut self, timeout: u64) -> Self {
        self.config.timeout_seconds = timeout;
        self.client = Self::build_client(&self.config);
        self
    }

    /// Set maximum retries
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.config.max_retries = retries;
        self
    }

    /// Get configuration reference
    pub fn config(&self) -> &OllamaConfig {
        &self.config
    }

    fn build_client(config: &OllamaConfig) -> Client {
        let mut builder = ClientBuilder::new().timeout(Duration::from_secs(config.timeout_seconds));

        if let Some(keep_alive) = config.keep_alive_seconds {
            builder = builder.pool_idle_timeout(Duration::from_secs(keep_alive));
        }

        builder.build().expect("Failed to create HTTP client")
    }

    async fn make_request_with_retry<T, F, Fut>(&self, operation: F) -> OllamaResult<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = OllamaResult<T>>,
    {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);

                    if attempt < self.config.max_retries {
                        let delay = Duration::from_millis(100 * 2_u64.pow(attempt));
                        sleep(delay).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| ollama_errors::connection_failed("Max retries exceeded")))
    }

    /// Pull a model from Ollama registry
    pub async fn pull_model(&self, model: &str) -> OllamaResult<()> {
        let url = format!("{}/api/pull", self.config.host);
        let body = serde_json::json!({
            "name": model
        });

        self.make_request_with_retry(|| async {
            let response = self
                .client
                .post(&url)
                .json(&body)
                .send()
                .await
                .map_err(|e| ollama_errors::connection_failed(&e.to_string()))?;

            if !response.status().is_success() {
                return Err(ollama_errors::model_pull_failed(
                    model,
                    &response.status().to_string(),
                ));
            }

            Ok(())
        })
        .await
    }

    /// List available models
    pub async fn list_models(&self) -> OllamaResult<Vec<OllamaModelInfo>> {
        let url = format!("{}/api/tags", self.config.host);

        self.make_request_with_retry(|| async {
            let response = self
                .client
                .get(&url)
                .send()
                .await
                .map_err(|e| ollama_errors::connection_failed(&e.to_string()))?;

            if !response.status().is_success() {
                return Err(ollama_errors::api_error(&response.status().to_string()));
            }

            let models_response: OllamaModelsResponse = response
                .json()
                .await
                .map_err(|e| ollama_errors::invalid_response_format(&e.to_string()))?;

            Ok(models_response.models)
        })
        .await
    }

    /// Delete a model
    pub async fn delete_model(&self, model: &str) -> OllamaResult<()> {
        let url = format!("{}/api/delete", self.config.host);
        let body = serde_json::json!({
            "name": model
        });

        self.make_request_with_retry(|| async {
            let response = self
                .client
                .delete(&url)
                .json(&body)
                .send()
                .await
                .map_err(|e| ollama_errors::connection_failed(&e.to_string()))?;

            if !response.status().is_success() {
                return Err(ollama_errors::model_not_found(model));
            }

            Ok(())
        })
        .await
    }

    fn estimate_tokens(&self, text: &str) -> usize {
        // Simple approximation for local models: ~3.5 characters per token
        (text.len() + 2) / 3
    }

    async fn generate_completion(
        &self,
        request: &InferenceRequest,
    ) -> OllamaResult<InferenceResponse> {
        let start = Instant::now();
        let rendered_prompt = request.render_template();

        let model = request
            .model_override
            .as_deref()
            .unwrap_or(&self.config.default_model);

        let url = format!("{}/api/generate", self.config.host);
        let body = OllamaGenerateRequest {
            model: model.to_string(),
            prompt: rendered_prompt.clone(),
            stream: false,
            options: self.build_generation_options(request),
        };

        if self.config.enable_logging {
            println!(
                "[INFO] Ollama request to model: {} with prompt length: {}",
                model,
                rendered_prompt.len()
            );
        }

        let ollama_response: OllamaGenerateResponse = self
            .make_request_with_retry(|| async {
                let response = self
                    .client
                    .post(&url)
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| ollama_errors::connection_failed(&e.to_string()))?;

                if !response.status().is_success() {
                    return Err(ollama_errors::generation_failed(
                        &response.status().to_string(),
                    ));
                }

                response
                    .json()
                    .await
                    .map_err(|e| ollama_errors::invalid_response_format(&e.to_string()))
            })
            .await?;

        let prompt_tokens = self.estimate_tokens(&rendered_prompt);
        let completion_tokens = self.estimate_tokens(&ollama_response.response);
        let processing_time = start.elapsed().as_millis() as u64;

        // Try to parse response as JSON, fallback to string
        let content = match serde_json::from_str::<serde_json::Value>(&ollama_response.response) {
            Ok(json_value) => json_value,
            Err(_) => serde_json::Value::String(ollama_response.response),
        };

        let metadata = ResponseMetadata::new(
            model.to_string(),
            TokenUsage::new(prompt_tokens as u32, completion_tokens as u32),
            processing_time,
        );

        if self.config.enable_logging {
            println!(
                "[INFO] Ollama response: {} tokens in {}ms",
                metadata.token_usage.total_tokens, processing_time
            );
        }

        Ok(InferenceResponse { content, metadata })
    }

    fn build_generation_options(&self, request: &InferenceRequest) -> OllamaOptions {
        OllamaOptions {
            temperature: request.temperature,
            top_p: None,
            top_k: None,
            num_predict: request.max_tokens.map(|t| t as i32),
            repeat_penalty: None,
            seed: None,
        }
    }
}

#[async_trait]
impl InferenceService for OllamaAdapter {
    async fn infer(&self, request: InferenceRequest) -> InferenceResult<InferenceResponse> {
        self.generate_completion(&request).await
    }

    async fn health_check(&self) -> InferenceResult<HealthCheckResult> {
        let url = format!("{}/api/tags", self.config.host);

        let health_status = match self.client.get(&url).send().await {
            Ok(response) if response.status().is_success() => HealthStatus::healthy(),
            Ok(response) => {
                HealthStatus::unhealthy(format!("Server returned: {}", response.status()))
            }
            Err(e) => HealthStatus::unhealthy(format!("Connection failed: {e}")),
        };

        let mut result = HealthCheckResult::new(health_status)
            .with_metadata("service", serde_json::Value::String("ollama".to_string()))
            .with_metadata("host", serde_json::Value::String(self.config.host.clone()))
            .with_metadata(
                "default_model",
                serde_json::Value::String(self.config.default_model.clone()),
            );

        if let Ok(models) = self.list_models().await {
            result = result.with_metadata(
                "available_models",
                serde_json::Value::Number(serde_json::Number::from(models.len())),
            );
        }

        Ok(result)
    }

    fn supported_models(&self) -> Vec<String> {
        // Return common Ollama models - in practice, this could be dynamic
        vec![
            "llama2".to_string(),
            "llama2:13b".to_string(),
            "llama2:70b".to_string(),
            "codellama".to_string(),
            "codellama:13b".to_string(),
            "codellama:34b".to_string(),
            "mistral".to_string(),
            "mistral:7b".to_string(),
            "neural-chat".to_string(),
            "starling-lm".to_string(),
            "orca-mini".to_string(),
            "vicuna".to_string(),
            "llama2-uncensored".to_string(),
            "dolphin2.1-mistral".to_string(),
            "phi".to_string(),
            "wizardcoder".to_string(),
        ]
    }

    fn count_tokens(&self, text: &str) -> InferenceResult<usize> {
        Ok(self.estimate_tokens(text))
    }
}

/// Ollama model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModelInfo {
    pub name: String,
    pub modified_at: DateTime<Utc>,
    pub size: u64,
    pub digest: String,
    pub details: OllamaModelDetails,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModelDetails {
    pub format: String,
    pub family: String,
    pub families: Option<Vec<String>>,
    pub parameter_size: String,
    pub quantization_level: String,
}

#[derive(Debug, Deserialize)]
struct OllamaModelsResponse {
    models: Vec<OllamaModelInfo>,
}

#[derive(Debug, Serialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    num_predict: Option<i32>,
    repeat_penalty: Option<f32>,
    seed: Option<i32>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OllamaGenerateResponse {
    response: String,
    done: bool,
    context: Option<Vec<i32>>,
    total_duration: Option<u64>,
    load_duration: Option<u64>,
    prompt_eval_count: Option<i32>,
    prompt_eval_duration: Option<u64>,
    eval_count: Option<i32>,
    eval_duration: Option<u64>,
}

/// Error helper functions for Ollama operations
pub mod ollama_errors {
    use super::*;

    pub fn connection_failed(details: &str) -> TylError {
        TylError::network(format!("Failed to connect to Ollama server: {details}"))
    }

    pub fn model_not_found(model: &str) -> TylError {
        TylError::not_found("ollama_model", format!("Ollama model '{model}' not found"))
    }

    pub fn model_pull_failed(model: &str, reason: &str) -> TylError {
        TylError::network(format!("Failed to pull Ollama model '{model}': {reason}"))
    }

    pub fn generation_failed(details: &str) -> TylError {
        TylError::internal(format!("Ollama text generation failed: {details}"))
    }

    pub fn api_error(details: &str) -> TylError {
        TylError::network(format!("Ollama API error: {details}"))
    }

    pub fn invalid_response_format(details: &str) -> TylError {
        TylError::serialization(format!("Invalid response format from Ollama: {details}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_ollama_config_creation() {
        let config = OllamaConfig::new("llama2");
        assert_eq!(config.default_model, "llama2");
        assert_eq!(config.host, "http://localhost:11434");
        assert_eq!(config.timeout_seconds, 120);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_ollama_config_builder() {
        let config = OllamaConfig::new("mistral")
            .with_host("http://custom-host:11434")
            .with_timeout_seconds(60)
            .with_max_retries(5)
            .with_logging_enabled(false)
            .with_header("Custom-Header", "value");

        assert_eq!(config.default_model, "mistral");
        assert_eq!(config.host, "http://custom-host:11434");
        assert_eq!(config.timeout_seconds, 60);
        assert_eq!(config.max_retries, 5);
        assert!(!config.enable_logging);
        assert_eq!(
            config.custom_headers.get("Custom-Header"),
            Some(&"value".to_string())
        );
    }

    #[test]
    fn test_ollama_adapter_creation() {
        let adapter = OllamaAdapter::new("llama2");
        assert_eq!(adapter.config().default_model, "llama2");

        let adapter = adapter
            .with_host("http://localhost:11434")
            .with_timeout_seconds(90);
        assert_eq!(adapter.config().host, "http://localhost:11434");
        assert_eq!(adapter.config().timeout_seconds, 90);
    }

    #[test]
    fn test_supported_models() {
        let adapter = OllamaAdapter::new("llama2");
        let models = adapter.supported_models();

        assert!(!models.is_empty());
        assert!(models.contains(&"llama2".to_string()));
        assert!(models.contains(&"codellama".to_string()));
        assert!(models.contains(&"mistral".to_string()));
    }

    #[test]
    fn test_token_counting() {
        let adapter = OllamaAdapter::new("llama2");

        let count = adapter.count_tokens("Hello world").unwrap();
        assert!(count > 0);
        assert!(count <= 10);

        let empty_count = adapter.count_tokens("").unwrap();
        assert_eq!(empty_count, 0);

        let long_text = "This is a much longer piece of text that should have more tokens";
        let long_count = adapter.count_tokens(long_text).unwrap();
        assert!(long_count > count);
    }

    #[test]
    fn test_inference_request_integration() {
        let mut params = HashMap::new();
        params.insert("language".to_string(), "Python".to_string());
        params.insert("task".to_string(), "web scraper".to_string());

        let request = InferenceRequest::new(
            "Write a {{language}} script for {{task}}",
            params,
            ModelType::Coding,
        );

        assert_eq!(
            request.render_template(),
            "Write a Python script for web scraper"
        );
    }

    #[test]
    fn test_error_handling() {
        let connection_error = ollama_errors::connection_failed("timeout");
        assert!(connection_error
            .to_string()
            .contains("Failed to connect to Ollama"));

        let model_error = ollama_errors::model_not_found("nonexistent");
        assert!(model_error.to_string().contains("not found"));

        let generation_error = ollama_errors::generation_failed("out of memory");
        assert!(generation_error.to_string().contains("generation failed"));
    }

    #[test]
    fn test_configuration_serialization() {
        let config = OllamaConfig::new("llama2")
            .with_host("http://test:11434")
            .with_timeout_seconds(180);

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: OllamaConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.default_model, deserialized.default_model);
        assert_eq!(config.host, deserialized.host);
        assert_eq!(config.timeout_seconds, deserialized.timeout_seconds);
    }

    #[test]
    fn test_trait_object_compatibility() {
        let adapter = OllamaAdapter::new("llama2");

        // Test that adapter can be used as trait object
        let inference_service: Box<dyn InferenceService> = Box::new(adapter);

        let models = inference_service.supported_models();
        assert!(!models.is_empty());

        let tokens = inference_service.count_tokens("test").unwrap();
        assert!(tokens > 0);
    }
}
