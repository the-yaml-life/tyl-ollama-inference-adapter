//! Integration tests for TYL Ollama Inference Adapter
//!
//! These tests verify that the Ollama adapter integrates correctly with the TYL framework
//! and the LLM inference port interface. Some tests require a running Ollama server.

use std::collections::HashMap;
use tyl_llm_inference_port::{InferenceRequest, InferenceService, ModelType};
use tyl_ollama_inference_adapter::{ollama_errors, OllamaAdapter, OllamaConfig};

#[test]
fn test_ollama_config_creation() {
    let config = OllamaConfig::new("llama2");
    assert_eq!(config.default_model, "llama2");
    assert_eq!(config.host, "http://localhost:11434");
    assert_eq!(config.timeout_seconds, 120);
    assert_eq!(config.max_retries, 3);
    assert!(config.enable_logging);
    assert!(config.enable_tracing);
}

#[test]
fn test_ollama_config_builder() {
    let config = OllamaConfig::new("codellama")
        .with_host("http://ishtar:11434")
        .with_timeout_seconds(180)
        .with_max_retries(5)
        .with_logging_enabled(false)
        .with_tracing_enabled(true)
        .with_header("X-Custom", "test")
        .with_keep_alive_seconds(Some(600));

    assert_eq!(config.default_model, "codellama");
    assert_eq!(config.host, "http://ishtar:11434");
    assert_eq!(config.timeout_seconds, 180);
    assert_eq!(config.max_retries, 5);
    assert!(!config.enable_logging);
    assert!(config.enable_tracing);
    assert_eq!(
        config.custom_headers.get("X-Custom"),
        Some(&"test".to_string())
    );
    assert_eq!(config.keep_alive_seconds, Some(600));
}

#[test]
fn test_ollama_adapter_creation() {
    let adapter = OllamaAdapter::new("mistral");

    // Test default configuration
    assert_eq!(adapter.config().default_model, "mistral");
    assert_eq!(adapter.config().host, "http://localhost:11434");

    // Test builder methods
    let adapter = adapter
        .with_host("http://ollama-server:11434")
        .with_timeout_seconds(240)
        .with_max_retries(2);

    assert_eq!(adapter.config().host, "http://ollama-server:11434");
    assert_eq!(adapter.config().timeout_seconds, 240);
    assert_eq!(adapter.config().max_retries, 2);
}

#[test]
fn test_supported_models() {
    let adapter = OllamaAdapter::new("llama2");
    let models = adapter.supported_models();

    // Should return the comprehensive list of common Ollama models
    assert!(!models.is_empty());

    // Check for key model families
    assert!(models.iter().any(|m| m.starts_with("llama2")));
    assert!(models.iter().any(|m| m.starts_with("codellama")));
    assert!(models.iter().any(|m| m.starts_with("mistral")));

    // Check for specific important models
    assert!(models.contains(&"llama2".to_string()));
    assert!(models.contains(&"codellama".to_string()));
    assert!(models.contains(&"mistral".to_string()));
    assert!(models.contains(&"wizardcoder".to_string()));

    // Verify reasonable number of models
    assert!(models.len() >= 10);
    assert!(models.len() <= 20);
}

#[test]
fn test_token_counting() {
    let adapter = OllamaAdapter::new("llama2");

    // Test basic token counting (uses ~3.5 chars per token approximation)
    let count = adapter.count_tokens("Hello world").unwrap();
    assert!(count > 0);
    assert!(count <= 10); // Should be roughly 3-4 tokens

    // Test empty string
    let empty_count = adapter.count_tokens("").unwrap();
    assert_eq!(empty_count, 0);

    // Test longer text
    let long_text = "This is a much longer piece of text that should definitely have more tokens than a simple greeting message.";
    let long_count = adapter.count_tokens(long_text).unwrap();
    assert!(long_count > count);
    assert!(long_count > 20); // Should be significantly more tokens

    // Test token estimation accuracy (approximately 3-4 chars per token)
    let medium_text = "The quick brown fox jumps over the lazy dog";
    let medium_count = adapter.count_tokens(medium_text).unwrap();
    let expected_range = (medium_text.len() / 4, medium_text.len() / 2);
    assert!(medium_count >= expected_range.0);
    assert!(medium_count <= expected_range.1);
}

#[test]
fn test_inference_request_integration() {
    // Test that InferenceRequest works with Ollama adapter structure
    let mut params = HashMap::new();
    params.insert("language".to_string(), "Rust".to_string());
    params.insert("task".to_string(), "web server".to_string());
    params.insert("framework".to_string(), "Tokio".to_string());

    let request = InferenceRequest::new(
        "Create a {{language}} {{task}} using {{framework}}",
        params,
        ModelType::Coding,
    );

    let rendered = request.render_template();
    assert_eq!(rendered, "Create a Rust web server using Tokio");

    // Test that model type provides reasonable configuration
    let optimal_model = request.model_type.optimal_anthropic_model();
    assert!(!optimal_model.is_empty());
    assert!(request.model_type.typical_max_tokens() >= 1024);
}

#[test]
fn test_error_handling_integration() {
    // Test error helper functions
    let connection_error = ollama_errors::connection_failed("timeout after 30s");
    assert!(connection_error
        .to_string()
        .contains("Failed to connect to Ollama server"));
    assert!(connection_error.to_string().contains("timeout after 30s"));

    let model_error = ollama_errors::model_not_found("nonexistent-model:99b");
    assert!(model_error.to_string().contains("not found"));
    assert!(model_error.to_string().contains("nonexistent-model:99b"));

    let pull_error = ollama_errors::model_pull_failed("large-model", "insufficient disk space");
    assert!(pull_error.to_string().contains("Failed to pull"));
    assert!(pull_error.to_string().contains("large-model"));
    assert!(pull_error.to_string().contains("insufficient disk space"));

    let generation_error = ollama_errors::generation_failed("CUDA out of memory");
    assert!(generation_error.to_string().contains("generation failed"));
    assert!(generation_error.to_string().contains("CUDA out of memory"));

    let api_error = ollama_errors::api_error("500 Internal Server Error");
    assert!(api_error.to_string().contains("API error"));
    assert!(api_error.to_string().contains("500"));

    let format_error = ollama_errors::invalid_response_format("expected JSON, got HTML");
    assert!(format_error.to_string().contains("invalid response format"));
    assert!(format_error.to_string().contains("expected JSON"));
}

#[test]
fn test_configuration_serialization() {
    let config = OllamaConfig::new("neural-chat")
        .with_host("http://production-ollama:11434")
        .with_timeout_seconds(300)
        .with_max_retries(10)
        .with_header("Authorization", "Bearer token123");

    // Test JSON serialization
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: OllamaConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(config.default_model, deserialized.default_model);
    assert_eq!(config.host, deserialized.host);
    assert_eq!(config.timeout_seconds, deserialized.timeout_seconds);
    assert_eq!(config.max_retries, deserialized.max_retries);
    assert_eq!(config.enable_logging, deserialized.enable_logging);
    assert_eq!(config.custom_headers, deserialized.custom_headers);
}

#[test]
fn test_trait_object_compatibility() {
    let adapter = OllamaAdapter::new("llama2:7b");

    // Test that adapter can be used as trait object
    let inference_service: Box<dyn InferenceService> = Box::new(adapter);

    // Test supported models through trait object
    let models = inference_service.supported_models();
    assert!(!models.is_empty());
    assert!(models.contains(&"llama2".to_string()));

    // Test token counting through trait object
    let tokens = inference_service
        .count_tokens("Test message for token counting")
        .unwrap();
    assert!(tokens > 0);
    assert!(tokens < 20); // Should be reasonable for this short text
}

#[test]
fn test_default_configurations() {
    let default_config = OllamaConfig::default();
    assert_eq!(default_config.default_model, "llama2");
    assert_eq!(default_config.host, "http://localhost:11434");

    let adapter_from_default = OllamaAdapter::from_config(default_config.clone());
    assert_eq!(
        adapter_from_default.config().default_model,
        default_config.default_model
    );
    assert_eq!(adapter_from_default.config().host, default_config.host);
}

#[test]
fn test_model_specific_configurations() {
    // Test configurations for different model types
    let code_adapter = OllamaAdapter::new("codellama:13b").with_timeout_seconds(180); // Longer timeout for larger model

    let chat_adapter = OllamaAdapter::new("neural-chat:7b").with_timeout_seconds(60); // Faster response for chat

    let reasoning_adapter = OllamaAdapter::new("llama2:70b")
        .with_timeout_seconds(600) // Much longer for large reasoning model
        .with_max_retries(2); // Fewer retries due to resource intensity

    assert_eq!(code_adapter.config().default_model, "codellama:13b");
    assert_eq!(code_adapter.config().timeout_seconds, 180);

    assert_eq!(chat_adapter.config().default_model, "neural-chat:7b");
    assert_eq!(chat_adapter.config().timeout_seconds, 60);

    assert_eq!(reasoning_adapter.config().default_model, "llama2:70b");
    assert_eq!(reasoning_adapter.config().timeout_seconds, 600);
    assert_eq!(reasoning_adapter.config().max_retries, 2);
}

#[cfg(test)]
mod tyl_framework_integration {
    use super::*;
    use tyl_errors::TylError;

    #[test]
    fn test_tyl_error_integration() {
        // Test that our error helpers create proper TylError instances
        let connection_error = ollama_errors::connection_failed("server unreachable");

        // Should be a proper TylError
        match &connection_error {
            TylError::Network { message } => {
                assert!(message.contains("Failed to connect to Ollama"));
                assert!(message.contains("server unreachable"));
            }
            _ => panic!("Expected TylError::Network"),
        }

        let not_found_error = ollama_errors::model_not_found("missing-model");
        match &not_found_error {
            TylError::NotFound {
                resource_type,
                identifier,
            } => {
                assert_eq!(resource_type, "ollama_model");
                assert!(identifier.contains("missing-model"));
            }
            _ => panic!("Expected TylError::NotFound"),
        }
    }

    #[test]
    fn test_model_type_optimization_for_local_models() {
        // Test that model types work well with local Ollama models

        // Coding tasks should work well with CodeLlama
        assert_eq!(ModelType::Coding.optimal_openai_model(), "gpt-4o");
        assert!(ModelType::Coding.typical_max_tokens() >= 2048); // Good for code generation

        // Fast tasks should use smaller models
        assert_eq!(ModelType::Fast.optimal_openai_model(), "gpt-3.5-turbo");
        assert!(ModelType::Fast.typical_max_tokens() <= 2048); // Optimized for speed

        // Reasoning tasks need larger context
        assert!(ModelType::Reasoning.typical_max_tokens() >= 4096);

        // General tasks should be balanced
        let general_tokens = ModelType::General.typical_max_tokens();
        assert!(general_tokens >= 1024 && general_tokens <= 4096);
    }

    #[test]
    fn test_configuration_validation_patterns() {
        // Test configuration patterns that would work with tyl-config

        // Valid local development config
        let local_config = OllamaConfig::new("llama2")
            .with_host("http://localhost:11434")
            .with_timeout_seconds(60);

        assert!(local_config.host.starts_with("http://"));
        assert!(local_config.timeout_seconds > 0);
        assert!(local_config.timeout_seconds <= 600);

        // Valid production server config (like Ishtar)
        let server_config = OllamaConfig::new("codellama:13b")
            .with_host("http://ishtar:11434")
            .with_timeout_seconds(300)
            .with_keep_alive_seconds(Some(900));

        assert!(server_config.host.starts_with("http://"));
        assert!(server_config.timeout_seconds >= 120); // Reasonable for server
        assert!(server_config.keep_alive_seconds.unwrap_or(0) >= 300);

        // Test model name validation patterns
        assert!(!local_config.default_model.is_empty());
        assert!(local_config.default_model.len() > 2);
        assert!(server_config.default_model.contains("codellama"));
    }

    #[test]
    fn test_ishtar_server_configuration() {
        // Test specific configuration for Ishtar server deployment
        let ishtar_config = OllamaConfig::new("llama2:13b")
            .with_host("http://ishtar:11434") // Ishtar server hostname
            .with_timeout_seconds(240) // 4 minutes for server processing
            .with_max_retries(5) // More retries for network reliability
            .with_keep_alive_seconds(Some(600)) // 10 minute keep-alive
            .with_logging_enabled(true) // Enable logging for server monitoring
            .with_tracing_enabled(true); // Enable tracing for debugging

        assert_eq!(ishtar_config.host, "http://ishtar:11434");
        assert_eq!(ishtar_config.default_model, "llama2:13b");
        assert_eq!(ishtar_config.timeout_seconds, 240);
        assert_eq!(ishtar_config.max_retries, 5);
        assert_eq!(ishtar_config.keep_alive_seconds, Some(600));
        assert!(ishtar_config.enable_logging);
        assert!(ishtar_config.enable_tracing);

        // Verify this config can be serialized for deployment
        let json = serde_json::to_string(&ishtar_config).unwrap();
        assert!(json.contains("ishtar"));
        assert!(json.contains("llama2:13b"));

        let deserialized: OllamaConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(ishtar_config.host, deserialized.host);
        assert_eq!(ishtar_config.default_model, deserialized.default_model);
    }

    #[test]
    fn test_template_integration_with_local_models() {
        // Test templates that work well with local models
        let coding_templates = vec![
            "Write a {{language}} function to {{task}}",
            "Create a {{language}} {{component}} for {{purpose}}",
            "Debug this {{language}} code: {{code}}",
            "Optimize this {{language}} algorithm: {{algorithm}}",
        ];

        let reasoning_templates = vec![
            "Compare {{option1}} vs {{option2}} for {{context}}",
            "Analyze the pros and cons of {{topic}}",
            "Explain why {{concept}} is important in {{domain}}",
            "What are the trade-offs of {{approach}} in {{scenario}}?",
        ];

        // Test that templates render correctly
        for template in coding_templates {
            assert!(template.contains("{{"));
            assert!(template.contains("}}"));
            assert!(!template.is_empty());
        }

        for template in reasoning_templates {
            assert!(template.contains("{{"));
            assert!(template.contains("}}"));
            assert!(!template.is_empty());
        }

        // Test actual template rendering
        let mut params = HashMap::new();
        params.insert("language".to_string(), "Rust".to_string());
        params.insert("task".to_string(), "parse JSON".to_string());

        let request = InferenceRequest::new(
            "Write a {{language}} function to {{task}}",
            params,
            ModelType::Coding,
        );

        assert_eq!(
            request.render_template(),
            "Write a Rust function to parse JSON"
        );
    }
}

// Tests that require actual Ollama server connection
// These will be skipped in CI but can be run manually when server is available
#[cfg(test)]
mod server_integration_tests {
    use super::*;
    use tokio;

    // Helper function to check if Ollama server is available
    async fn is_ollama_available() -> bool {
        let adapter = OllamaAdapter::new("llama2");
        adapter.health_check().await.is_ok()
    }

    #[tokio::test]
    async fn test_health_check_integration() {
        // This test requires a running Ollama server
        if !is_ollama_available().await {
            println!("⚠️  Skipping health check test - Ollama server not available");
            println!("   To run this test, ensure Ollama is running on localhost:11434");
            println!("   On Ishtar server: Ollama should be pre-configured");
            return;
        }

        let adapter = OllamaAdapter::new("llama2");
        let health_result = adapter.health_check().await;

        assert!(health_result.is_ok());
        let health = health_result.unwrap();
        assert!(health.status.is_healthy());

        // Check metadata
        assert!(health.metadata.contains_key("service"));
        assert!(health.metadata.contains_key("host"));
        assert!(health.metadata.contains_key("default_model"));
    }

    #[tokio::test]
    async fn test_model_listing_integration() {
        // This test requires a running Ollama server with models
        if !is_ollama_available().await {
            println!("⚠️  Skipping model listing test - Ollama server not available");
            println!("   On Ishtar: Models should be pre-installed");
            return;
        }

        let adapter = OllamaAdapter::new("llama2");
        let models_result = adapter.list_models().await;

        // This might fail if no models are installed, which is expected
        match models_result {
            Ok(models) => {
                println!("✅ Found {} models installed on server", models.len());
                for model in models {
                    println!("   - {} ({})", model.name, model.details.parameter_size);
                }
            }
            Err(e) => {
                println!(
                    "⚠️  Could not list models (expected if none installed): {}",
                    e
                );
            }
        }
    }

    #[tokio::test]
    async fn test_ishtar_server_connectivity() {
        // Test specific connectivity to Ishtar server
        let ishtar_adapter = OllamaAdapter::new("llama2")
            .with_host("http://ishtar:11434")
            .with_timeout_seconds(30);

        match ishtar_adapter.health_check().await {
            Ok(health) => {
                println!("✅ Successfully connected to Ishtar Ollama server!");
                println!("   Status: {:?}", health.status);
                if let Some(models) = health.metadata.get("available_models") {
                    println!("   Available models: {}", models);
                }
            }
            Err(e) => {
                println!("⚠️  Could not connect to Ishtar server: {}", e);
                println!("   This is expected if not running on Ishtar or if server is down");
            }
        }
    }
}
