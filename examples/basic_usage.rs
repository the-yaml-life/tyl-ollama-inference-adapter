//! Basic usage example for TYL Ollama Inference Adapter
//!
//! This example demonstrates how to use the Ollama adapter for local LLM inference
//! with the TYL framework integration and template-based interface.
//!
//! Note: This requires a running Ollama server. On Ishtar server, Ollama is available.

use std::collections::HashMap;
use tyl_llm_inference_port::{InferenceRequest, InferenceService, ModelType};
use tyl_ollama_inference_adapter::{OllamaAdapter, OllamaConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 TYL Ollama Inference Adapter - Local LLM Usage Example");
    println!("========================================================\n");

    // Initialize environment logger for TYL logging
    env_logger::init();

    // Example 1: Basic adapter creation and configuration
    println!("🔧 Example 1: Adapter Configuration");
    println!("----------------------------------");

    let adapter = OllamaAdapter::new("llama2") // Default model
        .with_host("http://localhost:11434") // Local Ollama server
        .with_timeout_seconds(120) // 2 minute timeout for local inference
        .with_max_retries(3); // Retry on connection issues

    println!("✅ Ollama adapter created successfully");
    println!("   Default model: {}", adapter.config().default_model);
    println!("   Host: {}", adapter.config().host);
    println!("   Timeout: {} seconds", adapter.config().timeout_seconds);
    println!("   Max retries: {}", adapter.config().max_retries);
    println!();

    // Example 2: Show supported models
    println!("📋 Example 2: Supported Models");
    println!("-----------------------------");

    let models = adapter.supported_models();
    println!("Available Ollama models ({}):", models.len());

    // Group models by category
    let mut base_models = Vec::new();
    let mut code_models = Vec::new();
    let mut chat_models = Vec::new();

    for model in &models {
        if model.contains("codellama") || model.contains("wizardcoder") {
            code_models.push(model);
        } else if model.contains("chat") || model.contains("neural") || model.contains("orca") {
            chat_models.push(model);
        } else {
            base_models.push(model);
        }
    }

    if !base_models.is_empty() {
        println!(
            "  🦙 Base models: {}",
            base_models
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<&str>>()
                .join(", ")
        );
    }
    if !code_models.is_empty() {
        println!(
            "  💻 Code models: {}",
            code_models
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<&str>>()
                .join(", ")
        );
    }
    if !chat_models.is_empty() {
        println!(
            "  💬 Chat models: {}",
            chat_models
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<&str>>()
                .join(", ")
        );
    }

    println!();

    // Example 3: Template-based requests for different use cases
    println!("📝 Example 3: Template-Based Requests for Local Models");
    println!("-----------------------------------------------------");

    let examples = vec![
        (
            ModelType::Coding,
            "Write a {{language}} {{component}} that {{purpose}}",
            vec![
                ("language", "Python"),
                ("component", "function"),
                ("purpose", "calculates Fibonacci numbers"),
            ],
        ),
        (
            ModelType::Reasoning,
            "Analyze the trade-offs between {{option1}} and {{option2}} for {{context}}",
            vec![
                ("option1", "local LLM deployment"),
                ("option2", "cloud API services"),
                ("context", "a small startup"),
            ],
        ),
        (
            ModelType::General,
            "Explain {{concept}} in simple terms for {{audience}}",
            vec![("concept", "machine learning"), ("audience", "beginners")],
        ),
        (
            ModelType::Fast,
            "Quick summary: {{topic}}",
            vec![("topic", "benefits of local LLM inference")],
        ),
    ];

    for (model_type, template, params_vec) in examples {
        println!("  🎯 Model Type: {:?}", model_type);
        println!("  📄 Template: \"{}\"", template);

        let mut params = HashMap::new();
        for (key, value) in params_vec {
            params.insert(key.to_string(), value.to_string());
        }

        let request = InferenceRequest::new(template, params, model_type);
        let rendered = request.render_template();

        println!("  ✨ Rendered: \"{}\"", rendered);
        println!("  🔧 Best model: codellama for code, llama2 for general tasks");
        println!();
    }

    // Example 4: Health check (will work when Ollama server is running)
    println!("🏥 Example 4: Health Check");
    println!("-------------------------");

    match adapter.health_check().await {
        Ok(health) => {
            println!("✅ Health check successful!");
            println!("   Status: {:?}", health.status);
            println!("   Service: ollama");
            if let Some(models_count) = health.metadata.get("available_models") {
                println!("   Available models: {}", models_count);
            }
        }
        Err(e) => {
            println!("❌ Health check failed: {}", e);
            println!("   This is expected if Ollama server is not running");
            println!("   On Ishtar server: Ollama should be available");
        }
    }

    println!();

    // Example 5: Token counting
    println!("🔢 Example 5: Token Estimation");
    println!("------------------------------");

    let sample_texts = vec![
        "Hello, world!",
        "Write a Python function that calculates the factorial of a number.",
        "Explain quantum computing in simple terms for a beginner audience.",
    ];

    for text in sample_texts {
        let token_count = adapter.count_tokens(text)?;
        println!("  Text: \"{}\"", text);
        println!("  Estimated tokens: {}", token_count);
        println!();
    }

    // Example 6: Configuration for different environments
    println!("⚙️ Example 6: Environment-Specific Configuration");
    println!("-----------------------------------------------");

    // Local development
    let local_config = OllamaConfig::new("llama2")
        .with_host("http://localhost:11434")
        .with_timeout_seconds(60);

    // Production server configuration (configurable via environment)
    let server_host =
        std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let server_model =
        std::env::var("OLLAMA_DEFAULT_MODEL").unwrap_or_else(|_| "codellama".to_string());

    let server_config = OllamaConfig::new(&server_model)
        .with_host(&server_host)
        .with_timeout_seconds(180) // Longer timeout for server
        .with_max_retries(5);

    println!("  🖥️  Local development:");
    println!("    Host: {}", local_config.host);
    println!("    Model: {}", local_config.default_model);

    println!("\n  🌐 Production server:");
    println!(
        "    Host: {} (from OLLAMA_HOST env var)",
        server_config.host
    );
    println!(
        "    Model: {} (from OLLAMA_DEFAULT_MODEL env var)",
        server_config.default_model
    );
    println!("    Timeout: {} seconds", server_config.timeout_seconds);

    println!();

    // Example 7: Error handling patterns
    println!("🚨 Example 7: Error Handling");
    println!("---------------------------");

    use tyl_ollama_inference_adapter::ollama_errors;

    let error_examples = vec![
        ollama_errors::connection_failed("Ollama server not running"),
        ollama_errors::model_not_found("nonexistent-model"),
        ollama_errors::generation_failed("out of memory"),
    ];

    println!("  Common error scenarios:");
    for (i, error) in error_examples.iter().enumerate() {
        println!("  {}. {}", i + 1, error);
    }

    println!("\n  💡 On production server:");
    println!("     • Set OLLAMA_HOST environment variable (e.g., http://server-name:11434)");
    println!("     • Set OLLAMA_DEFAULT_MODEL environment variable (e.g., llama2:13b)");
    println!("     • Ollama should be pre-configured and running");
    println!("     • Models should be already pulled and available");

    println!("\n✅ All examples completed successfully!");

    println!("\n📖 Key Features Demonstrated:");
    println!("• Template-based prompts with parameter substitution");
    println!("• Local LLM inference without external API calls");
    println!("• Health monitoring and server connectivity");
    println!("• Token estimation for cost and performance planning");
    println!("• Environment-specific configuration");
    println!("• Comprehensive error handling");

    println!("\n🚀 Testing on Ishtar Server:");
    println!("1. Ollama server should already be available");
    println!("2. Models like llama2, codellama should be pre-installed");
    println!("3. Use server-appropriate timeouts and retry settings");
    println!("4. Monitor resource usage during inference");
    println!("5. Test with different model types for various use cases");

    Ok(())
}
