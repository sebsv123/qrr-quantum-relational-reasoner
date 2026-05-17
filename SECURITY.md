# Security Policy

## Scope

QRR is a research codebase. Security vulnerabilities most likely to apply:
- Malicious model weights loaded via HuggingFace (pickle deserialization)
- Prompt injection in agent loop examples
- Dependency vulnerabilities in PyTorch / Transformers

## Reporting a Vulnerability

Please **do not** open a public issue for security vulnerabilities.

Instead, contact the maintainer directly via GitHub private message or email listed on the profile.

We will respond within 5 business days and coordinate a fix before public disclosure.

## Best Practices for Users

- Only load model weights from trusted sources
- Pin dependency versions in production environments
- Treat agent tool outputs as untrusted input
