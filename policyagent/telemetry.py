"""OpenTelemetry tracing hooks.

Spans are always created through the OpenTelemetry API, which is a no-op until
a provider is configured. When APPLICATIONINSIGHTS_CONNECTION_STRING is set the
Azure Monitor distro exports them to Application Insights; otherwise setting
POLICYAGENT_TRACE_CONSOLE=1 prints them locally. Question text is never put on
spans: traces are retained longer and seen by more people than the app itself.
"""

from __future__ import annotations

import logging
import os

from opentelemetry import trace

from .config import Settings

log = logging.getLogger(__name__)
_configured = False


def setup_tracing(settings: Settings) -> str:
    """Configure an exporter once per process. Returns the exporter name for logging/health."""
    global _configured
    if _configured:
        return "already-configured"
    _configured = True
    if settings.appinsights_connection_string and not settings.force_offline:
        try:
            from azure.monitor.opentelemetry import configure_azure_monitor

            configure_azure_monitor(connection_string=settings.appinsights_connection_string)
            return "azure-monitor"
        except ImportError:
            log.warning("azure-monitor-opentelemetry not installed; tracing disabled")
            return "none"
    if os.environ.get("POLICYAGENT_TRACE_CONSOLE") == "1":
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        return "console"
    return "none"


tracer = trace.get_tracer("policyagent")
