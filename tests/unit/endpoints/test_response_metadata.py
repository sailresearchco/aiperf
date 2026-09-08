# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selected response metadata stays separate from content and replay."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import orjson
import pytest
from pytest import param

from aiperf.common.models import ModelEndpointInfo, ParsedResponseRecord, RequestRecord
from aiperf.common.models.record_models import TextResponse
from aiperf.config.config import BenchmarkConfig
from aiperf.config.endpoint import EndpointConfig
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.endpoints.base_endpoint import BaseEndpoint
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.endpoints.openai_completions import CompletionsEndpoint
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.decode_duration_metric import DecodeDurationMetric
from aiperf.metrics.types.inter_chunk_latency_metric import InterChunkLatencyMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.ttft_metric import TTFTMetric
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import create_model_endpoint, create_request_info

SELECTORS = {"server_queue_duration_ms": "jib_metrics.server_queue_duration_ms"}


@pytest.fixture(
    params=[
        param((ChatEndpoint, EndpointType.CHAT, True), id="chat-stream"),
        param((ChatEndpoint, EndpointType.CHAT, False), id="chat-buffered"),
        param(
            (CompletionsEndpoint, EndpointType.COMPLETIONS, True),
            id="completions-stream",
        ),
        param(
            (CompletionsEndpoint, EndpointType.COMPLETIONS, False),
            id="completions-buffered",
        ),
    ]
)
def endpoint(request: pytest.FixtureRequest) -> BaseEndpoint:
    endpoint_class, endpoint_type, streaming = request.param
    model = create_model_endpoint(endpoint_type, streaming=streaming)
    model.endpoint.response_metadata = SELECTORS
    return endpoint_class(model_endpoint=model)


def _response(body: dict[str, Any], perf_ns: int = 200) -> TextResponse:
    return TextResponse(perf_ns=perf_ns, text=orjson.dumps(body).decode())


def _content(endpoint: BaseEndpoint, text: str) -> dict[str, Any]:
    if isinstance(endpoint, ChatEndpoint):
        streaming = endpoint.model_endpoint.endpoint.streaming
        return {
            "object": "chat.completion.chunk" if streaming else "chat.completion",
            "choices": [{"delta" if streaming else "message": {"content": text}}],
        }
    return {"object": "text_completion", "choices": [{"text": text}]}


@pytest.mark.parametrize(
    "fields,expected",
    [
        param({"server_queue_duration_ms": 12.5}, 12.5, id="positive"),
        param({"server_queue_duration_ms": 0}, 0, id="zero"),
        param({"server_queue_duration_ms": None}, None, id="null"),
        param({}, None, id="missing"),
    ],
)  # fmt: skip
def test_response_metadata_preserves_selected_value(
    endpoint: BaseEndpoint, fields: dict[str, Any], expected: float | None
) -> None:
    body = _content(endpoint, "hello")
    body.update(jib_metrics=fields, unrelated="not selected")
    parsed = endpoint.parse_response(_response(body))
    assert parsed.metadata == {"server_queue_duration_ms": expected}
    assert parsed.data.get_text() == "hello"


def test_metadata_only_response_has_no_content(endpoint: BaseEndpoint) -> None:
    parsed = endpoint.parse_response(
        _response({"jib_metrics": {"server_queue_duration_ms": 0}})
    )
    assert parsed.data is None
    assert parsed.metadata == {"server_queue_duration_ms": 0}


def test_metadata_default_does_not_capture_response_fields(
    endpoint: BaseEndpoint,
) -> None:
    model = endpoint.model_endpoint.model_copy(deep=True)
    model.endpoint.response_metadata = {}
    disabled = type(endpoint)(model_endpoint=model)
    assert (
        disabled.parse_response(
            _response({"jib_metrics": {"server_queue_duration_ms": 1}})
        )
        is None
    )
    parsed = disabled.parse_response(_response(_content(disabled, "hello")))
    assert parsed.metadata == {}


def test_metadata_queries_are_compiled_before_parsing(endpoint: BaseEndpoint) -> None:
    with patch("jmespath.compile", side_effect=AssertionError("compiled per response")):
        endpoint.parse_response(_response(_content(endpoint, "first")))
        endpoint.parse_response(_response(_content(endpoint, "second")))


def test_metadata_does_not_change_content_timing_usage_or_replay(
    endpoint: BaseEndpoint,
) -> None:
    record = RequestRecord(
        start_perf_ns=100,
        end_perf_ns=500,
        responses=[
            _response({"jib_metrics": {"server_queue_duration_ms": 10}}, 150),
            _response(_content(endpoint, "hello"), 200),
            _response(_content(endpoint, " world"), 300),
            _response({"usage": {"prompt_tokens": 5, "completion_tokens": 2}}, 400),
        ],
    )
    parsed, assistant = endpoint.process_responses(record, capture_assistant_turn=True)
    assert len(parsed) == 4
    assert parsed[-1].metadata == {"server_queue_duration_ms": None}
    assert assistant.texts[0].contents == ["hello world"]
    result = ParsedResponseRecord(request=record, responses=parsed)
    assert [response.perf_ns for response in result.content_responses] == [200, 300]
    assert result.final_usage.prompt_tokens == 5
    assert result.final_usage.completion_tokens == 2
    metrics = MetricRecordDict()
    for metric in (
        TTFTMetric(),
        RequestLatencyMetric(),
        DecodeDurationMetric(),
        InterChunkLatencyMetric(),
    ):
        metrics[metric.tag] = metric.parse_record(result, metrics)
    assert metrics == {
        "time_to_first_token": 100,
        "request_latency": 200,
        "decode_duration": 100,
        "inter_chunk_latency": [100],
    }


def test_response_metadata_is_not_sent_in_payload(endpoint: BaseEndpoint) -> None:
    info = create_request_info(endpoint.model_endpoint, texts=["hello"])
    payload = endpoint.format_payload(info)
    assert "response_metadata" not in payload
    assert "responseMetadata" not in payload
    assert "server_queue_duration_ms" not in payload


@pytest.mark.parametrize("alias", ["response_metadata", "responseMetadata"])
def test_response_metadata_config_alias_reaches_runtime(
    alias: str, tmp_path: Path
) -> None:
    cfg = BenchmarkConfig.model_validate(
        {
            "model": "test-model",
            "endpoint": {"url": "http://localhost:8000", alias: SELECTORS},
            "dataset": {"type": "synthetic", "prompts": {"isl": 5, "osl": 2}},
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 1,
                    "requests": 1,
                }
            ],
        }
    )
    assert cfg.endpoint.model_dump(by_alias=True)["responseMetadata"] == SELECTORS
    run = BenchmarkRun(benchmark_id="test", cfg=cfg, artifact_dir=tmp_path)
    info = ModelEndpointInfo.from_run(run)
    assert info.endpoint.response_metadata == SELECTORS
    restored = ModelEndpointInfo.model_validate_json(info.model_dump_json())
    assert restored.endpoint.response_metadata == SELECTORS


@pytest.mark.parametrize("expression", ["", "unclosed["])
def test_response_metadata_invalid_expression_fails_config(expression: str) -> None:
    with pytest.raises(ValueError, match="response_metadata.queue"):
        EndpointConfig(
            urls=["http://localhost:8000"], response_metadata={"queue": expression}
        )


def test_response_metadata_default_is_empty() -> None:
    assert EndpointConfig(urls=["http://localhost:8000"]).response_metadata == {}
