# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-window end for streams whose generated tokens do not all stream text.

ITL with server token counts must divide the window that covers every counted
token. A server can stream chunks whose delta is empty but which still advance
generation (tokens that decode to "", text held back by a detokenizer, and the
terminal finish_reason chunk of a tail that produced no text, e.g. past EOS
under ignore_eos). Those chunks end the window; usage-only/[DONE] chunks do not.
"""

import pytest
from pytest import approx

from aiperf.common.models import ParsedResponse, ParsedResponseRecord, RequestRecord
from aiperf.common.models.record_models import TextResponseData, TokenCounts
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.metrics.types.decode_duration_metric import DecodeDurationMetric
from aiperf.metrics.types.generation_latency_metric import GenerationLatencyMetric
from aiperf.metrics.types.inter_token_latency_metric import InterTokenLatencyMetric
from aiperf.metrics.types.output_token_throughput_metrics import (
    OutputTokenThroughputPerUserMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
)
from tests.unit.metrics.conftest import (
    _create_test_request_info,
    run_simple_metrics_pipeline,
)

START_NS = 1_000


def _text(perf_ns: int, text: str = "x") -> ParsedResponse:
    return ParsedResponse(
        perf_ns=perf_ns, data=TextResponseData(text=text), choice_update=True
    )


def _empty_delta(perf_ns: int) -> ParsedResponse:
    return ParsedResponse(perf_ns=perf_ns, choice_update=True)


def _usage_only(perf_ns: int, completion_tokens: int) -> ParsedResponse:
    return ParsedResponse(
        perf_ns=perf_ns,
        usage={"prompt_tokens": 8, "completion_tokens": completion_tokens},
    )


def _record(
    responses: list[ParsedResponse],
    output_tokens: int,
    *,
    from_server_usage: bool = True,
) -> ParsedResponseRecord:
    request = RequestRecord(
        request_info=_create_test_request_info(),
        model_name="test-model",
        start_perf_ns=START_NS,
        timestamp_ns=START_NS,
        end_perf_ns=responses[-1].perf_ns,
    )
    return ParsedResponseRecord(
        request=request,
        responses=responses,
        token_counts=TokenCounts(
            input=8, output=output_tokens, from_server_usage=from_server_usage
        ),
    )


def _metrics(record: ParsedResponseRecord):
    return run_simple_metrics_pipeline(
        [record],
        RequestLatencyMetric.tag,
        GenerationLatencyMetric.tag,
        DecodeDurationMetric.tag,
        InterTokenLatencyMetric.tag,
        OutputTokenThroughputPerUserMetric.tag,
    )


class TestGenerationEnd:
    def test_normal_stream_is_unchanged(self):
        """(a) Every token streams text: window, ITL and TPS/user are unchanged."""
        responses = [_text(1_100), _text(1_200), _text(1_300), _usage_only(1_310, 3)]
        record = _record(responses, output_tokens=3)

        results = _metrics(record)

        assert record.generation_end_perf_ns == 1_300
        assert results[RequestLatencyMetric.tag] == [300]
        assert results[GenerationLatencyMetric.tag] == [300]
        # (300 - 100) / (3 - 1) = 100 ns, the pre-change value.
        assert results[InterTokenLatencyMetric.tag] == approx([100.0])
        assert results[OutputTokenThroughputPerUserMetric.tag] == approx([1e7])

    def test_finish_reason_on_last_content_chunk_is_unchanged(self):
        """(a) vLLM-style: finish_reason rides the last content delta. Exact no-op."""
        responses = [_text(1_100), _text(1_200), _text(1_300), _usage_only(1_301, 3)]
        record = _record(responses, output_tokens=3)

        results = _metrics(record)

        assert results[GenerationLatencyMetric.tag] == results[RequestLatencyMetric.tag]
        assert results[InterTokenLatencyMetric.tag] == approx([100.0])

    def test_natural_eos_finish_chunk_adds_the_eos_step(self):
        """Natural EOS: the EOS token is counted in completion_tokens but streams no
        text, so its separate finish_reason chunk (one decode step later) ends the
        window. The change is bounded by that one step."""
        responses = [
            _text(1_100),
            _text(1_200),
            _text(1_300),
            _empty_delta(1_400),  # finish_reason="stop", delta {}
            _usage_only(1_401, 4),
        ]
        record = _record(responses, output_tokens=4)

        results = _metrics(record)

        assert results[RequestLatencyMetric.tag] == [300]
        assert results[GenerationLatencyMetric.tag] == [400]
        # (400 - 100) / (4 - 1) = 100 ns: every counted token, including EOS,
        # has its step inside the window. The old window gave (300 - 100) / 3.
        assert results[InterTokenLatencyMetric.tag] == approx([100.0])

    def test_trailing_empty_deltas_extend_window(self):
        """(b) Generated tokens after the last text chunk keep the window open."""
        responses = [
            _text(1_100),
            _text(1_200),
            _empty_delta(1_300),
            _empty_delta(1_400),
            _empty_delta(1_500),  # terminal finish_reason chunk with delta {}
            _usage_only(1_510, 5),
        ]
        record = _record(responses, output_tokens=5)

        results = _metrics(record)

        assert results[RequestLatencyMetric.tag] == [200]  # unchanged contract
        assert results[GenerationLatencyMetric.tag] == [500]
        # Old window: (200 - 100) / 4 = 25 ns. Generation window: (500 - 100) / 4.
        assert results[InterTokenLatencyMetric.tag] == approx([100.0])
        assert results[OutputTokenThroughputPerUserMetric.tag] == approx([1e7])
        # Decode Duration keeps its first-to-last-content definition.
        assert results[DecodeDurationMetric.tag] == [100]

    def test_trailing_usage_only_chunk_is_excluded(self):
        """(c) A late usage-only chunk does not stretch the window."""
        responses = [_text(1_100), _text(1_200), _text(1_300), _usage_only(9_000, 3)]
        record = _record(responses, output_tokens=3)

        results = _metrics(record)

        assert record.generation_end_perf_ns == 1_300
        assert results[GenerationLatencyMetric.tag] == [300]
        assert results[InterTokenLatencyMetric.tag] == approx([100.0])

    def test_interior_empty_deltas_do_not_change_window(self):
        """(d) Held-back UTF-8 between text chunks: the window still ends at the last text."""
        responses = [
            _text(1_100),
            _empty_delta(1_150),
            _empty_delta(1_175),
            _text(1_200, "é"),
            _text(1_300),
            _usage_only(1_310, 5),
        ]
        record = _record(responses, output_tokens=5)

        results = _metrics(record)

        assert results[GenerationLatencyMetric.tag] == [300]
        assert results[RequestLatencyMetric.tag] == [300]
        assert results[InterTokenLatencyMetric.tag] == approx([50.0])

    def test_client_token_counts_keep_last_content_window(self):
        """Client-side counts only see streamed text, so they pair with request latency."""
        responses = [_text(1_100), _text(1_200), _empty_delta(1_500)]
        record = _record(responses, output_tokens=3, from_server_usage=False)

        results = _metrics(record)

        assert results[GenerationLatencyMetric.tag] == [500]
        # (200 - 100) / (3 - 1)
        assert results[InterTokenLatencyMetric.tag] == approx([50.0])

    def test_no_content_has_no_generation_latency(self):
        record = _record([_empty_delta(1_100), _usage_only(1_200, 1)], output_tokens=1)
        results = _metrics(record)
        assert (
            GenerationLatencyMetric.tag not in results
            or not results[GenerationLatencyMetric.tag]
        )


class TestChatStreamIgnoreEos:
    """Parse a jib-style chat stream end to end: text stops at EOS, the server
    keeps generating to min_tokens without streaming text, then sends a
    finish_reason chunk with ``delta: {}``, a usage-only chunk, and [DONE]."""

    @pytest.fixture
    def endpoint(self):
        return create_endpoint_with_mock_transport(
            ChatEndpoint, create_model_endpoint(EndpointType.CHAT)
        )

    def test_finish_chunk_ends_window(self, endpoint):
        chunk = "chat.completion.chunk"
        raw = [
            (1_050, {"object": chunk, "choices": [{"delta": {"role": "assistant"}}]}),
            (1_100, {"object": chunk, "choices": [{"delta": {"content": "a"}}]}),
            (1_200, {"object": chunk, "choices": [{"delta": {"content": "b"}}]}),
            (1_300, {"object": chunk, "choices": [{"delta": {"content": "c"}}]}),
            # 7 more tokens are generated after EOS; none stream text.
            (
                2_000,
                {
                    "object": chunk,
                    "choices": [{"delta": {}, "finish_reason": "length"}],
                },
            ),
            (
                2_001,
                {
                    "object": chunk,
                    "choices": [],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 10},
                },
            ),
        ]
        responses = [
            parsed
            for perf_ns, obj in raw
            if (parsed := endpoint.parse_response(create_mock_response(perf_ns, obj)))
        ]
        record = _record(responses, output_tokens=10)

        results = _metrics(record)

        assert results[RequestLatencyMetric.tag] == [300]
        assert results[GenerationLatencyMetric.tag] == [1_000]
        # TTFT = 100. Old: (300 - 100) / 9 = 22.2 ns. Fixed: (1000 - 100) / 9 = 100 ns.
        assert results[InterTokenLatencyMetric.tag] == approx([100.0])
