# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class GenerationLatencyMetric(BaseRecordMetric[int]):
    """
    Post-processor for the time until the last streamed generation update.

    Formula:
        Generation Latency = Last Generation Update Timestamp - Request Start Timestamp

    A generation update is a streamed response with content or with an update
    for the first choice whose text is empty: a token that decoded to ``""``,
    text held back by an incremental detokenizer, or the terminal
    ``finish_reason`` chunk. Trailing usage-only and ``[DONE]`` chunks are
    excluded. It equals Request Latency when every generated token streams text
    and the finish reason arrives with the last content, and is later when the
    stream's tail produced no text (e.g. tokens past EOS under ``ignore_eos``).
    Inter-token latency uses it as the end of the decode window when output
    tokens come from server-reported usage.
    """

    tag = "generation_latency"
    header = "Generation Latency"
    short_header = "Gen Latency"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    display_order = 310
    flags = MetricFlags.STREAMING_TOKENS_ONLY
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """Return the interval from request start to the last generation update."""
        generation_end_ns = record.generation_end_perf_ns
        if generation_end_ns is None or not record.content_responses:
            raise NoMetricValue(
                "Generation latency requires at least 1 non-empty content response."
            )
        if generation_end_ns < record.start_perf_ns:
            raise ValueError(
                "Last generation update timestamp is less than request timestamp."
            )
        return generation_end_ns - record.start_perf_ns
