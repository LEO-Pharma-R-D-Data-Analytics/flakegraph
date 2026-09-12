"""Check the shim's admission queue against a real PostgreSQL server.

The ordering guarantee is the whole point of putting this queue in a database
rather than in process memory, and it is expressed in SQL — an advisory lock, a
bounded count, and a priority-ordered window. None of that is exercised by a
stub, so it is verified here.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from uuid import uuid4

import psycopg
import pytest
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from kg_processor.adapters.distributed.postgres import PostgresDistributedStore
from kg_processor.serving.ocr_shim import OcrQueue

_POSTGRES_DSN = os.getenv("KG_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(
    not _POSTGRES_DSN,
    reason="Set KG_TEST_POSTGRES_DSN to run live PostgreSQL coordination checks.",
)


@pytest.fixture
def isolated_postgres_dsn() -> Iterator[str]:
    """Give each test its own schema so parallel runs cannot collide."""

    assert _POSTGRES_DSN is not None
    schema_name = f"ocr_shim_{uuid4().hex}"
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(f'CREATE SCHEMA "{schema_name}"')
    try:
        yield make_conninfo(_POSTGRES_DSN, options=f"-c search_path={schema_name}")
    finally:
        with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
            connection.execute(f'DROP SCHEMA "{schema_name}" CASCADE')


async def _with_queue(dsn: str, owner: str, body):  # type: ignore[no-untyped-def]
    async with AsyncConnectionPool(
        dsn,
        min_size=1,
        max_size=4,
        open=False,
        kwargs={"row_factory": dict_row, "autocommit": True},
    ) as pool:
        return await body(OcrQueue(pool, owner, stale_after_seconds=60.0))


def _initialize(dsn: str) -> None:
    """Apply the coordination schema the chart's bootstrap Job would apply."""

    PostgresDistributedStore(dsn).initialize()


def test_the_queue_table_ships_with_the_coordination_schema(
    isolated_postgres_dsn: str,
) -> None:
    _initialize(isolated_postgres_dsn)

    with psycopg.connect(isolated_postgres_dsn, row_factory=dict_row) as connection:
        columns = connection.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'flakegraph_ocr_request'
            """
        ).fetchall()

    assert {row["column_name"] for row in columns} == {
        "id",
        "priority",
        "consumer_class",
        "status",
        "shim_owner",
        "created_at",
        "heartbeat_at",
    }


def test_a_full_pool_admits_nothing(isolated_postgres_dsn: str) -> None:
    _initialize(isolated_postgres_dsn)

    async def body(queue: OcrQueue) -> list[bool]:
        await queue.enqueue("first", 100, "batch")
        await queue.enqueue("second", 100, "batch")
        return [
            await queue.try_admit("first", capacity=1),
            await queue.try_admit("second", capacity=1),
        ]

    admitted = asyncio.run(_with_queue(isolated_postgres_dsn, "shim-a", body))

    assert admitted == [True, False]


def test_interactive_work_is_admitted_ahead_of_a_batch_backlog(
    isolated_postgres_dsn: str,
) -> None:
    _initialize(isolated_postgres_dsn)

    async def body(queue: OcrQueue) -> list[bool]:
        for index in range(5):
            await queue.enqueue(f"batch-{index}", 100, "batch")
        await queue.enqueue("interactive-0", 0, "interactive")
        # One free slot, six waiting. Lower is served first here, matching the
        # inference plane, so the interactive row takes it.
        return [
            await queue.try_admit("batch-0", capacity=1),
            await queue.try_admit("interactive-0", capacity=1),
        ]

    admitted = asyncio.run(_with_queue(isolated_postgres_dsn, "shim-a", body))

    assert admitted == [False, True]


def test_ordering_holds_across_two_shim_replicas(isolated_postgres_dsn: str) -> None:
    """Two replicas must not each admit into the same free slot."""

    _initialize(isolated_postgres_dsn)

    async def body(queue: OcrQueue) -> None:
        await queue.enqueue("a", 100, "batch")
        await queue.enqueue("b", 100, "batch")

    asyncio.run(_with_queue(isolated_postgres_dsn, "shim-a", body))

    async def admit(name: str) -> bool:
        async def inner(queue: OcrQueue) -> bool:
            return await queue.try_admit(name, capacity=1)

        admitted: bool = await _with_queue(isolated_postgres_dsn, f"shim-{name}", inner)
        return admitted

    async def race() -> list[bool]:
        return list(await asyncio.gather(admit("a"), admit("b")))

    results = asyncio.run(race())

    assert sum(results) == 1


def test_a_released_request_frees_its_slot(isolated_postgres_dsn: str) -> None:
    _initialize(isolated_postgres_dsn)

    async def body(queue: OcrQueue) -> list[bool]:
        await queue.enqueue("first", 100, "batch")
        await queue.enqueue("second", 100, "batch")
        first = await queue.try_admit("first", capacity=1)
        blocked = await queue.try_admit("second", capacity=1)
        await queue.release("first")
        return [first, blocked, await queue.try_admit("second", capacity=1)]

    admitted = asyncio.run(_with_queue(isolated_postgres_dsn, "shim-a", body))

    assert admitted == [True, False, True]


def test_a_dead_replicas_rows_are_reclaimed(isolated_postgres_dsn: str) -> None:
    """A shim that dies mid-request must not hold a slot forever."""

    _initialize(isolated_postgres_dsn)

    async def body(queue: OcrQueue) -> None:
        await queue.enqueue("abandoned", 100, "batch")
        assert await queue.try_admit("abandoned", capacity=1)

    asyncio.run(_with_queue(isolated_postgres_dsn, "shim-dead", body))

    with psycopg.connect(isolated_postgres_dsn, autocommit=True) as connection:
        connection.execute(
            "UPDATE flakegraph_ocr_request SET heartbeat_at = CURRENT_TIMESTAMP - interval '1 hour'"
        )

    async def survivor(queue: OcrQueue) -> bool:
        await queue.enqueue("live", 100, "batch")
        return await queue.try_admit("live", capacity=1)

    assert asyncio.run(_with_queue(isolated_postgres_dsn, "shim-live", survivor))
