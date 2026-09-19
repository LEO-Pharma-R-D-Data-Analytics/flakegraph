"""Run independent provider calls concurrently and hand the results back in order."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from kg_processor.application.window_errors import is_systemic_provider_error


def ordered_map[TItem, TResult](
    items: Sequence[TItem],
    operation: Callable[[int, TItem], TResult],
    *,
    parallelism: int,
    on_complete: Callable[[int, TResult], None] | None = None,
    contain: Callable[[int, TItem, Exception], TResult] | None = None,
) -> list[TResult]:
    """Apply ``operation`` to every item and return the results in input order.

    Provider latency decides completion order, so ``on_complete`` fires as calls
    finish (with the running count) while the returned list follows the input;
    persisted traces and merge order therefore never depend on scheduling.

    ``contain`` turns one failed call into a substitute result so the rest of
    the batch is kept. It is never offered a systemic failure (authentication,
    configuration): that ends the whole map, and the calls not yet started are
    cancelled so no more provider capacity is spent on a run that cannot
    recover. Running calls keep their own provider timeout.

    A pool of one worker runs submissions in order, so there is no separate
    serial path.
    """

    if parallelism <= 0:
        raise ValueError("parallelism must be positive")
    if not items:
        return []
    results: dict[int, TResult] = {}
    executor = ThreadPoolExecutor(max_workers=min(parallelism, len(items)))
    try:
        futures = {
            executor.submit(operation, index, item): index for index, item in enumerate(items)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                if contain is None or is_systemic_provider_error(exc):
                    raise
                result = contain(index, items[index], exc)
            results[index] = result
            if on_complete is not None:
                on_complete(completed, result)
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    return [results[index] for index in range(len(items))]
