"""Async request coalescing and bounded exact-result cache, no ML dependencies."""
import asyncio
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor


class RetrievalBatcher:
    def __init__(self, search, max_batch=8, wait_ms=20, cache_size=10000, queue_size=1024, on_batch=None):
        if max_batch < 1 or wait_ms < 0 or cache_size < 0:
            raise ValueError('Invalid batching settings')
        self.search = search
        self.max_batch = max_batch
        self.wait_seconds = wait_ms / 1000
        self.cache_size = cache_size
        self.on_batch = on_batch
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.pending = {}
        self.cache = OrderedDict()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='retrieval-batch')
        self.worker = None
        self.closed = False
        self.stats = dict(requests=0, cache_hits=0, coalesced=0, batches=0, searched_queries=0)

    def start(self):
        self.worker = asyncio.create_task(self._run())

    async def query(self, query, topk):
        if self.closed:
            raise RuntimeError('Retriever is shutting down')
        self.stats['requests'] += 1
        key = (query, topk)
        if key in self.cache:
            self.stats['cache_hits'] += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        future = self.pending.get(key)
        if future is not None:
            self.stats['coalesced'] += 1
        else:
            if self.queue.full():
                raise RuntimeError('Retriever request queue is full')
            future = asyncio.get_running_loop().create_future()
            # Retrieve exceptions even if every HTTP client disconnects.
            future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
            self.pending[key] = future
            self.queue.put_nowait((key, future, time.monotonic()))
        return await asyncio.shield(future)

    async def _run(self):
        while True:
            first = await self.queue.get()
            batch = [first]
            deadline = time.monotonic() + self.wait_seconds
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        batch.append(await asyncio.wait_for(self.queue.get(), remaining))
                    except asyncio.TimeoutError:
                        break
            start = time.monotonic()
            success = False
            try:
                keys = [item[0] for item in batch]
                outputs = await asyncio.get_running_loop().run_in_executor(self.executor, self.search, keys)
                if len(outputs) != len(batch):
                    raise ValueError('Batch result count does not match input query count')
                for (key, future, _), result in zip(batch, outputs, strict=True):
                    if self.cache_size:
                        self.cache[key] = result
                        self.cache.move_to_end(key)
                        while len(self.cache) > self.cache_size:
                            self.cache.popitem(last=False)
                    if not future.done():
                        future.set_result(result)
                success = True
            except Exception as exc:
                for _, future, _ in batch:
                    if not future.done():
                        future.set_exception(exc)
            finally:
                for key, _, _ in batch:
                    self.pending.pop(key, None)
                    self.queue.task_done()
                self.stats['batches'] += 1
                self.stats['searched_queries'] += len(batch)
                if self.on_batch:
                    self.on_batch(dict(time=time.time(), batch_size=len(batch), success=success,
                        service_seconds=time.monotonic()-start,
                        mean_queue_seconds=sum(start-item[2] for item in batch)/len(batch)))

    async def close(self):
        self.closed = True
        await self.queue.join()
        if self.worker:
            self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                pass
        self.executor.shutdown(wait=True)
