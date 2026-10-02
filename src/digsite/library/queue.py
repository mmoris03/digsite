"""Building collections in the background, one at a time, and telling how it goes."""

import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace

import httpx

from digsite.embedding import DEFAULT_MODEL
from digsite.library.build import BuildError, BuildRequest, BuildStage, build_collection
from digsite.library.library import Library, new_collection_id

logger = logging.getLogger(__name__)

_FINISHED = (BuildStage.DONE, BuildStage.FAILED)


@dataclass(frozen=True, slots=True)
class BuildStatus:
    """Where a build stands.

    Attributes:
        id: Identifies the build.
        collection_id: The collection it builds. Reserved when the build is
            asked for, so that two builds never write the same file.
        url: Where its crawl starts.
        stage: What it is doing, or how it ended.
        done: How far the stage has got.
        total: Out of how much; 0 if unknown. For crawling, an upper bound.
        error: Why it failed, if it did.
        submitted: When it was asked for, in seconds since the epoch.
        finished: When it ended, if it has.
    """

    id: str
    collection_id: str
    url: str
    stage: BuildStage
    done: int = 0
    total: int = 0
    error: str | None = None
    submitted: float = 0.0
    finished: float | None = None

    @property
    def active(self) -> bool:
        return self.stage not in _FINISHED


class BuildQueue:
    """Builds collections in a background thread, in the order they were asked for.

    One build runs at a time: two would compete for the same processor and
    finish no sooner. The thread is a daemon, so stopping the program stops a
    build midway; `Library.remove_partial_builds` cleans up after it.

    Args:
        library: Where the collections are written, and the shared embedding model.
        embedding_model: Name of the model to embed passages with. None builds
            only the lexical index.
        client_factory: Makes the HTTP client of each crawl, which is closed
            after it. None uses the crawler's own.
        delay_seconds: Interval between two requests to a website.
        sleep: Blocking sleep function; injected so that tests do not wait.
    """

    def __init__(
        self,
        library: Library,
        *,
        embedding_model: str | None = DEFAULT_MODEL,
        client_factory: Callable[[], httpx.Client] | None = None,
        delay_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._library = library
        self._embedding_model = embedding_model
        self._client_factory = client_factory
        self._delay_seconds = delay_seconds
        self._sleep = sleep
        self._lock = threading.Lock()
        self._statuses: dict[str, BuildStatus] = {}
        self._requests: dict[str, BuildRequest] = {}
        self._finished: dict[str, threading.Event] = {}
        self._pending: queue.Queue[str] = queue.Queue()
        self._worker: threading.Thread | None = None

    def submit(self, request: BuildRequest) -> BuildStatus:
        """Ask for a collection to be built, after those asked for before it."""
        with self._lock:
            reserved = {status.collection_id for status in self._statuses.values() if status.active}
            collection_id = new_collection_id(request.url, {*self._library.ids(), *reserved})
            status = BuildStatus(
                id=uuid.uuid4().hex[:12],
                collection_id=collection_id,
                url=request.url,
                stage=BuildStage.QUEUED,
                submitted=time.time(),
            )
            self._statuses[status.id] = status
            self._requests[status.id] = request
            self._finished[status.id] = threading.Event()
            if self._worker is None:
                self._worker = threading.Thread(target=self._work, name="builds", daemon=True)
                self._worker.start()
        self._pending.put(status.id)
        return status

    def statuses(self) -> list[BuildStatus]:
        """Every build asked for since the program started, the latest first."""
        with self._lock:
            return sorted(self._statuses.values(), key=lambda status: -status.submitted)

    def status(self, build_id: str) -> BuildStatus | None:
        with self._lock:
            return self._statuses.get(build_id)

    def wait(self, build_id: str, timeout: float | None = None) -> BuildStatus:
        """Wait for a build to end, and say how it ended.

        Raises:
            KeyError: If there is no such build.
            TimeoutError: If it did not end in time.
        """
        if not self._finished[build_id].wait(timeout):
            raise TimeoutError(f"build {build_id} did not end in {timeout} s")
        status = self.status(build_id)
        assert status is not None
        return status

    def _advance(self, build_id: str, stage: BuildStage, done: int, total: int) -> None:
        with self._lock:
            status = self._statuses[build_id]
            self._statuses[build_id] = replace(status, stage=stage, done=done, total=total)

    def _end(self, build_id: str, error: str | None = None) -> None:
        stage = BuildStage.FAILED if error else BuildStage.DONE
        with self._lock:
            status = self._statuses[build_id]
            self._statuses[build_id] = replace(
                status, stage=stage, error=error, finished=time.time()
            )

    def _work(self) -> None:
        while True:
            build_id = self._pending.get()
            try:
                self._run(build_id)
            finally:
                self._finished[build_id].set()

    def _run(self, build_id: str) -> None:
        status = self.status(build_id)
        assert status is not None
        request = self._requests[build_id]

        client: httpx.Client | None = None
        try:
            if self._client_factory is not None:
                client = self._client_factory()
            embedder = (
                self._library.embedder(self._embedding_model) if self._embedding_model else None
            )
            build_collection(
                request,
                self._library.path_of(status.collection_id),
                embedder=embedder,
                client=client,
                progress=lambda stage, done, total: self._advance(build_id, stage, done, total),
                delay_seconds=self._delay_seconds,
                sleep=self._sleep,
            )
            # Load it now, so that its first search does not wait for that.
            self._library.corpus(status.collection_id)
        except BuildError as error:
            logger.warning("could not build %s: %s", status.collection_id, error)
            self._end(build_id, error=str(error))
        except Exception as error:
            logger.exception("could not build %s", status.collection_id)
            self._end(build_id, error=f"unexpected error: {error}")
        else:
            self._end(build_id)
        finally:
            if client is not None:
                client.close()
