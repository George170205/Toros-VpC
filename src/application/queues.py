import queue
import threading
import collections
from typing import Generic, TypeVar, Optional

T = TypeVar("T")

class TaskQueue(Generic[T]):
    """Cola acotada genérica para tareas biométricas y comandos de persistencia."""
    def __init__(self, maxsize: int = 100):
        self._queue = queue.Queue(maxsize=maxsize)
        self._maxsize = maxsize

    def put(self, item: T, block: bool = True, timeout: Optional[float] = None) -> None:
        self._queue.put(item, block=block, timeout=timeout)

    def get(self, timeout: Optional[float] = None) -> T:
        return self._queue.get(block=True, timeout=timeout)

    def try_put(self, item: T) -> bool:
        try:
            self._queue.put(item, block=False)
            return True
        except queue.Full:
            return False

    def qsize(self) -> int:
        return self._queue.qsize()

    def capacity(self) -> int:
        return self._maxsize

    def task_done(self) -> None:
        self._queue.task_done()


class LatestFrameQueue:
    """Cola acotada específica para video en tiempo real. Si está llena, descarta el frame más antiguo."""
    def __init__(self, maxsize: int = 3):
        self._maxsize = maxsize
        self._deque = collections.deque(maxlen=maxsize)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._closed = False

    def put(self, item: any) -> None:
        with self._lock:
            if self._closed:
                return
            # Si se supera el límite, deque automáticamente descarta el elemento más viejo (el de la izquierda)
            # cuando agregamos a la derecha
            self._deque.append(item)
            self._condition.notify()

    def get(self, timeout: Optional[float] = None) -> any:
        with self._lock:
            while len(self._deque) == 0:
                if self._closed:
                    raise queue.Empty("La cola de frames ha sido cerrada.")
                if not self._condition.wait(timeout=timeout):
                    raise queue.Empty("Timeout al obtener frame.")
            return self._deque.popleft()

    def qsize(self) -> int:
        with self._lock:
            return len(self._deque)

    def capacity(self) -> int:
        return self._maxsize

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._deque.clear()
            self._condition.notify_all()
