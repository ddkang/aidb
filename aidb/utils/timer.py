import time
from typing import Dict


class Timer:
    def __init__(self) -> None:
        self.start_time = 0
        self.stop_time = 0
        self.events = []

    def start(self) -> float:
        self.start_time = time.time()
        return self.start_time

    def check(self, event: str) -> float:
        if len(self.events) == 0:
            self.events.append((event, time.time() - self.start_time))
        else:
            past_events_time = sum([e[1] for e in self.events])
            self.events.append(
                (event, time.time() - self.start_time - past_events_time)
            )
        return self.events[-1][1]

    def stop(self):
        self.stop_time = time.time()
        self.events.append(("runtime", self.stop_time - self.start_time))

    def get_records(self) -> Dict[str, float]:
        time_record = {}
        for event, time_taken in self.events:
            time_record[event] = time_taken
        return time_record