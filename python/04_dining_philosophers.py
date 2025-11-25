"""
PROBLEM: Dining Philosophers Problem

Implement a solution to the classic Dining Philosophers synchronization problem.
Five philosophers sit around a table with a chopstick between each pair. Each
philosopher needs to pick up both adjacent chopsticks to eat. The challenge is
to avoid deadlock while ensuring all philosophers can eat.

REQUIREMENTS:
- Implement 5 philosophers and 5 chopsticks
- Each philosopher must pick up both adjacent chopsticks to eat
- Must avoid deadlock (circular wait condition)
- Must ensure no philosopher starves (fairness)
- Should use proper synchronization primitives

PERFORMANCE NOTES:
- All philosophers should be able to eat within a reasonable time
- Should handle eating/thinking cycles for extended periods
- Minimal lock contention

TEST CASE EXPECTATIONS:
- Simulation should run without deadlock for 30+ seconds
- Each philosopher should eat multiple times
- No philosopher should be completely starved
"""

import threading
import time
from typing import Optional


class Chopstick:
    """Represents a chopstick that can be picked up by a philosopher."""

    def __init__(self, chopstick_id: int):
        self.chopstick_id = chopstick_id
        # TODO: Add lock for synchronization
        pass

    def pickup(self, philosopher_id: int, timeout: Optional[float] = None) -> bool:
        """
        Attempt to pick up the chopstick.

        Args:
            philosopher_id: ID of the philosopher trying to pick up
            timeout: Maximum time to wait

        Returns:
            True if successfully picked up, False if timeout
        """
        # TODO: Implement pickup with timeout
        pass

    def putdown(self, philosopher_id: int):
        """
        Put down the chopstick.

        Args:
            philosopher_id: ID of the philosopher putting down
        """
        # TODO: Implement putdown
        pass


class Philosopher:
    """Represents a philosopher in the dining philosophers problem."""

    def __init__(
        self,
        philosopher_id: int,
        left_chopstick: Chopstick,
        right_chopstick: Chopstick,
        duration: float = 10.0,
    ):
        self.philosopher_id = philosopher_id
        self.left_chopstick = left_chopstick
        self.right_chopstick = right_chopstick
        self.duration = duration
        self.times_eaten = 0
        self.total_eating_time = 0.0
        self.start_time = None
        self.running = False

    def think(self):
        """Philosopher thinks (do nothing for a random duration)."""
        think_time = 0.01
        time.sleep(think_time)

    def eat(self):
        """Philosopher eats (requires both chopsticks)."""
        eat_time = 0.02
        time.sleep(eat_time)

    def run(self):
        """Main philosopher thread - alternates between thinking and eating."""
        self.running = True
        self.start_time = time.time()

        while time.time() - self.start_time < self.duration:
            # Think
            self.think()

            # Try to eat
            # TODO: Implement logic to pick up both chopsticks
            # TODO: Must avoid deadlock - consider using timeout or ordering
            # For example, always pick up lower-numbered chopstick first
            # Or use timeout and put down if can't get both within time limit

            # TODO: Implement the eat and putdown logic
            self.eat()
            self.times_eaten += 1

    def start_dining(self):
        """Start the philosopher thread."""
        thread = threading.Thread(target=self.run)
        thread.daemon = True
        thread.start()
        return thread


class DiningPhilosophersSimulation:
    """Manages the dining philosophers simulation."""

    def __init__(self, num_philosophers: int = 5, duration: float = 10.0):
        self.num_philosophers = num_philosophers
        self.duration = duration
        self.chopsticks = [Chopstick(i) for i in range(num_philosophers)]
        self.philosophers = []

        # Create philosophers
        for i in range(num_philosophers):
            left = self.chopsticks[i]
            right = self.chopsticks[(i + 1) % num_philosophers]
            philosopher = Philosopher(i, left, right, duration)
            self.philosophers.append(philosopher)

    def start(self):
        """Start the simulation."""
        print(f"Starting dining philosophers simulation ({self.num_philosophers} philosophers, {self.duration}s)")
        threads = []
        for philosopher in self.philosophers:
            thread = philosopher.start_dining()
            threads.append(thread)

        # Wait for all philosophers to finish
        for thread in threads:
            thread.join()

        return self.philosophers

    def print_stats(self):
        """Print statistics about the simulation."""
        print("\nSimulation Complete - Statistics:")
        print("-" * 50)

        total_eaten = 0
        for philosopher in self.philosophers:
            print(
                f"Philosopher {philosopher.philosopher_id}: "
                f"ate {philosopher.times_eaten} times"
            )
            total_eaten += philosopher.times_eaten

        avg_eaten = total_eaten / len(self.philosophers)
        print("-" * 50)
        print(f"Total meals: {total_eaten}")
        print(f"Average meals per philosopher: {avg_eaten:.1f}")

        # Check for starvation
        min_eaten = min(p.times_eaten for p in self.philosophers)
        max_eaten = max(p.times_eaten for p in self.philosophers)

        if min_eaten == 0:
            print("⚠ WARNING: At least one philosopher starved!")
        elif max_eaten - min_eaten > avg_eaten * 0.5:
            print("⚠ WARNING: Unfair distribution - some philosophers eat much more than others")
        else:
            print("✓ Fair distribution of eating")

        return total_eaten


def test_no_deadlock():
    """Test that the simulation completes without deadlock."""
    simulation = DiningPhilosophersSimulation(num_philosophers=5, duration=5.0)
    start = time.time()
    simulation.start()
    elapsed = time.time() - start

    print(f"\n✓ No deadlock test passed (completed in {elapsed:.2f}s)")
    simulation.print_stats()


def test_fairness():
    """Test that philosophers eat roughly fairly."""
    simulation = DiningPhilosophersSimulation(num_philosophers=5, duration=10.0)
    simulation.start()

    total_eaten = simulation.print_stats()

    # Each philosopher should eat at least once
    for philosopher in simulation.philosophers:
        assert philosopher.times_eaten > 0, (
            f"Philosopher {philosopher.philosopher_id} never ate (starvation detected)"
        )

    print("\n✓ Fairness test passed")


def test_concurrent_eating():
    """Test that philosophers can eat concurrently."""
    simulation = DiningPhilosophersSimulation(num_philosophers=5, duration=5.0)
    simulation.start()
    simulation.print_stats()

    total_eaten = sum(p.times_eaten for p in simulation.philosophers)
    assert total_eaten > 0, "No philosopher ate"

    print("\n✓ Concurrent eating test passed")


if __name__ == "__main__":
    test_no_deadlock()
    print("\n" + "=" * 50 + "\n")
    test_fairness()
    print("\n" + "=" * 50 + "\n")
    test_concurrent_eating()
    print("\n✓ All tests passed!")
