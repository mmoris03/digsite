from digsite.crawl.throttle import HostThrottle
from fakes import FakeClock


def test_first_request_to_a_host_does_not_wait(clock: FakeClock) -> None:
    throttle = HostThrottle(2.0, clock.sleep, clock)

    throttle.wait("https://example.com")

    assert clock.sleeps == []


def test_waits_only_for_the_time_still_missing(clock: FakeClock) -> None:
    throttle = HostThrottle(2.0, clock.sleep, clock)
    throttle.mark("https://example.com")
    clock.now += 0.5

    throttle.wait("https://example.com")

    assert clock.sleeps == [1.5]


def test_does_not_wait_when_enough_time_has_passed(clock: FakeClock) -> None:
    throttle = HostThrottle(2.0, clock.sleep, clock)
    throttle.mark("https://example.com")
    clock.now += 5.0

    throttle.wait("https://example.com")

    assert clock.sleeps == []


def test_hosts_are_paced_independently(clock: FakeClock) -> None:
    throttle = HostThrottle(2.0, clock.sleep, clock)
    throttle.mark("https://example.com")

    throttle.wait("https://other.org")

    assert clock.sleeps == []


def test_a_longer_delay_requested_by_the_host_wins(clock: FakeClock) -> None:
    throttle = HostThrottle(1.0, clock.sleep, clock)
    throttle.mark("https://example.com")

    throttle.wait("https://example.com", min_delay=5.0)

    assert clock.sleeps == [5.0]


def test_mark_first_contact_does_not_overwrite_a_later_request(clock: FakeClock) -> None:
    throttle = HostThrottle(2.0, clock.sleep, clock)
    throttle.mark_first_contact("https://example.com")
    clock.now += 10.0
    throttle.mark("https://example.com")
    throttle.mark_first_contact("https://example.com")

    throttle.wait("https://example.com")

    assert clock.sleeps == [2.0]
