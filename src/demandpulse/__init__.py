from importlib.metadata import version


def main() -> None:
    print(f"demandpulse {version('demandpulse')}")
