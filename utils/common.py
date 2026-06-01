
def exit(message: str | None = None, code: int = 0):
    if message:
        print(message)
    raise SystemExit(code)