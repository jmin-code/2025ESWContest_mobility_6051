class Bus:
    def __init__(self): self._h={}
    def on(self, evt, fn): self._h.setdefault(evt,[]).append(fn)
    def emit(self, evt, **data):
        for fn in self._h.get(evt, []): fn(**data)