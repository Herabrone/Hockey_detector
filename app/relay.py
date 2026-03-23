from __future__ import annotations


class PrintRelay:
    def trigger(self) -> None:
        print("LIGHT ON")


class UsbSerialRelay:
    def __init__(self, port: str, baudrate: int, command: str) -> None:
        self.port = port
        self.baudrate = baudrate
        self.command = command
        self._serial = None

    def _connect(self) -> None:
        if self._serial is not None:
            return
        import serial  # Imported lazily so print mode works without pyserial.

        self._serial = serial.Serial(self.port, self.baudrate, timeout=1)

    def trigger(self) -> None:
        self._connect()
        assert self._serial is not None
        self._serial.write(self.command.encode("utf-8"))

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None
