from pathlib import Path

import httpx


class Shakespeare:
    url: str = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"  # noqa: E501
    )
    raw_data_path: Path = Path("datasets/shakespeare.txt")

    def __init__(self):
        if not self.raw_data_path.exists():
            self.download()
        self.text = self.raw_data_path.read_text()

    def download(self):
        raw_data = httpx.get(self.url).text
        with open(self.raw_data_path, "wb") as f:
            f.write(raw_data.encode("utf-8"))
