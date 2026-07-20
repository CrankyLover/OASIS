import os
import re
import time
import threading
import requests

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List


class LLM_Oracle:
    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o",
        max_workers: int = 10,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
    ):
        self.api_key = api_key
        self.model = model
        self.max_workers = max_workers
        self.base_url = base_url

        self.token_window = []
        self.token_lock = threading.Lock()
        self.token_limit = 150_000


    def _acquire_token(self, tokens: int):
        while True:
            with self.token_lock:
                now = time.time()

                self.token_window = [(t, n) for t, n in self.token_window if now - t < 60]
                used = sum(n for _, n in self.token_window)

                if used + tokens <= self.token_limit:
                    self.token_window.append((now, tokens))
                    return

                wait = 60 - (now - self.token_window[0][0])

            time.sleep(max(wait, 0.1))


    def _build_prompt(self, query: str, document: str):

        system_prompt = (
            "You are a careful semantic classification assistant."
        )

        user_prompt = f"""
Determine whether the document satisfies the query.

Return only one integer:
1 means True.
0 means False.

Do not output explanations.

Query:
{query}

Document:
{document}
"""

        return system_prompt, user_prompt


    def _call_one(self, document: str, query: str):

        system_prompt, user_prompt = self._build_prompt(query, document)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "temperature": 0,
            "max_tokens": 5,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for _ in range(5):
            try:
                resp = requests.post(self.base_url, json=payload, headers=headers, timeout=60)

                if resp.status_code == 429:
                    time.sleep(2)
                    continue

                resp.raise_for_status()
                break

            except Exception:
                time.sleep(2)

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        match = re.search(r"\b[01]\b", content)

        if match:
            return bool(int(match.group()))

        return False


    def LLM_call(self, document_contents: List[str], query_content: str) -> Dict[int, bool]:

        labels = {}
        def worker(idx):

            doc = document_contents[idx]

            # 粗略估计token
            self._acquire_token(min(len(doc)//4, 8000)+200)
            label = self._call_one(doc, query_content)
            return idx, label


        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:

            futures = [executor.submit(worker, i) for i in range(len(document_contents))]

            for future in as_completed(futures):
                idx, label = future.result()
                labels[idx] = label

        return labels
        