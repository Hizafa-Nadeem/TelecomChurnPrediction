"""Load test for the churn API (target: p95 latency < 100 ms).

    make serve                 # in one terminal
    make loadtest              # in another

Customers are random but valid under docs/data_contract.md, so no real data is needed.
"""

import random

from locust import HttpUser, between, task

RATES = {"day": 0.17, "eve": 0.085, "night": 0.045, "intl": 0.27}  # charge per minute
rng = random.Random(42)


def customer() -> dict[str, object]:
    row: dict[str, object] = {
        "state": str(rng.randint(0, 50)),
        "area_code": rng.choice(["408", "415", "510"]),
        "account_length": rng.randint(1, 240),
        "international_plan": rng.choice([0, 1]),
        "voice_mail_plan": rng.choice([0, 1]),
        "number_vmail_messages": rng.randint(0, 50),
        "number_customer_service_calls": rng.randint(0, 9),
    }
    for period, rate in RATES.items():
        minutes = round(
            rng.uniform(0, 350 if period == "day" else 20 if period == "intl" else 300), 1
        )
        row[f"total_{period}_minutes"] = minutes
        row[f"total_{period}_calls"] = rng.randint(0, 150)
        row[f"total_{period}_charge"] = round(minutes * rate, 2)
    return row


class ChurnUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(9)
    def predict(self) -> None:
        self.client.post("/predict", json=customer(), name="/predict")

    @task(1)
    def predict_batch(self) -> None:
        batch = {"customers": [customer() for _ in range(10)]}
        self.client.post("/predict/batch", json=batch, name="/predict/batch (10)")
