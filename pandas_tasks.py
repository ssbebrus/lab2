"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    frame = pd.read_csv(data.csv_path)
    by_class = frame.groupby("Pclass")
    return TitanicSummary(
        row_count=len(frame),
        missing_by_column={name: int(count) for name, count in frame.isna().sum().items()},
        adults_over_30_count=int((frame["Age"] > 30).sum()),
        mean_age_by_pclass={int(key): float(value) for key, value in by_class["Age"].mean().items()},
        survival_rate_by_pclass={int(key): float(value) for key, value in by_class["Survived"].mean().items()},
        highest_fares=frame["Fare"].nlargest(5).tolist(),
    )
