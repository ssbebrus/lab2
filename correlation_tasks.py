"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    frame = pd.read_csv(data.csv_path, sep="\t", na_values="NA")
    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
    men = frame.loc[frame["Gender"] == "Male"]
    women = frame.loc[frame["Gender"] == "Female"]
    men_correlations = men[features].corrwith(men["MRI_Count"]).to_dict()
    women_correlations = women[features].corrwith(women["MRI_Count"]).to_dict()
    correlations = list(men_correlations.items()) + list(women_correlations.items())
    strongest_feature = max(
        ((name, value) for name, value in correlations if pd.notna(value)),
        key=lambda item: abs(item[1]),
    )[0]
    return BrainCorrelationSummary(
        men_count=len(men),
        women_count=len(women),
        women_mri_correlation=women_correlations,
        men_mri_correlation=men_correlations,
        strongest_mri_feature=strongest_feature,
    )
