import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

from correlation_tasks import analyze_brain_correlations
from grader_contracts.correlation_tasks import BrainDataInput
from grader_contracts.pandas_tasks import TitanicInput
from pandas_tasks import analyze_titanic


ROOT = Path(__file__).resolve().parent


class Lab2Tests(unittest.TestCase):
    def test_titanic_dataset_totals(self):
        result = analyze_titanic(TitanicInput(str(ROOT / "titanic.csv")))
        self.assertEqual(result.row_count, 891)
        self.assertEqual(result.adults_over_30_count, 305)
        self.assertEqual(result.missing_by_column["Age"], 177)
        self.assertEqual(result.missing_by_column["Cabin"], 687)
        self.assertEqual(result.missing_by_column["Embarked"], 2)
        self.assertAlmostEqual(result.mean_age_by_pclass[1], 38.233440860215055)
        self.assertAlmostEqual(result.survival_rate_by_pclass[3], 119 / 491)
        self.assertEqual(result.highest_fares, [512.3292, 512.3292, 512.3292, 263.0, 263.0])
        self.assertTrue(all(type(key) is int for key in result.mean_age_by_pclass))

    def test_titanic_missing_age_boundary_and_repeated_fares(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            pd.DataFrame({
                "Pclass": [1, 1, 2, 2, 3, 3],
                "Age": [30, 40, 20, None, None, None],
                "Survived": [1, 0, 1, 1, 0, 0],
                "Fare": [5, 10, 10, 3, None, 2],
            }).to_csv(path, index=False)
            result = analyze_titanic(TitanicInput(str(path)))
        self.assertEqual(result.adults_over_30_count, 1)
        self.assertEqual(result.mean_age_by_pclass[1], 35)
        self.assertEqual(result.mean_age_by_pclass[2], 20)
        self.assertTrue(math.isnan(result.mean_age_by_pclass[3]))
        self.assertEqual(result.survival_rate_by_pclass, {1: 0.5, 2: 1.0, 3: 0.0})
        self.assertEqual(result.highest_fares, [10, 10, 5, 3, 2])
        self.assertEqual(result.missing_by_column["Age"], 3)

    def test_titanic_fewer_than_five_fares(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.csv"
            path.write_text("Pclass,Age,Survived,Fare\n2,31,1,12\n2,,0,7\n")
            result = analyze_titanic(TitanicInput(str(path)))
        self.assertEqual(result.highest_fares, [12, 7])
        self.assertEqual(result.mean_age_by_pclass, {2: 31})

    def test_brain_correlations_match_scipy_on_each_available_pair(self):
        frame = pd.read_csv(ROOT / "brainsize.txt", sep="\t", na_values="NA")
        result = analyze_brain_correlations(BrainDataInput(str(ROOT / "brainsize.txt")))
        self.assertEqual((result.men_count, result.women_count), (20, 20))
        for gender, correlations in [
            ("Male", result.men_mri_correlation),
            ("Female", result.women_mri_correlation),
        ]:
            group = frame.loc[frame["Gender"] == gender]
            self.assertEqual(set(correlations), {"FSIQ", "VIQ", "PIQ", "Weight", "Height"})
            for feature, value in correlations.items():
                pair = group[[feature, "MRI_Count"]].dropna()
                self.assertAlmostEqual(value, pearsonr(pair[feature], pair["MRI_Count"]).statistic)

    def test_strongest_correlation_uses_absolute_value_and_both_groups(self):
        frame = pd.DataFrame({
            "Gender": ["Male"] * 5 + ["Female"] * 5,
            "FSIQ": [1, 4, 2, 5, 3] * 2,
            "VIQ": [3, 1, 5, 2, 4] * 2,
            "PIQ": [2, 5, 1, 3, 4] * 2,
            "Weight": [4, 1, 3, 5, 2] * 2,
            "Height": [1, 3, 2, 5, 4, 5, 4, 3, 2, 1],
            "MRI_Count": [1, 2, 3, 4, 5] * 2,
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "brain.tsv"
            frame.to_csv(path, sep="\t", index=False)
            result = analyze_brain_correlations(BrainDataInput(str(path)))
        self.assertEqual(result.strongest_mri_feature, "Height")
        self.assertAlmostEqual(result.women_mri_correlation["Height"], -1)
        self.assertLess(abs(result.men_mri_correlation["Height"]), 1)


if __name__ == "__main__":
    unittest.main()
