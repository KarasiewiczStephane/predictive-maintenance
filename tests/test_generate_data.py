"""Tests for the sample data generation script."""

from scripts.generate_sample_data import generate_sample_data


class TestGenerateSampleData:
    """Tests for generate_sample_data function."""

    def test_default_output_shape(self) -> None:
        """Generates expected number of records."""
        df = generate_sample_data(n_equipment=5, n_days=30)
        assert len(df) == 5 * 30 * 24  # 5 equipment * 30 days * 24 hours

    def test_columns_present(self) -> None:
        """All required columns are present."""
        df = generate_sample_data(n_equipment=2, n_days=3)
        expected = {
            "timestamp",
            "equipment_id",
            "temperature",
            "vibration",
            "pressure",
            "rpm",
            "failure",
        }
        assert set(df.columns) == expected

    def test_equipment_ids(self) -> None:
        """Correct number of unique equipment IDs."""
        df = generate_sample_data(n_equipment=3, n_days=2)
        assert df["equipment_id"].nunique() == 3

    def test_failure_labels(self) -> None:
        """Failure labels are binary (0 or 1)."""
        df = generate_sample_data(n_equipment=2, n_days=30)
        assert set(df["failure"].unique()).issubset({0, 1})

    def test_custom_parameters(self) -> None:
        """Custom n_equipment and n_days produce correct output."""
        df = generate_sample_data(n_equipment=2, n_days=5)
        assert len(df) == 2 * 5 * 24
