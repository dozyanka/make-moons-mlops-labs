from moons_lab.eda import compute_eda, plot_confusion, plot_problem


def test_eda_outputs(tmp_path):
    stats, summary = compute_eda("data/raw/moons.csv")
    assert summary["rows"] == 24000
    assert summary["linear_train_accuracy"] < 0.9
    plot_problem("data/raw/moons.csv", tmp_path / "p.png")
    plot_confusion("data/raw/moons.csv", tmp_path / "c.png")
    assert (tmp_path / "p.png").exists() and (tmp_path / "c.png").exists()
