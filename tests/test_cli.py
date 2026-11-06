import json

from gradcam.cli import main


def test_train_predict_explain_sanity(image_folder, tmp_path):
    out = tmp_path / "run"
    results = main(["train", "--data", str(image_folder), "--out", str(out), "--arch", "resnet18",
                    "--no-pretrained", "--finetune", "--image-size", "32", "--epochs", "2", "--batch-size", "4",
                    "--exclude", "my_pets", "--workers", "0", "--device", "cpu"])
    assert results["classes"] == ["cat", "corgi", "husky"]
    assert (out / "best.pt").exists() and (out / "history.csv").exists()
    assert json.loads((out / "results.json").read_text())["val_accuracy"] == results["val_accuracy"]

    photo = image_folder / "my_pets" / "my_pets0.jpg"
    rows = main(["predict", "--checkpoint", str(out / "best.pt"), str(photo), "--device", "cpu"])
    assert len(rows[0]["top"]) == 3

    png = main(["explain", "--checkpoint", str(out / "best.pt"), str(photo), "--target", "husky",
                "--out", str(tmp_path / "h.png"), "--device", "cpu"])
    assert png.exists()

    steps = main(["sanity", "--checkpoint", str(out / "best.pt"), str(photo), "--steps", "2", "--device", "cpu"])
    assert len(steps) == 2
