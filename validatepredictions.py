import glob
import cv2

folder = "predictions"

files = glob.glob("predictions/*")
assert len(files) == 933 * 2, "Incorrect number of files"
for i in range(933):
    for pred_type in ("damage", "localization"):
        path = f"{folder}/test_{pred_type}_{i:05d}_prediction.png"
        assert path in files, f"{path} is not in folder"
        try:
            img = cv2.imread(path)
        except Exception as e:
            print(f"failed to open {path}")
            raise e
        assert img.shape == (
            1024,
            1024,
            3,
        ), f"{path} shape is not (1024, 1024, 3)"  # not sure if (1024, 1024) is allowed
        assert img.dtype == "uint8", f"{path} is not uint8"
        assert img.max() <= 4, f"{path} contains values greater than 4"

print("Successful validation!")
