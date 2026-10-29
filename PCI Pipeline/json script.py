import os
import json
from collections import Counter

# =====================================================
# CONFIGURATION
# =====================================================

DATASET_ROOT = r"B:\METI Dataset\Pavement cracks from UAV imagery-xchen-1388\Pavement cracks from UAV imagery-1000"

OUTPUT_FILE = os.path.join(
    DATASET_ROOT,
    "uav_crack_dataset_report.json"
)

CLASS_MAP = {
    "0": "transverse_crack",
    "1": "longitudinal_crack",
    "2": "alligator_crack",
    "3": "other_crack"
}

# =====================================================
# SPLIT ANALYSIS
# =====================================================

def analyse_split(split_path):

    images_dir = os.path.join(split_path, "images")
    labels_dir = os.path.join(split_path, "labels")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    images = []
    labels = []

    if os.path.exists(images_dir):
        images = [
            f for f in os.listdir(images_dir)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]

    if os.path.exists(labels_dir):
        labels = [
            f for f in os.listdir(labels_dir)
            if f.endswith(".txt")
        ]

    class_counts = Counter()

    total_objects = 0
    empty_labels = 0

    for label_file in labels:

        label_path = os.path.join(
            labels_dir,
            label_file
        )

        with open(label_path, "r", encoding="utf-8") as f:

            lines = [
                line.strip()
                for line in f.readlines()
                if line.strip()
            ]

        if len(lines) == 0:
            empty_labels += 1
            continue

        total_objects += len(lines)

        for line in lines:

            parts = line.split()

            if len(parts) >= 5:

                class_id = parts[0]

                class_name = CLASS_MAP.get(
                    class_id,
                    f"unknown_{class_id}"
                )

                class_counts[class_name] += 1

    image_basenames = {
        os.path.splitext(f)[0]
        for f in images
    }

    label_basenames = {
        os.path.splitext(f)[0]
        for f in labels
    }

    missing_labels_count = len(
        image_basenames - label_basenames
    )

    class_percentages = {}

    for class_name, count in class_counts.items():

        class_percentages[class_name] = round(
            (count / total_objects) * 100,
            2
        )

    return {
        "num_images": len(images),
        "num_label_files": len(labels),
        "total_objects": total_objects,
        "empty_labels": empty_labels,
        "missing_labels_count": missing_labels_count,
        "class_counts": dict(class_counts),
        "class_percentages": class_percentages
    }

# =====================================================
# MAIN
# =====================================================

def main():

    report = {
        "dataset_root": DATASET_ROOT,
        "annotation_format": "YOLO",
        "class_mapping": CLASS_MAP,
        "splits": {}
    }

    for split_name in ["train", "val", "test"]:

        split_path = os.path.join(
            DATASET_ROOT,
            split_name
        )

        if os.path.isdir(split_path):

            report["splits"][split_name] = analyse_split(
                split_path
            )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    print(f"\nReport saved to:\n{OUTPUT_FILE}")

# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()