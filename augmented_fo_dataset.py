import fiftyone as fo

name = "coco_diesel_engine"

# The directory containing the source images
data_path = input("/path/to/dataset/images: ") # "/home/opendr/project-4-at-2022-11-08-12-58-d9d5f8c2/images"

# The path to the COCO labels JSON file
labels_path = input("/path/to/dataset/annotations.json: ") # "/home/opendr/project-4-at-2022-11-08-12-58-d9d5f8c2/result.json"

# Import the dataset
dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=labels_path,
        name=name,
)

session = fo.launch_app(dataset)

session.wait()