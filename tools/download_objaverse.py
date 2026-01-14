import objaverse.xl as oxl

download_dir = "~/scratch/objaverse"

annotations = oxl.get_annotations(
    download_dir= download_dir
)
print(annotations)