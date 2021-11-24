import requests
import os

cate= ['laptop', 'bike', 'book', 'bottle', 'camera', 'cereal_box', 'chair', 'cup', 'shoe']
save_dir = '/media/lang/My Passport/Dataset/Objectron'
public_url = "https://storage.googleapis.com/objectron"
mode = 'train'
for c in cate:
    blob_path = public_url + "/v1/index/"+ c +"_annotations_" + mode
    video_ids = requests.get(blob_path).text
    video_ids = video_ids.split('\n')

    # Download the first ten videos in cup test dataset
    for i in range(len(video_ids)):
        print(video_ids[i])
        video_filename = public_url + "/videos/" + video_ids[i] + "/video.MOV"
        metadata_filename = public_url + "/videos/" + video_ids[i] + "/geometry.pbdata"
        annotation_filename = public_url + "/annotations/" + video_ids[i] + ".pbdata"
        # video.content contains the video file.
        video = requests.get(video_filename)
        metadata = requests.get(metadata_filename)
        annotation = requests.get(annotation_filename)

        os.makedirs(os.path.join(save_dir, 'videos', mode, c), exist_ok=True)
        file = open(os.path.join(save_dir, 'videos', mode, c, video_ids[i].replace('/', '-') + ".MOV"), "wb")
        file.write(video.content)
        file.close()
        file_meta = open(os.path.join(save_dir, 'videos', mode, c, video_ids[i].replace('/', '-') + "-geometry.pbdata"), "wb")
        file_meta.write(metadata.content)
        file_meta.close()
        file_an = open(os.path.join(save_dir, 'videos', mode, c, video_ids[i].replace('/', '-') + "-annotation.pbdata"), "wb")
        file_an.write(annotation.content)
        file_an.close()