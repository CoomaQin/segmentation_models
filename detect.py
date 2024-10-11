import argparse as ap
import os
import cv2
import numpy as np
from tqdm import tqdm
import time
import torchvision.transforms.functional as F
import torch
import random

from utility.fs_handler import get_filename_from_path
from mytransforms import seg_train_transform
from model import BASModel


def detect(input_path, output_path, vname, weight_path, write_video=False):
    video_path = input_path
    if not os.path.isfile(input_path):
        video_path = os.path.join(input_path, vname)
    video = cv2.VideoCapture(video_path)
    CODEC_fourcc = "mp4v"
    fps = video.get(cv2.CAP_PROP_FPS)
    h = 640
    w = 640
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    if write_video:
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*CODEC_fourcc), fps, (w, h))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = BASModel("DeepLabV3", "resnext50_32x4d", in_channels=3, obj_classes=2, area_classes=4)
    model.load_state_dict(torch.load(weight_path, weights_only=True)["state_dict"])
    model.to(device)
    model.eval()

    idx = 0
    exe_time = []
    with tqdm(total=frame_count) as pbar:
        while True:
            ret, frame = video.read()
            pbar.update(1)
            idx += 1
            # if idx % 10 > 0:
            #     continue
            if ret == True:
                t0 = time.time()
                # format inputs
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                inputs = F.to_pil_image(frame).convert('RGB')
                inputs = seg_train_transform(inputs)
                inputs = inputs.float().unsqueeze(0).to(device)
                # inference
                output = model(inputs)
                t1 = time.time()
                output = output.contiguous()
                obj_output = output[:, :2, :, :]
                area_output = output[:, 2:, :, :]
                area_pred = (torch.argmax(torch.squeeze(area_output), dim=0)).data.cpu().numpy()
                obj_pred = (torch.argmax(torch.squeeze(obj_output), dim=0)).data.cpu().numpy()
                # visualize
                area_mask, obj_mask = np.zeros((h, w, 3)), np.zeros((h, w, 3))
                area_mask[area_pred == 1] = (0,0,204)
                area_mask[area_pred == 2] = (0,102,204)
                area_mask[area_pred == 3] = (0,204,204)
                obj_mask[obj_pred == 1] = (204,102,0)
                tres_pred = obj_pred * area_pred
                combined_frame = 0.5 * frame + 0.25 * area_mask + 0.25 * obj_mask
                combined_frame[tres_pred == 3] = (0,255,0)
                combined_frame[tres_pred == 2] = (0,255,255)
                combined_frame[tres_pred == 1] = (0,0,255)

                tres_found = False
                msg, color = "", (0,0,0)
                tres = list(np.unique(tres_pred))
                if 1 in tres:
                    msg, color = "Level-1 Trespasser", (0,0,255)
                elif 2 in tres:
                    msg, color = "Level-2 Trespasser", (0,255,255)
                elif 3 in tres:
                    msg, color = "Level-3 Trespasser", (0,255,0)  
                tres_found = (msg != "")
                # combined_frame = cv2.putText(combined_frame, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
                
                output_frame = combined_frame.astype(np.uint8)
                exe_time.append(1 / (t1 - t0))
                if write_video:
                    writer.write(output_frame)
                elif tres_found:
                    snapshot_path = os.path.join(output_path, f"{vname}_{idx}.png")
                    cv2.imwrite(snapshot_path, output_frame)
            else:
                if write_video:
                    writer.release()
                break
        print(f"mean FPS {np.mean(exe_time):.1f}")


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("-i", "--input", help="input path")
    parser.add_argument("-o", "--output", help="output dir")
    parser.add_argument('-w',"--weight", help="weight path")
    args = vars(parser.parse_args())
    print(args)
    if os.path.isfile(args["input"]):
        vname = get_filename_from_path(args["input"])
        detect(args["input"], args["output"], vname, args["weight"], write_video=True)
    else:
        video_list = []
        for (_, _, filenames) in os.walk(args["input"]):
            for file in filenames:
                video_list.append(file)
            break
        for idx, v in enumerate(video_list):
            output_path = os.path.join(args["output"], v)
            print(f"{idx}/{len(video_list)}")
            detect(args["input"], args["output"], v, args["weight"], write_video=False)