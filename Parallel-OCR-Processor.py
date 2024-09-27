import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from GOT.model import *
from GOT.utils.utils import KeywordsStoppingCriteria

from PIL import Image
import requests
from io import BytesIO
from GOT.model.plug.blip_process import BlipImageEvalProcessor
import re
from GOT.demo.process_results import punctuation_dict

from tqdm import tqdm  # 添加进度条库
import concurrent.futures

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'
translation_table = str.maketrans(punctuation_dict)

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def eval_image(image_file, args, model, tokenizer, image_processor, image_processor_high, use_im_start_end, image_token_len, stop_str):
    # 在生成 input_ids 后初始化 stopping_criteria
    conv_mode = "mpt"
    args.conv_mode = conv_mode
    conv = conv_templates[args.conv_mode].copy()
    keywords = [stop_str]
    
    image = load_image(image_file)
    w, h = image.size

    if args.type == 'format':
        qs = 'OCR with format: '
    else:
        qs = 'OCR: '

    if args.box:
        bbox = eval(args.box)
        if len(bbox) == 2:
            bbox[0] = int(bbox[0]/w*1000)
            bbox[1] = int(bbox[1]/h*1000)
        if len(bbox) == 4:
            bbox[0] = int(bbox[0]/w*1000)
            bbox[1] = int(bbox[1]/h*1000)
            bbox[2] = int(bbox[2]/w*1000)
            bbox[3] = int(bbox[3]/h*1000)
        if args.type == 'format':
            qs = str(bbox) + ' ' + 'OCR with format: '
        else:
            qs = str(bbox) + ' ' + 'OCR: '

    if args.color:
        if args.type == 'format':
            qs = '[' + args.color + ']' + ' ' + 'OCR with format: '
        else:
            qs = '[' + args.color + ']' + ' ' + 'OCR: '

    if use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs 
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    image_1 = image.copy()
    image_tensor = image_processor(image)
    image_tensor_1 = image_processor_high(image_1)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids,
            images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
            do_sample=False,
            num_beams=1,
            no_repeat_ngram_size=20,
            max_new_tokens=4096,
            stopping_criteria=[stopping_criteria]
        )

        if args.render:
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            print(f"Generated OCR outputs for {image_file}: {outputs}")

            # 实时保存输出到对应的 txt 文件
            output_dir = os.path.join(args.output_dir, os.path.relpath(os.path.dirname(image_file), args.image_dir))
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_file))[0]}.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(outputs)
                print(f"Saving OCR result to: {output_path}")
                print(f"OCR content: {outputs}")


def process_image_batch(image_files, args):
    # 模型初始化
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = GOTQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=151643).eval()
    model.to(device='cuda', dtype=torch.bfloat16)

    image_processor = BlipImageEvalProcessor(image_size=1024)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)

    use_im_start_end = True
    image_token_len = 256

    conv_mode = "mpt"
    args.conv_mode = conv_mode
    conv = conv_templates[args.conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    for image_file in image_files:
        # 修改这里：移除额外的参数
        eval_image(image_file, args, model, tokenizer, image_processor, image_processor_high, use_im_start_end, image_token_len, stop_str)


def split_list(lst, n):
    """将列表分割成n个子列表"""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="/mnt/disk1/GOT-OCR2.0/weight")
    parser.add_argument("--image-dir", type=str, default="/mnt/disk1/testgot", help="Directory containing images")
    parser.add_argument("--output-dir", type=str, default="/mnt/disk1/testgot_output", help="Directory to save OCR results")
    parser.add_argument("--type", default="ocr", type=str)
    parser.add_argument("--box", type=str, default='')
    parser.add_argument("--color", type=str, default='')
    parser.add_argument("--render", type=bool, default=True, help="Enable rendering by default")
    parser.add_argument("--num-processes", type=int, default=3, help="Number of parallel processes")

    args = parser.parse_args()

    # 获取所有图片文件
    image_files = []
    for root, _, files in os.walk(args.image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_files.append(os.path.join(root, file))

    # 将图像文件分割成多个子列表
    image_batches = split_list(image_files, args.num_processes)

    # 使用concurrent.futures来创建多进程
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        futures = [executor.submit(process_image_batch, batch, args) for batch in image_batches]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")
