# 最好别用这个跑，很多逻辑还需要改

import argparse
import os
import concurrent.futures
from io import BytesIO
import time  # 引入时间库来计时

import torch
from PIL import Image
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import disable_torch_init
from GOT.model import *
from GOT.model.plug.blip_process import BlipImageEvalProcessor
from GOT.demo.process_results import punctuation_dict
from GOT.utils.utils import KeywordsStoppingCriteria

# 设置环境变量
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

def eval_image(image_file, args, model, tokenizer, image_processor, image_processor_high, use_im_start_end, image_token_len, stop_str, device_str):
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
    input_ids = torch.as_tensor(inputs.input_ids).to(device=device_str)

    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    image_1 = image.copy()
    image_tensor = image_processor(image).to(device=device_str)
    image_tensor_1 = image_processor_high(image_1).to(device=device_str)

    # 构造输出路径
    output_dir = os.path.join(args.output_dir, os.path.relpath(os.path.dirname(image_file), args.image_dir))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_file))[0]}.txt")
    
    # 如果目标文件已经存在，跳过处理
    if os.path.exists(output_path):
        print(f"{output_path} 已存在，跳过处理。")
        return

    with torch.autocast("cuda", dtype=torch.float16):
        output_ids = model.generate(
            input_ids,
            images=[(image_tensor.unsqueeze(0).half(), image_tensor_1.unsqueeze(0).half())],
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

            # 保存结果
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(outputs)

def process_image_batch(image_files, args, device):
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    device_str = f'cuda:{device}'
    model = GOTQwenForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        pad_token_id=151643
    ).eval().to(device=device_str, dtype=torch.float16)

    image_processor = BlipImageEvalProcessor(image_size=1024)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)

    use_im_start_end = True
    image_token_len = 256

    conv_mode = "mpt"
    args.conv_mode = conv_mode
    conv = conv_templates[args.conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    start_time = time.time()  # 记录开始时间

    with tqdm(total=len(image_files), desc=f"Processing images on GPU {device}") as pbar:
        for image_file in image_files:
            eval_image(
                image_file,
                args,
                model,
                tokenizer,
                image_processor,
                image_processor_high,
                use_im_start_end,
                image_token_len,
                stop_str,
                device_str
            )
            pbar.update(1)

    end_time = time.time()  # 记录结束时间
    print(f"GPU {device} 完成时间: {end_time - start_time:.2f} 秒")

def split_list(lst, n):
    avg_size = len(lst) // n
    remainder = len(lst) % n
    return [lst[i * avg_size + min(i, remainder):(i + 1) * avg_size + min(i + 1, remainder)] for i in range(n)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="/mnt/disk1/GOT-OCR2.0/weight")
    parser.add_argument("--image-dir", type=str, default="/mnt/disk1/spilt", help="Directory containing images")
    parser.add_argument("--output-dir", type=str, default="/mnt/disk1/txt", help="Directory to save OCR results")
    parser.add_argument("--type", default="ocr", type=str)
    parser.add_argument("--box", type=str, default='')
    parser.add_argument("--color", type=str, default='')
    parser.add_argument("--render", type=bool, default=True, help="Enable rendering by default")
    parser.add_argument("--num-processes-per-gpu", type=int, default=3, help="Number of parallel processes per GPU")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs to use")

    args = parser.parse_args()

    image_files = []
    for root, _, files in os.walk(args.image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_files.append(os.path.join(root, file))

    image_batches = split_list(image_files, args.num_gpus * args.num_processes_per_gpu)

    total_start_time = time.time()  # 总时间开始计时
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_gpus * args.num_processes_per_gpu) as executor:
        futures = [executor.submit(process_image_batch, batch, args, i % args.num_gpus) for i, batch in enumerate(image_batches)]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    total_end_time = time.time()  # 总时间结束计时
    total_images = len(image_files)
    print(f"程序总运行时间: {total_end_time - total_start_time:.2f} 秒")
    print(f"平均每张图片用时: {(total_end_time - total_start_time) / total_images:.2f} 秒")
