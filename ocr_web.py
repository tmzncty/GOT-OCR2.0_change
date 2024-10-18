import gradio as gr
import argparse
import os
import time
import torch
from PIL import Image
import asyncio
from fastapi import FastAPI
from transformers import AutoTokenizer
from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import disable_torch_init, KeywordsStoppingCriteria
from GOT.model import GOTQwenForCausalLM
from GOT.model.plug.blip_process import BlipImageEvalProcessor

class OCRProcessor:
    IM_START_TOKEN = '<img>'
    IM_END_TOKEN = '</img>'
    IMAGE_PATCH_TOKEN = '<imgpad>'

    def __init__(self, model_name, device_str):
        self.device_str = device_str
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = GOTQwenForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            pad_token_id=151643
        ).eval().to(device=device_str, dtype=torch.float16)
        self.image_processor = BlipImageEvalProcessor(image_size=1024)
        self.image_processor_high = BlipImageEvalProcessor(image_size=1024)
        self.image_token_len = 256
        self.conv = conv_templates["mpt"].copy()
        self.stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2

    def load_image(self, image_file):
        return Image.open(image_file).convert('RGB')

    def construct_query(self, task='OCR'):
        return (f'{task}: {self.IM_START_TOKEN}'
                f'{self.IMAGE_PATCH_TOKEN * self.image_token_len}'
                f'{self.IM_END_TOKEN}\n')

    async def process_image(self, image_file):
        qs = self.construct_query()
        conv = self.conv.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        inputs = self.tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).to(device=self.device_str)

        stopping_criteria = KeywordsStoppingCriteria([self.stop_str], self.tokenizer, input_ids)

        image = self.load_image(image_file)
        image_tensor = self.image_processor(image).to(device=self.device_str)
        image_tensor_high = self.image_processor_high(image).to(device=self.device_str)

        with torch.autocast("cuda", dtype=torch.float16):
            output_ids = self.model.generate(
                input_ids,
                images=[(image_tensor.unsqueeze(0).half(), image_tensor_high.unsqueeze(0).half())],
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=20,
                max_new_tokens=4096,
                stopping_criteria=[stopping_criteria]
            )

        output = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:]).strip()
        output = output[:-len(self.stop_str)] if output.endswith(self.stop_str) else output

        return output.strip()

# 全局变量，用于存储OCRProcessor实例
processor = None

def initialize_model():
    global processor
    disable_torch_init()
    model_path = "/home/tmzn/GOT-OCR2.0/GOT-OCR-2.0-master/weights"
    processor = OCRProcessor(model_path, 'cuda:0')

async def ocr_interface(image):
    if processor is None:
        return "Model not initialized. Please wait."
    
    # 如果传递的 `image` 是文件路径字符串
    image_path = image if isinstance(image, str) else image.name
    result = await processor.process_image(image_path)
    return image, result


app = FastAPI()

@app.post("/api/ocr")
async def api_ocr(image_file: str):
    if processor is None:
        return {"error": "Model not initialized. Please wait."}
    
    result = await processor.process_image(image_file)
    return {"result": result}

def launch_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# GOT-OCR Web Interface")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="filepath", label="Upload Image")
                run_button = gr.Button("Run OCR")
            with gr.Column():
                output_image = gr.Image(label="Processed Image")
                output_text = gr.Textbox(label="OCR Result", lines=10)
        
        run_button.click(fn=ocr_interface, inputs=input_image, outputs=[output_image, output_text])
    
    # 初始化模型
    initialize_model()
    
    # 启动Gradio界面，监听所有IP
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

if __name__ == "__main__":
    launch_interface()
