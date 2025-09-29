import gradio as gr
import json
import os
from pathlib import Path

from PIL import Image
import gradio as gr

def resize_image(image_path, scale=0.5):
    img = Image.open(image_path)
    new_size = (int(img.width * scale), int(img.height * scale))
    return img.resize(new_size, Image.LANCZOS)

class JSONLViewer:
    def __init__(self, jsonl_path, save_path="save/saved_data.jsonl"):
        self.jsonl_path = jsonl_path
        self.save_path = save_path
        self.data = self.load_jsonl()
        self.current_index = 0
        self.total_items = len(self.data)
        
    def load_jsonl(self):
        """加载JSONL文件数据"""
        try:
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                return [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            print(f"错误：未找到文件 {self.jsonl_path}")
            return []
        except Exception as e:
            print(f"加载文件时出错：{str(e)}")
            return []
    
    def save_current_item(self):
        """保存当前项到新的JSONL文件"""
        if 0 <= self.current_index < self.total_items:
            current_item = self.data[self.current_index]
            try:
                # 检查文件是否存在，如果不存在则创建
                file_exists = os.path.exists(self.save_path)
                
                with open(self.save_path, 'a', encoding='utf-8') as f:
                    # 如果是新文件，不需要添加换行符
                    if file_exists:
                        f.write('\n')
                    json.dump(current_item, f, ensure_ascii=False)
                return f"已保存第 {self.current_index + 1} 项到 {self.save_path}"
            except Exception as e:
                return f"保存失败：{str(e)}"
        return "没有可保存的项"
    
    def save_and_next(self):
        """保存当前项并获取下一项"""
        save_msg = self.save_current_item()
        # 保存后获取下一项
        return self.get_next_item() + (save_msg,)
    
    def get_prev_item(self):
        """获取上一项数据"""
        if self.total_items == 0:
            return None, "没有数据", "", "", "已是第一项"
        
        self.current_index = (self.current_index - 1) % self.total_items
        return self.get_current_item()
    
    def get_next_item(self):
        """获取下一项数据"""
        if self.total_items == 0:
            return None, "没有数据", "", "", "已是最后一项"
        
        self.current_index = (self.current_index + 1) % self.total_items
        return self.get_current_item()
    
    def get_current_item(self):
        """获取当前项数据"""
        if 0 <= self.current_index < self.total_items:
            item = self.data[self.current_index]
            # 获取图片路径
            image_path = item["image"]
            # 检查图片文件是否存在
            # if image_path and not os.path.exists(image_path):
            #     print(f"图片文件不存在: {image_path}")
            #     image_path = None  # 如果图片不存在，不显示
            
            question = item.get('question', "无问题信息")
            answer = item.get('answer', "无答案信息")
            predict = item.get('predict', "无预测信息")
            status = f"当前项：{self.current_index + 1}/{self.total_items}"
            
            return f"/data2/qinxb/qxb_task/{image_path}", question, answer, predict, status
        return None, "无效索引", "", "", "错误"

def create_interface(jsonl_path):
    """创建Gradio界面"""
    viewer = JSONLViewer(jsonl_path)
    
    with gr.Blocks(title="筛选数据") as interface:
        gr.Markdown("# 数据筛选")
        
        with gr.Row():
            # 左侧显示图片
            with gr.Column(scale=1):
                image = gr.Image(label="图片", type="filepath",height=800, width=600)
            
            # 右侧显示文本信息
            with gr.Column(scale=1):
                status = gr.Textbox(label="状态", interactive=False)
                question = gr.Textbox(label="问题", lines=3,interactive=True)
                answer = gr.Textbox(label="答案", lines=4,interactive=True)
                predict = gr.Textbox(label="预测", lines=6)
                save_status = gr.Textbox(label="保存状态", interactive=False)
        
        with gr.Row():
            prev_btn = gr.Button("上一个")
            next_btn = gr.Button("下一个")
            save_btn = gr.Button("保留")
        
        # 加载初始数据
        def load_initial():
            return viewer.get_current_item()
        
        # 绑定按钮事件
        prev_btn.click(
            fn=viewer.get_prev_item,
            outputs=[image, question, answer, predict, status]
        )
        
        next_btn.click(
            fn=viewer.get_next_item,
            outputs=[image, question, answer, predict, status]
        )
        
        # save_btn.click(
        #     fn=viewer.save_current_item,
        #     outputs=[save_status]
        # )
        save_btn.click(
            fn=viewer.save_and_next,
            outputs=[image, question, answer, predict, status, save_status]
        )
        
        # 初始化界面
        interface.load(
            fn=load_initial,
            outputs=[image, question, answer, predict, status]
        )
    
    return interface

if __name__ == "__main__":
    # 请替换为您的JSONL文件路径
    jsonl_file_path = ""  # 用户需要修改为实际的JSONL文件路径
    app = create_interface(jsonl_file_path)
    app.launch(server_name="0.0.0.0", server_port=8199,allowed_paths=[""])
