from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# โหลดโมเดลและ tokenizer แบบแข็งแรง
model_path = r"C:\Users\User\Downloads\ProjectDeep\fineturn"

def load_model():
    try:
        # ตรวจสอบและเคลียร์ cache ถ้ามี
        torch.cuda.empty_cache()
        
        # ระบุ device อย่างชัดเจน
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # โหลด tokenizer และ model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        
        # ตั้งค่า padding token ถ้าจำเป็น
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("✅ โหลดโมเดลสำเร็จ!")
        return model, tokenizer, device
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        return None, None, None

model, tokenizer, device = load_model()

# ฟังก์ชันสร้างคำตอบแบบปรับปรุงแล้ว
def generate_text(prompt, max_length=200):
    if model is None or tokenizer is None:
        return "ขอโทษด้วย ระบบกำลังมีปัญหา technical issues กรุณาลองใหม่ภายหลัง"
    
    try:
        # จัดรูปแบบ prompt ให้ชัดเจน
        system_prompt = "คุณเป็นผู้ช่วยอัจฉริยะที่ตอบคำถามเกี่ยวกับประเทศไทย โดยเฉพาะจังหวัดชลบุรี\n"
        formatted_prompt = f"{system_prompt}คำถาม: {prompt}\nคำตอบ:"
        
        # Tokenize และส่งไปยัง device
        inputs = tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(device)
        
        # สร้างคำตอบ
        output = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            num_return_sequences=1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # ถอดรหัสคำตอบ
        full_response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # ตัดส่วน prompt ออก
        response = full_response.replace(formatted_prompt, "").strip()
        
        # ตรวจสอบและแก้ไขคำตอบ
        if not response or len(response) < 5:
            response = "ขออภัย ฉันไม่สามารถสร้างคำตอบที่เหมาะสมได้ในขณะนี้"
        
        # ตัดประโยคแรกเท่านั้น
        response = response.split("\n")[0].split(".")[0]
        
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Full Response: {full_response}")
        logger.info(f"Final Response: {response}")
        
        return response
    
    except Exception as e:
        logger.error(f"Error in generation: {e}")
        return "ขออภัย เกิดข้อผิดพลาดในการสร้างคำตอบ"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "กรุณากรอกข้อความที่จะถาม"})
    
    # ตรวจสอบคำถามเกี่ยวกับชลบุรีโดยตรง
    if "ชลบุรี" in user_input and ("อยู่ที่ไหน" in user_input or "ตั้งอยู่" in user_input):
        return jsonify({
            "response": "จังหวัดชลบุรีตั้งอยู่ในภาคตะวันออกของประเทศไทย ติดกับอ่าวไทย มีเมืองสำคัญเช่น พัทยา, ศรีราชา, และบางแสน เป็นจังหวัดท่องเที่ยวที่มีชื่อเสียง"
        })
    
    response = generate_text(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
