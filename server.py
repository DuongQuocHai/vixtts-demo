import asyncio
import websockets
import os
import signal
import torch
import torchaudio
from underthesea import sent_tokenize, text_normalize
from unidecode import unidecode

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from datetime import datetime
import string
import wave
import aiofiles

# =====================
# CẤU HÌNH ĐƯỜNG DẪN VÀ THÔNG SỐ
# =====================
# Thư mục chứa model, audio mẫu và output
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
REFERENCE_AUDIO = os.path.join(MODEL_DIR, "samples/nu-luu-loat.wav")  # File audio mẫu mặc định
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHUNK_SIZE = 4 * 1024  # Kích thước mỗi chunk gửi qua websocket (bytes) 4kb

# =====================
# HÀM LOAD MÔ HÌNH TTS
# =====================
def clear_gpu_cache():
    # Giải phóng bộ nhớ GPU nếu có
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    print(f"[INFO] Bắt đầu load config từ: {xtts_config}")
    clear_gpu_cache()
    config = XttsConfig()
    config.load_json(xtts_config)
    print(f"[INFO] Khởi tạo model từ config...")
    model = Xtts.init_from_config(config)
    print(f"[INFO] Đang load checkpoint: {xtts_checkpoint}")
    model.load_checkpoint(config,
                         checkpoint_path=xtts_checkpoint,
                         vocab_path=xtts_vocab,
                         use_deepspeed=False)
    if torch.cuda.is_available():
        print("[INFO] Đưa model lên GPU...")
        model.cuda()
    print("[INFO] Model Loaded!")
    return model

# =====================
# XỬ LÝ CHUỖI, TÁCH CÂU, CHUẨN HÓA VĂN BẢN
# =====================
def get_file_name(text, max_char=50):
    # Sinh tên file từ nội dung text (dùng cho lưu tạm file WAV)
    filename = text[:max_char].lower().replace(" ", "_")
    filename = filename.translate(str.maketrans("", "", string.punctuation.replace("_", "")))
    filename = unidecode(filename)
    current_datetime = datetime.now().strftime("%m%d%H%M%S")
    return f"{current_datetime}_{filename}"

def calculate_keep_len(text, lang):
    # Tính độ dài audio cần giữ lại cho câu ngắn (tránh dư âm)
    if lang in ["ja", "zh-cn"]:
        return -1
    word_count = len(text.split())
    num_punct = text.count(".") + text.count("!") + text.count("?") + text.count(",")
    if word_count < 5:
        return 15000 * word_count + 2000 * num_punct
    elif word_count < 10:
        return 13000 * word_count + 2000 * num_punct
    return -1

def normalize_vietnamese_text(text):
    # Chuẩn hóa văn bản tiếng Việt (dùng underthesea)
    print(f"[INFO] Chuẩn hóa tiếng Việt: {text}")
    try:
        text = text_normalize(text)
        text = (
            text
            .replace("..", ".")
            .replace("!.", "!")
            .replace("?.", "?")
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace('"', "")
            .replace("'", "")
            .replace("AI", "Ây Ai")
            .replace("A.I", "Ây Ai")
        )
        return text
    except Exception as e:
        print(f"[ERROR] Lỗi khi chuẩn hóa tiếng Việt: {e}")
        print(f"[ERROR] Đầu vào: {text}")
        raise

# =====================
# HÀM CHẠY TTS (TEXT -> WAV)
# =====================
def run_tts(model, lang, tts_text, speaker_audio_file, normalize_text=True, isChunk=False):
    try:
        if model is None or not speaker_audio_file:
            raise RuntimeError("Model or reference audio not loaded!")
        output_dir = OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=speaker_audio_file,
            gpt_cond_len=model.config.gpt_cond_len,
            max_ref_length=model.config.max_ref_len,
            sound_norm_refs=model.config.sound_norm_refs,
        )
        if normalize_text and lang == "vi":
            tts_text = normalize_vietnamese_text(tts_text)

        wav_chunks = []
        if isChunk:
            if lang in ["ja", "zh-cn"]:
                tts_texts = tts_text.split("。")
            else:
                tts_texts = sent_tokenize(tts_text)
            print(f"[INFO] TTS input (chuẩn hóa): {tts_text}")
            print(f"[INFO] Ngôn ngữ synthesize: {lang}")
            for text in tts_texts:
                print(f"[DEBUG] Đang synthesize câu: {text}")
                if text.strip() == "":
                    continue
                wav_chunk = model.inference(
                    text=text,
                    language=lang,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=0.3,
                    length_penalty=1.0,
                    repetition_penalty=10.0,
                    top_k=30,
                    top_p=0.85,
                    enable_text_splitting=True,
                )
                keep_len = calculate_keep_len(text, lang)
                wav_chunk["wav"] = torch.tensor(wav_chunk["wav"][:keep_len])
                wav_chunks.append(wav_chunk["wav"])
        else:
            print(f"[INFO] TTS input (không chunk): {tts_text}")
            print(f"[INFO] Ngôn ngữ synthesize: {lang}")
            wav_chunk = model.inference(
                text=tts_text,
                language=lang,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.3,
                length_penalty=1.0,
                repetition_penalty=10.0,
                top_k=30,
                top_p=0.85,
                enable_text_splitting=True,
            )
            keep_len = calculate_keep_len(tts_text, lang)
            wav_chunk["wav"] = torch.tensor(wav_chunk["wav"][:keep_len])
            wav_chunks.append(wav_chunk["wav"])
        out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0)
        out_path = os.path.join(output_dir, f"{get_file_name(tts_text)}.wav")
        # Chuyển sang int16 trước khi lưu để wave đọc được
        out_wav_int16 = (out_wav * 32767).clamp(-32768, 32767).short()
        torchaudio.save(out_path, out_wav_int16, 24000, encoding="PCM_S", bits_per_sample=16)
        return out_path
    except Exception as e:
        print(f"[ERROR] Lỗi trong run_tts: {e}")
        print(f"[ERROR] Đầu vào: lang={lang}, tts_text={tts_text}, speaker_audio_file={speaker_audio_file}")
        raise

# =====================
# WEBSOCKET SERVER: XỬ LÝ KẾT NỐI VÀ STREAM AUDIO
# =====================
async def tts_handler(websocket):
    print(f"[INFO] Client connected: {websocket.remote_address}")
    try:
        # Nhận JSON từ client: {"text": ..., "lang": ..., "reference_audio": ..., "isChunk": ..., "isCache": ...}
        import json
        async for data in websocket:
            print(f"[INFO] Đã nhận message từ client {websocket.remote_address}: {data}")
            try:
                req = json.loads(data)
                text = req.get("text", "")
                lang = req.get("lang", "vi")
                reference_audio = req.get("reference_audio", REFERENCE_AUDIO)
                is_chunk = req.get("isChunk", False)
                is_cache = req.get("isCache", False)
            except Exception as e:
                print(f"[ERROR] Lỗi parse input từ client {websocket.remote_address}: {e}")
                await websocket.send(json.dumps({"error": "Invalid input format. Expect JSON with 'text', 'lang', 'reference_audio', 'isChunk', 'isCache'."}))
                return

            # Chạy TTS để sinh file WAV
            try:
                audio_path = run_tts(tts_model, lang, text, reference_audio, isChunk=is_chunk)
            except Exception as e:
                print(f"[ERROR] Lỗi khi chạy TTS cho client {websocket.remote_address}: {e}")
                await websocket.send(json.dumps({"error": f"TTS error: {str(e)}"}))
                return
            
            # Đọc thông tin WAV header
            with wave.open(audio_path, 'rb') as wf:
                sr = wf.getframerate()
                sample_width = wf.getsampwidth()
                channels = wf.getnchannels()
            # Gửi header JSON đầu tiên (thông tin định dạng audio)
            await websocket.send(json.dumps({
                "audio_format": "pcm",
                "sample_rate": sr,
                "sample_width": sample_width,
                "channels": channels,
                "dtype": "int16",
                "status": "start"
            }))
            # Gửi từng chunk PCM bytes từ file WAV (bỏ header 44 bytes)
            async with aiofiles.open(audio_path, mode='rb') as f:
                await f.seek(44)  # Bỏ header WAV
                while True:
                    chunk = await f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    await websocket.send(chunk)
            # Gửi thông báo hoàn thành
            await websocket.send(json.dumps({"status": "done"}))
            # Xóa file nếu không cache
            if not is_cache:
                try:
                    os.remove(audio_path)
                    print(f"[INFO] Đã xóa file output: {audio_path}")
                except Exception as e:
                    print(f"[ERROR] Không thể xóa file output: {audio_path}, lỗi: {e}")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"[INFO] Client disconnected: {websocket.remote_address}. Reason: {e}")
        pass
    except Exception as e:
        print(f"[ERROR] Lỗi không xác định với client {websocket.remote_address}: {e}")
        await websocket.send(json.dumps({"error": f"Server error: {str(e)}"}))

# =====================
# MAIN ENTRY: KHỞI ĐỘNG SERVER
# =====================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="viXTTS WebSocket TTS Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    parser.add_argument("--reference_audio", type=str, default=REFERENCE_AUDIO, help="Path to reference audio")
    args = parser.parse_args()

    print("[INFO] ====== BẮT ĐẦU KHỞI ĐỘNG SERVER ======")
    print("[INFO] Bắt đầu load mô hình TTS...")
    tts_model = load_model(
        xtts_checkpoint=os.path.join(MODEL_DIR, "model.pth"),
        xtts_config=os.path.join(MODEL_DIR, "config.json"),
        xtts_vocab=os.path.join(MODEL_DIR, "vocab.json")
    )
    print("[INFO] Đã load xong mô hình TTS!")
    print("[INFO] Đang khởi tạo WebSocket server...")

    async def main():
        try:
            async with websockets.serve(tts_handler, args.host, args.port, max_size=None, max_queue=None):
                print(f"[INFO] WebSocket server đã sẵn sàng tại ws://{args.host}:{args.port}")
                print("[INFO] Đang chờ client kết nối...")
                await asyncio.Future()  # Run forever
        except Exception as e:
            print(f"[ERROR] Lỗi khi khởi động hoặc chạy WebSocket server: {e}")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user (Ctrl+C). Terminating process group...")
        os.killpg(os.getpgrp(), signal.SIGTERM) 
