from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, ImageMessage, TextSendMessage
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from dotenv import load_dotenv
import logging
import time
from io import BytesIO

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 載入環境變數
load_dotenv()
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("LINE_CHANNEL_ACCESS_TOKEN 或 LINE_CHANNEL_SECRET 未設置")

# 初始化 LINE Bot
try:
    line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
    handler = WebhookHandler(LINE_CHANNEL_SECRET)
    logger.info("LINE Bot 初始化成功")
except Exception as e:
    logger.error(f"LINE Bot 初始化失敗: {str(e)}")
    raise

# 載入模型
try:
    model = load_model('temp_model/digit_recognizer.h5')
    logger.info("模型載入成功")
except Exception as e:
    logger.error(f"模型載入失敗: {str(e)}")
    raise

# 圖片預處理（從記憶體處理）
def preprocess_image(image_data):
    try:
        start_time = time.time()
        img = Image.open(BytesIO(image_data)).convert('L')
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        logger.info(f"圖片預處理完成，耗時: {time.time() - start_time:.2f}秒")
        return img_array
    except Exception as e:
        logger.error(f"圖片預處理失敗: {str(e)}")
        raise

# Webhook 路由
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    logger.info(f"收到 Webhook 請求: {body}")
    
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.error("無效簽名，請檢查 LINE_CHANNEL_SECRET")
        abort(400)
    
    return 'OK'

# 處理文字訊息
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    logger.info("收到文字訊息")
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text='請傳送一張手寫數字圖片！')
    )

# 處理圖片訊息
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    try:
        start_time = time.time()
        logger.info(f"收到圖片訊息，ID: {event.message.id}")
        
        # 獲取圖片內容
        message_content = line_bot_api.get_message_content(event.message.id)
        image_data = message_content.content
        logger.info(f"圖片下載完成，耗時: {time.time() - start_time:.2f}秒")
        
        # 預處理圖片（記憶體處理）
        img_array = preprocess_image(image_data)
        
        # 模型預測
        prediction_start = time.time()
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction, axis=1)[0]
        logger.info(f"預測結果: {predicted_digit}, 預測耗時: {time.time() - prediction_start:.2f}秒")
        
        # 回應
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f'辨識結果：{predicted_digit}')
        )
        logger.info(f"總處理耗時: {time.time() - start_time:.2f}秒")
    except LineBotApiError as e:
        logger.error(f"LINE API 錯誤: {str(e)}")
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f'圖片處理失敗 (LINE API 錯誤: {str(e)})')
        )
    except Exception as e:
        logger.error(f"圖片處理失敗: {str(e)}")
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text='圖片處理失敗，請稍後再試！')
        )

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)